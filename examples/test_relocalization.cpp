/**
 * @file test_relocalization.cpp
 * @brief Test coarse-to-fine ICP relocalization using real recorded data
 *
 * Tests the global localization pipeline with:
 * 1. Loading a pre-built map from PLY file
 * 2. Loading query scans from recorded data
 * 3. Applying random offsets to simulate unknown initial position
 * 4. Using coarse-to-fine ICP to recover the true pose
 */

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
#endif

#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "slam/types.hpp"
#include "slam/icp.hpp"
#include "slam/icp_accelerated.hpp"
#include "slam/global_localization.hpp"

using namespace slam;

//=============================================================================
// Utility Functions
//=============================================================================

std::vector<V3D> loadPlyFile(const std::string& filename) {
    std::vector<V3D> points;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filename << std::endl;
        return points;
    }

    std::string line;
    int vertex_count = 0;
    bool header_done = false;
    bool is_binary = false;

    // Parse header
    while (std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos) {
            sscanf(line.c_str(), "element vertex %d", &vertex_count);
        }
        if (line.find("format binary") != std::string::npos) {
            is_binary = true;
        }
        if (line == "end_header") {
            header_done = true;
            break;
        }
    }

    if (!header_done || vertex_count == 0) {
        return points;
    }

    points.reserve(vertex_count);

    if (is_binary) {
        // Binary format: x, y, z, intensity (float32 each)
        for (int i = 0; i < vertex_count; i++) {
            float x, y, z, intensity;
            file.read(reinterpret_cast<char*>(&x), sizeof(float));
            file.read(reinterpret_cast<char*>(&y), sizeof(float));
            file.read(reinterpret_cast<char*>(&z), sizeof(float));
            file.read(reinterpret_cast<char*>(&intensity), sizeof(float));
            if (file.good()) {
                points.emplace_back(x, y, z);
            }
        }
    } else {
        // ASCII format
        for (int i = 0; i < vertex_count; i++) {
            float x, y, z;
            if (file >> x >> y >> z) {
                points.emplace_back(x, y, z);
                // Skip intensity if present
                file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
        }
    }

    return points;
}

struct RecordedPointCloud {
    uint64_t timestamp_ns;
    std::vector<V3D> points;
};

struct RecordedIMU {
    uint64_t timestamp_ns;
    V3D accel;
    V3D gyro;
};

bool loadRecordedScans(const std::string& filename,
                       std::vector<RecordedPointCloud>& clouds,
                       int max_frames = 50) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open recording: " << filename << std::endl;
        return false;
    }

    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x534C414D) {
        std::cerr << "Invalid recording format (magic=" << std::hex << magic << ")" << std::endl;
        return false;
    }

    // Accumulate packets into complete scans (100ms period like replay_slam)
    const double SCAN_PERIOD_NS = 100e6;  // 100ms in nanoseconds
    RecordedPointCloud current_scan;
    uint64_t scan_start_time = 0;
    int frame_count = 0;
    int total_packets = 0;

    while (file.good() && !file.eof()) {
        uint8_t msg_type;
        file.read(reinterpret_cast<char*>(&msg_type), 1);
        if (file.eof() || !file.good()) break;

        if (msg_type == 1) {  // Point cloud packet
            uint64_t timestamp_ns;
            file.read(reinterpret_cast<char*>(&timestamp_ns), sizeof(timestamp_ns));

            uint32_t num_points;
            file.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));

            // Initialize scan start time
            if (scan_start_time == 0) {
                scan_start_time = timestamp_ns;
                current_scan.timestamp_ns = timestamp_ns;
                current_scan.points.reserve(20000);
            }

            // Read points
            for (uint32_t i = 0; i < num_points; i++) {
                float x, y, z;
                uint8_t refl;
                file.read(reinterpret_cast<char*>(&x), sizeof(x));
                file.read(reinterpret_cast<char*>(&y), sizeof(y));
                file.read(reinterpret_cast<char*>(&z), sizeof(z));
                file.read(reinterpret_cast<char*>(&refl), sizeof(refl));

                double dist = std::sqrt(x*x + y*y + z*z);
                if (dist > 0.5 && dist < 50.0) {
                    current_scan.points.emplace_back(x, y, z);
                }
            }
            total_packets++;

            // Check if scan period is complete
            if ((timestamp_ns - scan_start_time) >= SCAN_PERIOD_NS &&
                current_scan.points.size() > 1000) {
                clouds.push_back(std::move(current_scan));
                frame_count++;

                // Reset for next scan
                current_scan = RecordedPointCloud();
                current_scan.points.reserve(20000);
                scan_start_time = timestamp_ns;
                current_scan.timestamp_ns = timestamp_ns;

                if (frame_count >= max_frames) break;
            }

        } else if (msg_type == 2) {  // IMU - skip
            uint64_t ts;
            float ax, ay, az, gx, gy, gz;
            file.read(reinterpret_cast<char*>(&ts), sizeof(ts));
            file.read(reinterpret_cast<char*>(&ax), sizeof(ax));
            file.read(reinterpret_cast<char*>(&ay), sizeof(ay));
            file.read(reinterpret_cast<char*>(&az), sizeof(az));
            file.read(reinterpret_cast<char*>(&gx), sizeof(gx));
            file.read(reinterpret_cast<char*>(&gy), sizeof(gy));
            file.read(reinterpret_cast<char*>(&gz), sizeof(gz));
        } else {
            break;  // Unknown message type
        }
    }

    std::cout << "  Read " << total_packets << " packets, assembled " << frame_count << " complete scans\n";
    return !clouds.empty();
}

M4D makePose(double x, double y, double z, double yaw) {
    M3D R = Eigen::AngleAxisd(yaw, V3D::UnitZ()).toRotationMatrix();
    M4D pose = M4D::Identity();
    pose.block<3,3>(0,0) = R;
    pose(0, 3) = x;
    pose(1, 3) = y;
    pose(2, 3) = z;
    return pose;
}

std::vector<V3D> transformPoints(const std::vector<V3D>& points, const M4D& T) {
    std::vector<V3D> result;
    result.reserve(points.size());
    M3D R = T.block<3,3>(0,0);
    V3D t = T.block<3,1>(0,3);
    for (const auto& p : points) {
        result.push_back(R * p + t);
    }
    return result;
}

//=============================================================================
// Coarse-to-Fine ICP Relocalization
//=============================================================================

struct RelocalizationConfig {
    // Coarse stage (fast, rough alignment)
    double coarse_voxel_size = 0.5;       // Downsample to 50cm voxels
    double coarse_max_dist = 3.0;          // 3m correspondence distance
    int coarse_iterations = 20;

    // Medium stage (refine alignment)
    double medium_voxel_size = 0.2;        // 20cm voxels
    double medium_max_dist = 1.0;          // 1m correspondence distance
    int medium_iterations = 30;

    // Fine stage (precise alignment)
    double fine_voxel_size = 0.1;          // 10cm voxels
    double fine_max_dist = 0.5;            // 50cm correspondence distance
    int fine_iterations = 50;

    // Success thresholds
    double min_fitness = 0.3;              // At least 30% inliers
    double convergence_threshold = 1e-6;
};

struct RelocalizationResult {
    M4D pose = M4D::Identity();
    double fitness_score = 0.0;
    double rmse = 0.0;
    bool converged = false;
    double coarse_time_ms = 0.0;
    double medium_time_ms = 0.0;
    double fine_time_ms = 0.0;
    double total_time_ms = 0.0;
};

RelocalizationResult coarseToFineRelocalize(
    const std::vector<V3D>& query_scan,
    const std::vector<V3D>& map_points,
    const M4D& initial_guess,
    const RelocalizationConfig& config = RelocalizationConfig()) {

    RelocalizationResult result;
    M4D current_pose = initial_guess;

    auto total_start = std::chrono::high_resolution_clock::now();

    //=========================================================================
    // Stage 1: Coarse ICP
    //=========================================================================
    {
        auto stage_start = std::chrono::high_resolution_clock::now();

        auto query_down = voxelDownsample(query_scan, config.coarse_voxel_size);
        auto map_down = voxelDownsample(map_points, config.coarse_voxel_size);

        ICPConfig icp_config;
        icp_config.method = ICPMethod::POINT_TO_POINT;
        icp_config.max_iterations = config.coarse_iterations;
        icp_config.max_correspondence_dist = config.coarse_max_dist;
        icp_config.convergence_threshold = 1e-4;
        icp_config.fitness_threshold = 0.1;

#ifdef HAS_NANOFLANN
        ICPAccelerated icp(icp_config);
#else
        ICP icp(icp_config);
#endif
        auto icp_result = icp.align(query_down, map_down, current_pose);
        current_pose = icp_result.transformation;

        auto stage_end = std::chrono::high_resolution_clock::now();
        result.coarse_time_ms = std::chrono::duration<double, std::milli>(
            stage_end - stage_start).count();

        std::cout << "  Coarse: fitness=" << std::fixed << std::setprecision(3)
                  << icp_result.fitness_score
                  << " rmse=" << icp_result.rmse
                  << " (" << result.coarse_time_ms << "ms)\n";

        if (icp_result.fitness_score < 0.05) {
            std::cout << "  Coarse stage failed - no good correspondences\n";
            return result;
        }
    }

    //=========================================================================
    // Stage 2: Medium ICP
    //=========================================================================
    {
        auto stage_start = std::chrono::high_resolution_clock::now();

        auto query_down = voxelDownsample(query_scan, config.medium_voxel_size);
        auto map_down = voxelDownsample(map_points, config.medium_voxel_size);

        ICPConfig icp_config;
        icp_config.method = ICPMethod::POINT_TO_POINT;
        icp_config.max_iterations = config.medium_iterations;
        icp_config.max_correspondence_dist = config.medium_max_dist;
        icp_config.convergence_threshold = 1e-5;
        icp_config.fitness_threshold = 0.2;

#ifdef HAS_NANOFLANN
        ICPAccelerated icp(icp_config);
#else
        ICP icp(icp_config);
#endif
        auto icp_result = icp.align(query_down, map_down, current_pose);
        current_pose = icp_result.transformation;

        auto stage_end = std::chrono::high_resolution_clock::now();
        result.medium_time_ms = std::chrono::duration<double, std::milli>(
            stage_end - stage_start).count();

        std::cout << "  Medium: fitness=" << std::fixed << std::setprecision(3)
                  << icp_result.fitness_score
                  << " rmse=" << icp_result.rmse
                  << " (" << result.medium_time_ms << "ms)\n";
    }

    //=========================================================================
    // Stage 3: Fine ICP
    //=========================================================================
    {
        auto stage_start = std::chrono::high_resolution_clock::now();

        auto query_down = voxelDownsample(query_scan, config.fine_voxel_size);
        auto map_down = voxelDownsample(map_points, config.fine_voxel_size);

        ICPConfig icp_config;
        icp_config.method = ICPMethod::POINT_TO_POINT;
        icp_config.max_iterations = config.fine_iterations;
        icp_config.max_correspondence_dist = config.fine_max_dist;
        icp_config.convergence_threshold = config.convergence_threshold;
        icp_config.fitness_threshold = config.min_fitness;

#ifdef HAS_NANOFLANN
        ICPAccelerated icp(icp_config);
#else
        ICP icp(icp_config);
#endif
        auto icp_result = icp.align(query_down, map_down, current_pose);

        result.pose = icp_result.transformation;
        result.fitness_score = icp_result.fitness_score;
        result.rmse = icp_result.rmse;
        result.converged = icp_result.converged;

        auto stage_end = std::chrono::high_resolution_clock::now();
        result.fine_time_ms = std::chrono::duration<double, std::milli>(
            stage_end - stage_start).count();

        std::cout << "  Fine:   fitness=" << std::fixed << std::setprecision(3)
                  << icp_result.fitness_score
                  << " rmse=" << icp_result.rmse
                  << " (" << result.fine_time_ms << "ms)\n";
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(
        total_end - total_start).count();

    return result;
}

//=============================================================================
// Main Test
//=============================================================================

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <map.ply> <recording.bin> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --offset <x> <y> <yaw>  Apply position offset for testing\n";
    std::cout << "  --random                Use random offset\n";
    std::cout << "  --frames <n>            Number of frames to average (default: 10)\n";
    std::cout << "  --help                  Show this help\n";
}

int main(int argc, char** argv) {
    std::cout << "============================================\n";
    std::cout << "  Coarse-to-Fine ICP Relocalization Test\n";
    std::cout << "============================================\n\n";

    if (argc < 3 || std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return argc < 3 ? 1 : 0;
    }

    std::string map_file = argv[1];
    std::string recording_file = argv[2];
    double offset_x = 0.5, offset_y = 0.3, offset_yaw = M_PI / 6;  // 30 degrees
    bool random_offset = false;
    int num_frames = 10;

    // Parse arguments
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--offset" && i + 3 < argc) {
            offset_x = std::atof(argv[++i]);
            offset_y = std::atof(argv[++i]);
            offset_yaw = std::atof(argv[++i]) * M_PI / 180.0;
        } else if (arg == "--random") {
            random_offset = true;
        } else if (arg == "--frames" && i + 1 < argc) {
            num_frames = std::atoi(argv[++i]);
        }
    }

    // Random offset generation
    if (random_offset) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist_pos(-1.0, 1.0);
        std::uniform_real_distribution<> dist_yaw(-M_PI/4, M_PI/4);
        offset_x = dist_pos(gen);
        offset_y = dist_pos(gen);
        offset_yaw = dist_yaw(gen);
    }

    // Load map
    std::cout << "Loading map: " << map_file << std::endl;
    auto map_points = loadPlyFile(map_file);
    if (map_points.empty()) {
        std::cerr << "Failed to load map!" << std::endl;
        return 1;
    }
    std::cout << "  Map points: " << map_points.size() << std::endl;

    // Load recorded scans
    std::cout << "\nLoading recorded scans: " << recording_file << std::endl;
    std::vector<RecordedPointCloud> scans;
    if (!loadRecordedScans(recording_file, scans, num_frames * 10)) {
        std::cerr << "Failed to load recording!" << std::endl;
        return 1;
    }
    std::cout << "  Loaded " << scans.size() << " frames\n";

    // Accumulate points from first N frames (simulates a single query scan)
    std::cout << "\nAccumulating query scan from " << num_frames << " frames...\n";
    std::vector<V3D> query_scan;
    for (int i = 0; i < std::min(num_frames, static_cast<int>(scans.size())); i++) {
        for (const auto& pt : scans[i].points) {
            query_scan.push_back(pt);
        }
    }
    query_scan = voxelDownsample(query_scan, 0.05);  // 5cm voxels
    std::cout << "  Query scan points: " << query_scan.size() << std::endl;

    // Apply offset to query (simulates unknown initial position)
    std::cout << "\nApplying offset:\n";
    std::cout << "  X offset: " << offset_x << " m\n";
    std::cout << "  Y offset: " << offset_y << " m\n";
    std::cout << "  Yaw offset: " << (offset_yaw * 180 / M_PI) << " deg\n";

    M4D offset_pose = makePose(offset_x, offset_y, 0, offset_yaw);
    M4D true_pose = M4D::Identity();  // Query is actually at origin

    // Transform query to simulate being at offset position
    auto query_transformed = transformPoints(query_scan, offset_pose.inverse());

    // Run relocalization
    std::cout << "\nRunning coarse-to-fine ICP relocalization...\n";

    RelocalizationConfig config;
    config.coarse_voxel_size = 0.5;
    config.medium_voxel_size = 0.2;
    config.fine_voxel_size = 0.1;

    // Start with bad initial guess (at origin, but query is at offset)
    M4D initial_guess = M4D::Identity();

    auto result = coarseToFineRelocalize(query_transformed, map_points, initial_guess, config);

    // Evaluate result
    std::cout << "\n============================================\n";
    std::cout << "  Results\n";
    std::cout << "============================================\n";

    V3D recovered_pos = result.pose.block<3,1>(0,3);
    double recovered_yaw = std::atan2(result.pose(1,0), result.pose(0,0));

    V3D true_offset_pos(offset_x, offset_y, 0);
    double pos_error = (recovered_pos - true_offset_pos).norm();
    double yaw_error = std::abs(recovered_yaw - offset_yaw);
    if (yaw_error > M_PI) yaw_error = 2 * M_PI - yaw_error;

    std::cout << "\nRecovered pose:\n";
    std::cout << "  Position: [" << std::fixed << std::setprecision(3)
              << recovered_pos.x() << ", " << recovered_pos.y() << ", " << recovered_pos.z() << "]\n";
    std::cout << "  Yaw: " << (recovered_yaw * 180 / M_PI) << " deg\n";

    std::cout << "\nTrue pose (offset):\n";
    std::cout << "  Position: [" << offset_x << ", " << offset_y << ", 0.0]\n";
    std::cout << "  Yaw: " << (offset_yaw * 180 / M_PI) << " deg\n";

    std::cout << "\nError:\n";
    std::cout << "  Position error: " << (pos_error * 100) << " cm\n";
    std::cout << "  Yaw error: " << (yaw_error * 180 / M_PI) << " deg\n";

    std::cout << "\nTiming:\n";
    std::cout << "  Coarse stage: " << result.coarse_time_ms << " ms\n";
    std::cout << "  Medium stage: " << result.medium_time_ms << " ms\n";
    std::cout << "  Fine stage:   " << result.fine_time_ms << " ms\n";
    std::cout << "  Total:        " << result.total_time_ms << " ms\n";

    std::cout << "\nQuality:\n";
    std::cout << "  Fitness score: " << result.fitness_score << "\n";
    std::cout << "  RMSE: " << result.rmse << " m\n";
    std::cout << "  Converged: " << (result.converged ? "yes" : "no") << "\n";

    // Success criteria
    bool position_ok = pos_error < 0.5;  // 50cm tolerance for large offsets
    bool yaw_ok = yaw_error < 0.35;       // ~20 degree tolerance
    bool success = position_ok && yaw_ok && result.fitness_score > 0.2;

    std::cout << "\n============================================\n";
    if (success) {
        std::cout << "  RELOCALIZATION SUCCESSFUL!\n";
    } else {
        std::cout << "  RELOCALIZATION FAILED\n";
        if (!position_ok) std::cout << "  - Position error too large\n";
        if (!yaw_ok) std::cout << "  - Yaw error too large\n";
        if (result.fitness_score <= 0.2) std::cout << "  - Fitness score too low\n";
    }
    std::cout << "============================================\n";

    return success ? 0 : 1;
}
