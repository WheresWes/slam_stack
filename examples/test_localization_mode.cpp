/**
 * @file test_localization_mode.cpp
 * @brief Test FAST-LIO localization mode with pre-built map
 *
 * This demonstrates the full localization workflow:
 * 1. Load a pre-built map
 * 2. Run global localization to find initial pose
 * 3. Enable localization mode (no map updates)
 * 4. Track pose using IEKF against the pre-built map
 *
 * Use case: Localizing on a previously scanned boat hull
 */

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
#endif

#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <thread>
#include <unordered_set>
#include <algorithm>

#include "slam/slam_engine.hpp"
#include "slam/types.hpp"
#include "slam/icp.hpp"

using namespace slam;

//=============================================================================
// Recording Loader (from replay_slam.cpp)
//=============================================================================

struct RecordedPointCloud {
    uint64_t timestamp_ns;
    std::vector<V3D> points;
    std::vector<uint8_t> reflectivities;
};

struct RecordedIMU {
    uint64_t timestamp_ns;
    V3D accel;
    V3D gyro;
};

bool loadRecording(const std::string& filename,
                   std::vector<RecordedPointCloud>& clouds,
                   std::vector<RecordedIMU>& imus) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != 0x534C414D) return false;

    while (file.good() && !file.eof()) {
        uint8_t msg_type;
        file.read(reinterpret_cast<char*>(&msg_type), 1);
        if (file.eof()) break;

        if (msg_type == 1) {
            RecordedPointCloud cloud;
            file.read(reinterpret_cast<char*>(&cloud.timestamp_ns), sizeof(cloud.timestamp_ns));

            uint32_t num_points;
            file.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));

            cloud.points.reserve(num_points);
            cloud.reflectivities.reserve(num_points);

            for (uint32_t i = 0; i < num_points; i++) {
                float x, y, z;
                uint8_t refl;
                file.read(reinterpret_cast<char*>(&x), sizeof(x));
                file.read(reinterpret_cast<char*>(&y), sizeof(y));
                file.read(reinterpret_cast<char*>(&z), sizeof(z));
                file.read(reinterpret_cast<char*>(&refl), sizeof(refl));

                cloud.points.emplace_back(x, y, z);
                cloud.reflectivities.push_back(refl);
            }

            clouds.push_back(std::move(cloud));

        } else if (msg_type == 2) {
            RecordedIMU imu;
            file.read(reinterpret_cast<char*>(&imu.timestamp_ns), sizeof(imu.timestamp_ns));

            float ax, ay, az, gx, gy, gz;
            file.read(reinterpret_cast<char*>(&ax), sizeof(ax));
            file.read(reinterpret_cast<char*>(&ay), sizeof(ay));
            file.read(reinterpret_cast<char*>(&az), sizeof(az));
            file.read(reinterpret_cast<char*>(&gx), sizeof(gx));
            file.read(reinterpret_cast<char*>(&gy), sizeof(gy));
            file.read(reinterpret_cast<char*>(&gz), sizeof(gz));

            imu.accel = V3D(ax, ay, az);
            imu.gyro = V3D(gx, gy, gz);

            imus.push_back(imu);
        }
    }

    return !clouds.empty() && !imus.empty();
}

//=============================================================================
// Voxel Occupancy for Global Localization
//=============================================================================

class VoxelOccupancy {
public:
    VoxelOccupancy(double voxel_size = 0.5) : voxel_size_(voxel_size) {}

    void build(const std::vector<V3D>& points) {
        for (const auto& p : points) {
            occupied_.insert(toKey(p));
        }
    }

    double scorePose(const std::vector<V3D>& query, const M4D& pose, int samples = 500) const {
        M3D R = pose.block<3,3>(0,0);
        V3D t = pose.block<3,1>(0,3);

        int hits = 0;
        int step = std::max(1, static_cast<int>(query.size()) / samples);
        int count = 0;

        for (size_t i = 0; i < query.size(); i += step) {
            V3D world_pt = R * query[i] + t;
            if (occupied_.count(toKey(world_pt)) > 0) hits++;
            count++;
        }
        return static_cast<double>(hits) / count;
    }

private:
    double voxel_size_;
    std::unordered_set<int64_t> occupied_;

    int64_t toKey(const V3D& p) const {
        int64_t ix = static_cast<int64_t>(std::floor(p.x() / voxel_size_));
        int64_t iy = static_cast<int64_t>(std::floor(p.y() / voxel_size_));
        int64_t iz = static_cast<int64_t>(std::floor(p.z() / voxel_size_));
        return (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791);
    }
};

//=============================================================================
// Global Localization (Grid + Rotation Search)
//=============================================================================

M4D makePose(double x, double y, double z, double yaw) {
    M3D R = Eigen::AngleAxisd(yaw, V3D::UnitZ()).toRotationMatrix();
    M4D pose = M4D::Identity();
    pose.block<3,3>(0,0) = R;
    pose(0, 3) = x;
    pose(1, 3) = y;
    pose(2, 3) = z;
    return pose;
}

struct PoseCandidate {
    M4D pose;
    double score;
};

M4D globalLocalizeScan(const std::vector<V3D>& query,
                       const std::vector<V3D>& map_points,
                       double grid_step = 1.0,
                       double yaw_step_deg = 30.0) {

    std::cout << "[GlobalLocalize] Starting grid+rotation search...\n";

    // Build voxel map for fast scoring
    VoxelOccupancy voxel_map(0.5);
    voxel_map.build(map_points);

    // Find map bounds
    V3D min_b = map_points[0], max_b = map_points[0];
    for (const auto& p : map_points) {
        min_b = min_b.cwiseMin(p);
        max_b = max_b.cwiseMax(p);
    }

    // Generate candidates
    std::vector<PoseCandidate> candidates;
    double yaw_step = yaw_step_deg * M_PI / 180.0;
    int num_yaw = static_cast<int>(std::ceil(2.0 * M_PI / yaw_step));

    for (double x = min_b.x(); x <= max_b.x(); x += grid_step) {
        for (double y = min_b.y(); y <= max_b.y(); y += grid_step) {
            for (int iy = 0; iy < num_yaw; iy++) {
                double yaw = iy * yaw_step;
                M4D pose = makePose(x, y, 0, yaw);
                double score = voxel_map.scorePose(query, pose);
                candidates.push_back({pose, score});
            }
        }
    }

    std::cout << "  Generated " << candidates.size() << " candidates\n";

    // Sort by score
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });

    std::cout << "  Best voxel score: " << candidates[0].score << "\n";

    // Refine top candidates with ICP
    int num_refine = std::min(50, static_cast<int>(candidates.size()));
    auto query_down = voxelDownsample(query, 0.2);
    auto map_down = voxelDownsample(map_points, 0.2);

    PoseCandidate best;
    best.score = 0;

    for (int i = 0; i < num_refine; i++) {
        ICPConfig cfg;
        cfg.max_iterations = 30;
        cfg.max_correspondence_dist = 1.0;

        ICP icp(cfg);
        auto result = icp.align(query_down, map_down, candidates[i].pose);

        if (result.fitness_score > best.score) {
            best.pose = result.transformation;
            best.score = result.fitness_score;
        }
    }

    std::cout << "  Best ICP fitness: " << best.score << "\n";

    // Final refinement
    auto query_fine = voxelDownsample(query, 0.1);
    auto map_fine = voxelDownsample(map_points, 0.1);

    ICPConfig fine_cfg;
    fine_cfg.max_iterations = 50;
    fine_cfg.max_correspondence_dist = 0.5;

    ICP icp(fine_cfg);
    auto final_result = icp.align(query_fine, map_fine, best.pose);

    std::cout << "  Final fitness: " << final_result.fitness_score << "\n";

    return final_result.transformation;
}

//=============================================================================
// Main
//=============================================================================

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <map.ply> <recording.bin> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --offset <x> <y> <yaw>  Apply artificial offset (meters, degrees)\n";
    std::cout << "  --skip-global           Skip global localization (use offset as initial)\n";
    std::cout << "  --speed <x>             Playback speed (default: 0 = max)\n";
    std::cout << "  --help                  Show this help\n";
}

int main(int argc, char** argv) {
    std::cout << "============================================\n";
    std::cout << "  Localization Mode Test\n";
    std::cout << "  (FAST-LIO with Pre-Built Map)\n";
    std::cout << "============================================\n\n";

    if (argc < 3 || std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return argc < 3 ? 1 : 0;
    }

    std::string map_file = argv[1];
    std::string recording_file = argv[2];
    double offset_x = 1.0, offset_y = 0.5, offset_yaw = 45.0;
    bool skip_global = false;
    double playback_speed = 0;

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--offset" && i + 3 < argc) {
            offset_x = std::atof(argv[++i]);
            offset_y = std::atof(argv[++i]);
            offset_yaw = std::atof(argv[++i]);
        } else if (arg == "--skip-global") {
            skip_global = true;
        } else if (arg == "--speed" && i + 1 < argc) {
            playback_speed = std::atof(argv[++i]);
        }
    }

    // Load recording
    std::cout << "Loading recording: " << recording_file << "\n";
    std::vector<RecordedPointCloud> clouds;
    std::vector<RecordedIMU> imus;
    if (!loadRecording(recording_file, clouds, imus)) {
        std::cerr << "Failed to load recording!\n";
        return 1;
    }
    std::cout << "  Point cloud frames: " << clouds.size() << "\n";
    std::cout << "  IMU samples: " << imus.size() << "\n";

    // Initialize SLAM engine
    std::cout << "\nInitializing SLAM engine...\n";
    SlamEngine slam;

    SlamConfig config;
    config.filter_size_surf = 0.05;
    config.filter_size_map = 0.05;
    config.deskew_enabled = true;
    config.gyr_cov = 0.1;
    config.acc_cov = 0.1;
    config.max_iterations = 5;

    if (!slam.init(config)) {
        std::cerr << "Failed to initialize SLAM!\n";
        return 1;
    }

    // Load pre-built map
    std::cout << "\nLoading pre-built map: " << map_file << "\n";
    if (!slam.loadMap(map_file)) {
        std::cerr << "Failed to load map!\n";
        return 1;
    }

    // Get map points for global localization
    auto map_world_points = slam.getMapPoints();
    std::vector<V3D> map_v3d;
    map_v3d.reserve(map_world_points.size());
    for (const auto& wp : map_world_points) {
        map_v3d.emplace_back(wp.x, wp.y, wp.z);
    }

    // Accumulate initial scan for global localization
    std::cout << "\nAccumulating initial scan for localization...\n";
    std::vector<V3D> init_scan;
    for (int i = 0; i < std::min(10, static_cast<int>(clouds.size())); i++) {
        for (size_t j = 0; j < clouds[i].points.size(); j++) {
            const auto& p = clouds[i].points[j];
            double d = p.norm();
            if (d > 0.5 && d < 50.0) {
                init_scan.push_back(p);
            }
        }
    }
    init_scan = voxelDownsample(init_scan, 0.05);
    std::cout << "  Initial scan points: " << init_scan.size() << "\n";

    // Simulate artificial offset (unknown initial pose)
    double offset_yaw_rad = offset_yaw * M_PI / 180.0;
    M4D artificial_offset = makePose(offset_x, offset_y, 0, offset_yaw_rad);

    std::cout << "\nArtificial offset (simulating unknown initial pose):\n";
    std::cout << "  X: " << offset_x << " m\n";
    std::cout << "  Y: " << offset_y << " m\n";
    std::cout << "  Yaw: " << offset_yaw << " deg\n";

    // Transform initial scan by inverse offset (as if we're at offset position)
    M4D offset_inv = artificial_offset.inverse();
    std::vector<V3D> offset_scan;
    M3D R_inv = offset_inv.block<3,3>(0,0);
    V3D t_inv = offset_inv.block<3,1>(0,3);
    for (const auto& p : init_scan) {
        offset_scan.push_back(R_inv * p + t_inv);
    }

    // Run global localization
    M4D initial_pose;
    if (skip_global) {
        initial_pose = artificial_offset;
        std::cout << "\nSkipping global localization, using offset as initial pose\n";
    } else {
        std::cout << "\n--- Running Global Localization ---\n";
        initial_pose = globalLocalizeScan(offset_scan, map_v3d, 1.0, 30.0);

        V3D recovered_pos = initial_pose.block<3,1>(0,3);
        double recovered_yaw = std::atan2(initial_pose(1,0), initial_pose(0,0)) * 180.0 / M_PI;

        double pos_error = (recovered_pos - V3D(offset_x, offset_y, 0)).norm();
        double yaw_error = std::abs(recovered_yaw - offset_yaw);
        if (yaw_error > 180) yaw_error = 360 - yaw_error;

        std::cout << "\nGlobal localization result:\n";
        std::cout << "  Recovered: [" << recovered_pos.x() << ", " << recovered_pos.y() << "] yaw="
                  << recovered_yaw << " deg\n";
        std::cout << "  True: [" << offset_x << ", " << offset_y << "] yaw=" << offset_yaw << " deg\n";
        std::cout << "  Error: pos=" << (pos_error * 100) << "cm, yaw=" << yaw_error << " deg\n";
    }

    // Set initial pose and enable localization mode
    std::cout << "\n--- Enabling Localization Mode ---\n";
    slam.setInitialPose(initial_pose);
    slam.setLocalizationMode(true);

    // Process recording
    std::cout << "\n--- Processing Recording in Localization Mode ---\n";

    // Merge events by timestamp
    struct Event {
        uint64_t timestamp_ns;
        bool is_imu;
        size_t index;
    };
    std::vector<Event> events;
    for (size_t i = 0; i < clouds.size(); i++) {
        events.push_back({clouds[i].timestamp_ns, false, i});
    }
    for (size_t i = 0; i < imus.size(); i++) {
        events.push_back({imus[i].timestamp_ns, true, i});
    }
    std::sort(events.begin(), events.end(),
              [](const Event& a, const Event& b) { return a.timestamp_ns < b.timestamp_ns; });

    uint64_t base_time = events[0].timestamp_ns;
    auto start_wall = std::chrono::steady_clock::now();

    // Accumulator for complete scans
    std::vector<LidarPoint> accumulated;
    uint64_t scan_start = 0;
    const double SCAN_PERIOD_MS = 100.0;
    int scan_count = 0;
    size_t map_size_before = slam.getMapSize();

    for (size_t ei = 0; ei < events.size(); ei++) {
        const Event& e = events[ei];

        // Playback timing
        if (playback_speed > 0) {
            double event_sec = (e.timestamp_ns - base_time) / 1e9;
            double target_sec = event_sec / playback_speed;
            auto elapsed = std::chrono::steady_clock::now() - start_wall;
            double elapsed_sec = std::chrono::duration<double>(elapsed).count();
            if (elapsed_sec < target_sec) {
                std::this_thread::sleep_for(std::chrono::duration<double>(target_sec - elapsed_sec));
            }
        }

        if (e.is_imu) {
            const auto& rec = imus[e.index];
            ImuData imu;
            imu.timestamp_ns = rec.timestamp_ns;
            imu.acc = rec.accel * 9.81;  // Convert g to m/s²
            imu.gyro = rec.gyro;
            slam.addImuData(imu);
        } else {
            const auto& rec = clouds[e.index];

            if (accumulated.empty()) {
                scan_start = rec.timestamp_ns;
            }

            double time_offset_ms = (rec.timestamp_ns - scan_start) / 1e6;

            for (size_t i = 0; i < rec.points.size(); i++) {
                const auto& p = rec.points[i];
                double d = p.norm();
                if (d < 0.5 || d > 100.0) continue;

                LidarPoint lp;
                lp.x = p.x();
                lp.y = p.y();
                lp.z = p.z();
                lp.intensity = rec.reflectivities[i];
                lp.time_offset_ms = time_offset_ms;
                accumulated.push_back(lp);
            }

            if (time_offset_ms >= SCAN_PERIOD_MS && accumulated.size() > 100) {
                PointCloud cloud;
                cloud.timestamp_ns = scan_start;
                cloud.points = std::move(accumulated);
                slam.addPointCloud(cloud);
                slam.process();

                accumulated.clear();
                scan_start = rec.timestamp_ns;
                scan_count++;
            }
        }

        // Progress
        if ((ei + 1) % 2000 == 0 || ei == events.size() - 1) {
            double progress = 100.0 * (ei + 1) / events.size();
            SlamState state = slam.getState();
            std::cout << "\r  Progress: " << std::fixed << std::setprecision(1) << progress << "% | "
                      << "Scans: " << scan_count << " | "
                      << "Pos: [" << std::setprecision(2)
                      << state.pos.x() << ", " << state.pos.y() << ", " << state.pos.z() << "]    "
                      << std::flush;
        }
    }

    std::cout << "\n\n--- Results ---\n";

    // Verify map wasn't modified
    size_t map_size_after = slam.getMapSize();
    std::cout << "Map size before: " << map_size_before << "\n";
    std::cout << "Map size after:  " << map_size_after << "\n";
    std::cout << "Map modified:    " << (map_size_after != map_size_before ? "YES (ERROR!)" : "NO (correct)") << "\n";

    // Final pose
    SlamState final_state = slam.getState();
    std::cout << "\nFinal position: [" << final_state.pos.x() << ", "
              << final_state.pos.y() << ", " << final_state.pos.z() << "]\n";

    // Should return near initial pose (recording was a loop back to origin)
    double drift = final_state.pos.norm();
    std::cout << "Drift from origin: " << (drift * 100) << " cm\n";

    std::cout << "\nLocalization fitness: " << slam.getLocalizationFitness() << "\n";

    bool success = (map_size_after == map_size_before) && (drift < 0.5);
    std::cout << "\n============================================\n";
    if (success) {
        std::cout << "  LOCALIZATION MODE TEST PASSED!\n";
    } else {
        std::cout << "  LOCALIZATION MODE TEST FAILED\n";
        if (map_size_after != map_size_before) {
            std::cout << "  - Map was modified (should be unchanged)\n";
        }
        if (drift >= 0.5) {
            std::cout << "  - Drift too large: " << (drift * 100) << "cm\n";
        }
    }
    std::cout << "============================================\n";

    return success ? 0 : 1;
}
