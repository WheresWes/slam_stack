/**
 * @file test_progressive_localization.cpp
 * @brief Test progressive global localization using FAST-LIO for accumulation
 *
 * This test demonstrates the coverage-based localization workflow:
 * 1. Load pre-built map into memory (not ikd-tree)
 * 2. Run FAST-LIO in SLAM mode to build a local map
 * 3. Monitor coverage (voxels, rotation) of the local map
 * 4. When coverage is sufficient, run global localization
 * 5. Swap to pre-built map and continue in localization mode
 *
 * Usage:
 *   test_progressive_localization --map map.ply --recording data.bin
 */

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
#endif

#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <cstring>
#include <vector>

#include "slam/slam_engine.hpp"
#include "slam/progressive_localizer.hpp"

using namespace slam;

// Recording structures (same format as replay_slam.cpp)
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
    if (!file.is_open()) {
        std::cerr << "Failed to open recording: " << filename << std::endl;
        return false;
    }

    // Read header
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x534C414D) {  // "SLAM"
        std::cerr << "Invalid recording file format" << std::endl;
        return false;
    }

    std::cout << "Recording format version: " << version << std::endl;

    // Read messages
    while (file.good() && !file.eof()) {
        uint8_t msg_type;
        file.read(reinterpret_cast<char*>(&msg_type), 1);
        if (file.eof()) break;

        if (msg_type == 1) {  // Point cloud
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

        } else if (msg_type == 2) {  // IMU
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

    std::cout << "Loaded: " << clouds.size() << " point clouds, "
              << imus.size() << " IMU samples" << std::endl;

    return !clouds.empty();
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " --map <map.ply> --recording <data.bin> [options]\n"
              << "\nOptions:\n"
              << "  --map <file>        Pre-built map (PCD or PLY)\n"
              << "  --recording <file>  Recording file from live_slam\n"
              << "  --verbose           Print detailed progress\n"
              << std::endl;
}

int main(int argc, char** argv) {
    std::string map_file;
    std::string recording_file;
    bool verbose = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--map") == 0 && i + 1 < argc) {
            map_file = argv[++i];
        } else if (strcmp(argv[i], "--recording") == 0 && i + 1 < argc) {
            recording_file = argv[++i];
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (map_file.empty() || recording_file.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "============================================" << std::endl;
    std::cout << "  Progressive Localization Test" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Map: " << map_file << std::endl;
    std::cout << "Recording: " << recording_file << std::endl;
    std::cout << std::endl;

    // Load recording
    std::vector<RecordedPointCloud> clouds;
    std::vector<RecordedIMU> imus;
    if (!loadRecording(recording_file, clouds, imus)) {
        return 1;
    }

    // Initialize SLAM engine
    SlamConfig config;
    config.max_iterations = 3;
    config.filter_size_surf = 0.2;
    config.filter_size_map = 0.2;
    config.gyr_cov = 0.1;
    config.acc_cov = 0.1;

    SlamEngine slam;
    if (!slam.init(config)) {
        std::cerr << "Failed to initialize SLAM engine" << std::endl;
        return 1;
    }

    // Load pre-built map (into memory, not ikd-tree)
    if (!slam.loadPrebuiltMap(map_file)) {
        std::cerr << "Failed to load pre-built map" << std::endl;
        return 1;
    }

    // Start progressive localization
    ProgressiveLocalizerConfig loc_config;
    loc_config.min_confidence = 0.45;
    loc_config.high_confidence = 0.60;
    loc_config.grid_step = 1.5;
    loc_config.yaw_step_deg = 30.0;
    slam.startProgressiveLocalization(loc_config);

    // Sort IMU by timestamp for efficient lookup
    std::sort(imus.begin(), imus.end(),
              [](const RecordedIMU& a, const RecordedIMU& b) {
                  return a.timestamp_ns < b.timestamp_ns;
              });

    // Process recording
    size_t imu_idx = 0;
    int frame_count = 0;
    LocalizationStatus last_status = LocalizationStatus::NOT_STARTED;
    auto start_time = std::chrono::high_resolution_clock::now();

    constexpr double G_M_S2 = 9.81;

    for (const auto& cloud : clouds) {
        frame_count++;

        // Feed IMU data up to this point cloud timestamp
        while (imu_idx < imus.size() && imus[imu_idx].timestamp_ns <= cloud.timestamp_ns) {
            ImuData imu;
            imu.timestamp_ns = imus[imu_idx].timestamp_ns;
            imu.gyro = imus[imu_idx].gyro;
            imu.acc = imus[imu_idx].accel * G_M_S2;  // Convert from g to m/s^2
            slam.addImuData(imu);
            imu_idx++;
        }

        // Convert and add point cloud
        PointCloud pc;
        pc.timestamp_ns = cloud.timestamp_ns;
        for (size_t i = 0; i < cloud.points.size(); i++) {
            LidarPoint lp;
            lp.x = cloud.points[i].x();
            lp.y = cloud.points[i].y();
            lp.z = cloud.points[i].z();
            lp.intensity = cloud.reflectivities[i];
            lp.time_offset_ms = 0;  // Not available in this format
            pc.push_back(lp);
        }
        slam.addPointCloud(pc);

        // Process SLAM
        int processed = slam.process();

        if (processed > 0) {
            // Check progressive localization
            auto result = slam.checkProgressiveLocalization();

            // Print status updates
            if (result.status != last_status || verbose) {
                if (result.status == LocalizationStatus::ACCUMULATING ||
                    result.status == LocalizationStatus::VIEW_SATURATED ||
                    result.status == LocalizationStatus::LOW_CONFIDENCE) {
                    std::cout << "[Frame " << frame_count << "] " << result.message << std::endl;
                }
                last_status = result.status;
            }

            // Check if localization is complete
            if (result.status == LocalizationStatus::SUCCESS) {
                auto end_time = std::chrono::high_resolution_clock::now();
                double elapsed_s = std::chrono::duration<double>(end_time - start_time).count();

                std::cout << "\n============================================" << std::endl;
                std::cout << "  TEST COMPLETED SUCCESSFULLY" << std::endl;
                std::cout << "============================================" << std::endl;
                std::cout << "Time to localize: " << elapsed_s << " seconds" << std::endl;
                std::cout << "Frames processed: " << frame_count << std::endl;
                std::cout << "Final confidence: " << (result.confidence * 100) << "%" << std::endl;
                std::cout << std::endl;

                // Continue processing remaining frames to verify tracking
                std::cout << "Verifying tracking in localization mode..." << std::endl;
                int verify_frames = 0;
                size_t map_size_before = slam.getMapSize();

                for (size_t fi = frame_count; fi < clouds.size() && verify_frames < 20; fi++) {
                    const auto& c = clouds[fi];

                    // Feed IMU
                    while (imu_idx < imus.size() && imus[imu_idx].timestamp_ns <= c.timestamp_ns) {
                        ImuData imu;
                        imu.timestamp_ns = imus[imu_idx].timestamp_ns;
                        imu.gyro = imus[imu_idx].gyro;
                        imu.acc = imus[imu_idx].accel * G_M_S2;
                        slam.addImuData(imu);
                        imu_idx++;
                    }

                    // Add cloud
                    PointCloud p;
                    p.timestamp_ns = c.timestamp_ns;
                    for (size_t i = 0; i < c.points.size(); i++) {
                        LidarPoint lp;
                        lp.x = c.points[i].x();
                        lp.y = c.points[i].y();
                        lp.z = c.points[i].z();
                        lp.intensity = c.reflectivities[i];
                        lp.time_offset_ms = 0;
                        p.push_back(lp);
                    }
                    slam.addPointCloud(p);
                    slam.process();
                    verify_frames++;
                }

                size_t map_size_after = slam.getMapSize();
                V3D final_pos = slam.getPosition();

                std::cout << "Verified " << verify_frames << " frames in localization mode" << std::endl;
                std::cout << "Map size before: " << map_size_before << std::endl;
                std::cout << "Map size after:  " << map_size_after << std::endl;
                std::cout << "Map modified: " << (map_size_after != map_size_before ? "YES (ERROR!)" : "NO (correct)") << std::endl;
                std::cout << "Final position: [" << final_pos.x() << ", " << final_pos.y() << ", " << final_pos.z() << "]" << std::endl;
                std::cout << "============================================\n" << std::endl;

                return 0;
            }

            if (result.status == LocalizationStatus::FAILED) {
                std::cerr << "\n============================================" << std::endl;
                std::cerr << "  LOCALIZATION FAILED" << std::endl;
                std::cerr << "============================================" << std::endl;
                std::cerr << "Reason: " << result.message << std::endl;
                std::cerr << "============================================\n" << std::endl;
                return 1;
            }
        }
    }

    // If we get here, we ran out of recording data
    std::cerr << "\n============================================" << std::endl;
    std::cerr << "  RECORDING EXHAUSTED" << std::endl;
    std::cerr << "============================================" << std::endl;
    std::cerr << "Localization did not complete before end of recording" << std::endl;
    std::cerr << "Final status: " << toString(slam.getProgressiveLocalizationStatus()) << std::endl;
    std::cerr << "============================================\n" << std::endl;

    return 1;
}
