/**
 * Replay Localization Session
 *
 * Replays a recorded localization session offline for development and testing.
 * No hardware needed - reads from binary file created by slam_gui recording.
 *
 * Exit codes:
 *   0 = Localization succeeded
 *   1 = Localization failed (max attempts)
 *   2 = Timeout
 *   3 = File error (bad recording, missing map)
 *   4 = SLAM initialization failure
 *
 * Usage with Ralph Wiggum loop:
 *   while :; do cat PROMPT.md | claude ; done
 *   Each iteration: modify code -> build -> replay_localization --input session.bin
 */

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

#include "slam/slam_engine.hpp"
#include "slam/sensor_fusion.hpp"
#include "slam/types.hpp"
#include "slam/localization_recording.hpp"

using namespace slam;
using namespace slam::recording;

// ============================================================================
// Exit codes
// ============================================================================
constexpr int EXIT_SUCCESS_LOC = 0;
constexpr int EXIT_FAILED = 1;
constexpr int EXIT_TIMEOUT = 2;
constexpr int EXIT_FILE_ERROR = 3;
constexpr int EXIT_INIT_ERROR = 4;

// ============================================================================
// Command line options
// ============================================================================
struct Options {
    std::string input_file;
    std::string map_override;
    float hint_x = 0, hint_y = 0;
    float hint_heading_deg = 0;
    float hint_radius = 0;
    bool override_hint = false;
    bool override_heading = false;
    bool override_radius = false;
    bool no_hint = false;
    float max_time_s = 120.0f;
    bool verbose = false;
    bool json = false;
};

void printUsage() {
    std::cout << "Usage: replay_localization [options]\n"
              << "  --input FILE          Input recording (required)\n"
              << "  --map FILE            Override map file path\n"
              << "  --hint X,Y            Override hint position\n"
              << "  --hint-heading DEG    Override hint heading (degrees)\n"
              << "  --hint-radius M       Override search radius (meters)\n"
              << "  --no-hint             Ignore recorded hint (full map search)\n"
              << "  --max-time SECS       Max replay time (default: 120)\n"
              << "  --verbose             Print detailed progress\n"
              << "  --json                Output results as JSON\n"
              << "  --help                Show this help\n";
}

Options parseArgs(int argc, char* argv[]) {
    Options opts;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            opts.input_file = argv[++i];
        } else if (arg == "--map" && i + 1 < argc) {
            opts.map_override = argv[++i];
        } else if (arg == "--hint" && i + 1 < argc) {
            std::string val = argv[++i];
            auto comma = val.find(',');
            if (comma != std::string::npos) {
                opts.hint_x = std::stof(val.substr(0, comma));
                opts.hint_y = std::stof(val.substr(comma + 1));
                opts.override_hint = true;
            }
        } else if (arg == "--hint-heading" && i + 1 < argc) {
            opts.hint_heading_deg = std::stof(argv[++i]);
            opts.override_heading = true;
        } else if (arg == "--hint-radius" && i + 1 < argc) {
            opts.hint_radius = std::stof(argv[++i]);
            opts.override_radius = true;
        } else if (arg == "--max-time" && i + 1 < argc) {
            opts.max_time_s = std::stof(argv[++i]);
        } else if (arg == "--no-hint") {
            opts.no_hint = true;
        } else if (arg == "--verbose") {
            opts.verbose = true;
        } else if (arg == "--json") {
            opts.json = true;
        } else if (arg == "--help") {
            printUsage();
            std::exit(0);
        }
    }
    return opts;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    Options opts = parseArgs(argc, argv);

    if (opts.input_file.empty()) {
        std::cerr << "Error: --input is required\n";
        printUsage();
        return EXIT_FILE_ERROR;
    }

    if (!opts.json) {
        std::cout << "================================================\n";
        std::cout << "  Replay Localization Session\n";
        std::cout << "================================================\n";
    }

    // ================================================================
    // Read recording file
    // ================================================================
    std::ifstream input(opts.input_file, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "Failed to open: " << opts.input_file << std::endl;
        return EXIT_FILE_ERROR;
    }

    // Verify magic and version
    char magic[4];
    uint32_t version;
    input.read(magic, 4);
    input.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (std::string(magic, 4) != "FUSN" || version != 2) {
        std::cerr << "Invalid file format (expected FUSN v2, got "
                  << std::string(magic, 4) << " v" << version << ")\n";
        return EXIT_FILE_ERROR;
    }

    // Read session header
    SessionHeader session;
    input.read(reinterpret_cast<char*>(&session), sizeof(session));
    if (!input.good()) {
        std::cerr << "Failed to read session header\n";
        return EXIT_FILE_ERROR;
    }

    // Read map filename
    std::string map_path(session.map_filename_len, '\0');
    input.read(map_path.data(), session.map_filename_len);
    if (!input.good()) {
        std::cerr << "Failed to read map filename\n";
        return EXIT_FILE_ERROR;
    }

    // Apply overrides
    if (!opts.map_override.empty()) map_path = opts.map_override;

    float hint_x = session.hint_x;
    float hint_y = session.hint_y;
    float hint_heading = session.hint_heading_rad;
    float hint_radius = session.hint_search_radius_m;
    bool hint_valid = session.hint_valid != 0;
    bool hint_heading_known = session.hint_heading_known != 0;

    if (opts.no_hint) { hint_valid = false; hint_heading_known = false; }
    if (opts.override_hint) { hint_x = opts.hint_x; hint_y = opts.hint_y; hint_valid = true; }
    if (opts.override_heading) { hint_heading = opts.hint_heading_deg * 3.14159265f / 180.0f; hint_heading_known = true; }
    if (opts.override_radius) { hint_radius = opts.hint_radius; }

    if (!opts.json) {
        std::cout << "\nSession info:\n";
        std::cout << "  Map: " << map_path << "\n";
        std::cout << "  Hint: " << (hint_valid ? "yes" : "no");
        if (hint_valid) {
            std::cout << " (" << hint_x << ", " << hint_y << ") radius=" << hint_radius << "m";
            if (hint_heading_known) std::cout << " heading=" << (hint_heading * 180.0f / 3.14159265f) << "deg";
        }
        std::cout << "\n";
        std::cout << "  Config: voxel=" << session.voxel_size
                  << " gyr_cov=" << session.gyr_cov
                  << " deskew=" << (session.deskew_enabled ? "on" : "off")
                  << " lpf=" << session.imu_lpf_alpha << "\n";
    }

    // ================================================================
    // Initialize SLAM Engine
    // ================================================================
    if (!opts.json) std::cout << "\nInitializing SLAM engine...\n";

    SlamEngine slam;
    SlamConfig slam_config;
    slam_config.filter_size_surf = session.voxel_size;
    slam_config.filter_size_map = session.voxel_size;
    slam_config.gyr_cov = session.gyr_cov;
    slam_config.acc_cov = session.acc_cov;
    slam_config.imu_lpf_alpha = session.imu_lpf_alpha;
    slam_config.deskew_enabled = session.deskew_enabled != 0;
    slam_config.max_iterations = session.max_iterations;
    slam_config.max_points_icp = session.max_points_icp;
    slam_config.max_position_jump = session.max_position_jump;
    slam_config.max_rotation_jump_deg = session.max_rotation_jump_deg;

    if (!slam.init(slam_config)) {
        std::cerr << "Failed to initialize SLAM engine!\n";
        return EXIT_INIT_ERROR;
    }

    // Load pre-built map
    if (!opts.json) std::cout << "Loading pre-built map: " << map_path << "\n";
    if (!slam.loadPrebuiltMap(map_path)) {
        std::cerr << "Failed to load map: " << map_path << "\n";
        return EXIT_FILE_ERROR;
    }

    // ================================================================
    // Initialize Sensor Fusion
    // ================================================================
    SensorFusion fusion;
    FusionConfig fusion_config;
    fusion_config.stationary_erpm_threshold = 50;
    fusion_config.turning_angular_threshold = 0.2f;
    fusion_config.slam_position_alpha = 0.60f;
    fusion_config.slam_heading_alpha = 0.65f;
    fusion_config.straight_heading_correction = 4.0f;
    fusion_config.straight_position_correction = 4.0f;
    fusion_config.turning_position_correction = 12.0f;
    fusion.init(fusion_config);

    // ================================================================
    // Reset SLAM and start progressive localization (matches RELOCALIZE handler)
    // ================================================================
    slam.reset();
    slam.setVoxelSize(session.voxel_size);
    slam.setGyrCov(session.gyr_cov);
    slam.setDeskewEnabled(session.deskew_enabled != 0);
    slam.setImuLpfAlpha(session.imu_lpf_alpha);
    fusion.reset();

    // Configure progressive localizer
    ProgressiveLocalizerConfig loc_config;
    loc_config.max_attempts = 5;
    loc_config.min_confidence = 0.45;
    loc_config.high_confidence = 0.65;

    if (hint_valid) {
        loc_config.use_hint = true;
        loc_config.hint_x = hint_x;
        loc_config.hint_y = hint_y;
        loc_config.hint_radius = hint_radius;
        loc_config.hint_heading = hint_heading;
        loc_config.hint_heading_known = hint_heading_known;
        loc_config.hint_heading_range = 3.14159265 / 3.0;
        loc_config.grid_step = 1.0;
        loc_config.yaw_step_deg = 15.0;
    }

    slam.startProgressiveLocalization(loc_config);

    if (!opts.json) {
        std::cout << "\nStarting localization replay...\n";
        if (hint_valid) {
            std::cout << "  Hint: (" << hint_x << ", " << hint_y << ") r=" << hint_radius << "m\n";
        }
    }

    // ================================================================
    // Main replay loop
    // ================================================================
    uint64_t imu_count = 0, scan_count = 0, vesc_count = 0;
    bool slam_initialized = false;
    bool localization_complete = false;
    bool localization_success = false;
    float final_confidence = 0.0f;
    Eigen::Matrix4d final_pose = Eigen::Matrix4d::Identity();
    int attempts = 0;

    // Monotonic timestamps for SLAM (avoid loop-back issues)
    uint64_t mono_timestamp_ns = 0;
    const uint64_t IMU_PERIOD_NS = 5000000;   // 5ms = 200Hz

    auto replay_start = std::chrono::steady_clock::now();
    float recording_duration_s = 0.0f;

    while (input.good() && !input.eof() && !localization_complete) {
        // Check timeout
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - replay_start).count();
        if (elapsed > opts.max_time_s) {
            if (!opts.json) std::cerr << "\nTimeout after " << elapsed << "s\n";
            break;
        }

        RecordHeader header;
        input.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!input.good()) break;

        recording_duration_s = header.timestamp_us / 1e6f;

        switch (header.type) {
            case RecordType::IMU: {
                ImuRecord rec;
                input.read(reinterpret_cast<char*>(&rec), sizeof(rec));
                if (!input.good()) break;

                mono_timestamp_ns += IMU_PERIOD_NS;

                ImuData imu;
                imu.timestamp_ns = mono_timestamp_ns;
                // NOTE: acc values are already in m/s^2 (converted at record time)
                imu.acc = V3D(rec.acc_x, rec.acc_y, rec.acc_z);
                imu.gyro = V3D(rec.gyro_x, rec.gyro_y, rec.gyro_z);
                slam.addImuData(imu);
                imu_count++;

                // Process SLAM every 20 IMU samples (~100ms at 200Hz = 10Hz SLAM rate)
                if (imu_count % 20 == 0) {
                    int processed = slam.process();

                    if (!slam_initialized && slam.isInitialized()) {
                        slam_initialized = true;
                        if (opts.verbose) std::cout << "  SLAM initialized\n";

                        auto slam_pose = slam.getPose();
                        Pose3D init_pose;
                        init_pose.x = static_cast<float>(slam_pose(0, 3));
                        init_pose.y = static_cast<float>(slam_pose(1, 3));
                        init_pose.yaw = std::atan2(
                            static_cast<float>(slam_pose(1, 0)),
                            static_cast<float>(slam_pose(0, 0)));
                        fusion.reset(init_pose);
                    }

                    // Check progressive localization
                    if (slam_initialized && processed > 0) {
                        LocalizationResult loc_result = slam.checkProgressiveLocalization();

                        switch (loc_result.status) {
                            case LocalizationStatus::ACCUMULATING:
                                if (opts.verbose && imu_count % 200 == 0) {
                                    std::cout << "  Accumulating: "
                                              << loc_result.local_map_voxels << " voxels, "
                                              << static_cast<int>(loc_result.rotation_deg) << " deg\n";
                                }
                                break;

                            case LocalizationStatus::READY_FOR_LOCALIZATION:
                            case LocalizationStatus::AWAITING_ROBOT_STOP: {
                                // Robot is "stopped" in replay - run global localization
                                if (!opts.json) std::cout << "  Running global localization...\n";
                                LocalizationResult result = slam.runGlobalLocalization();
                                attempts = result.attempt_number;

                                if (result.status == LocalizationStatus::SUCCESS) {
                                    localization_complete = true;
                                    localization_success = true;
                                    final_confidence = static_cast<float>(result.confidence);
                                    final_pose = result.pose;
                                    // swapToPrebuiltMap already called internally by runGlobalLocalization

                                    // Reset fusion with discovered global pose
                                    Eigen::Vector3d pos = result.pose.block<3, 1>(0, 3);
                                    Eigen::Matrix3d rot = result.pose.block<3, 3>(0, 0);
                                    Pose3D discovered;
                                    discovered.x = static_cast<float>(pos.x());
                                    discovered.y = static_cast<float>(pos.y());
                                    discovered.z = static_cast<float>(pos.z());
                                    discovered.yaw = static_cast<float>(std::atan2(rot(1, 0), rot(0, 0)));
                                    fusion.reset(discovered);
                                } else if (result.status == LocalizationStatus::FAILED) {
                                    localization_complete = true;
                                    localization_success = false;
                                }
                                break;
                            }

                            case LocalizationStatus::SUCCESS:
                                // Auto-triggered by checkProgressiveLocalization
                                localization_complete = true;
                                localization_success = true;
                                final_confidence = static_cast<float>(loc_result.confidence);
                                final_pose = loc_result.pose;
                                break;

                            case LocalizationStatus::FAILED:
                                localization_complete = true;
                                localization_success = false;
                                attempts = loc_result.attempt_number;
                                break;

                            default:
                                break;
                        }
                    }
                }
                break;
            }

            case RecordType::POINT_CLOUD_BATCH: {
                PointCloudBatchHeader batch;
                input.read(reinterpret_cast<char*>(&batch), sizeof(batch));
                if (!input.good()) break;

                PointCloud cloud;
                cloud.timestamp_ns = mono_timestamp_ns;  // Use monotonic time
                cloud.points.reserve(batch.num_points);

                for (uint32_t i = 0; i < batch.num_points; i++) {
                    LidarPointRecord lpr;
                    input.read(reinterpret_cast<char*>(&lpr), sizeof(lpr));
                    if (!input.good()) break;

                    LidarPoint pt;
                    pt.x = lpr.x;
                    pt.y = lpr.y;
                    pt.z = lpr.z;
                    pt.intensity = lpr.intensity;
                    pt.time_offset_ms = lpr.time_offset_ms;
                    pt.tag = lpr.tag;
                    pt.line = lpr.line;
                    cloud.points.push_back(pt);
                }

                if (!cloud.points.empty()) {
                    slam.addPointCloud(cloud);
                    scan_count++;
                }
                break;
            }

            case RecordType::VESC_ODOM: {
                VescOdomRecord rec;
                input.read(reinterpret_cast<char*>(&rec), sizeof(rec));
                if (!input.good()) break;
                vesc_count++;

                if (slam_initialized) {
                    fusion.updateWheelOdometry(
                        rec.linear_vel,
                        rec.angular_vel,
                        rec.erpm_left,
                        rec.erpm_right,
                        0.02f  // ~50Hz dt
                    );

                    auto slam_pose_d = slam.getPose();
                    Eigen::Matrix4f slam_pose = slam_pose_d.cast<float>();
                    fusion.updateSlamPose(slam_pose, header.timestamp_us);
                }
                break;
            }

            default:
                // Skip unknown record types
                if (header.size > 0) {
                    input.seekg(header.size, std::ios::cur);
                }
                break;
        }
    }

    auto replay_end = std::chrono::steady_clock::now();
    float replay_time = std::chrono::duration<float>(replay_end - replay_start).count();

    // ================================================================
    // Determine exit code
    // ================================================================
    int exit_code;
    std::string status_str;
    if (localization_success) {
        exit_code = EXIT_SUCCESS_LOC;
        status_str = "SUCCESS";
    } else if (localization_complete) {
        exit_code = EXIT_FAILED;
        status_str = "FAILED";
    } else {
        float elapsed = std::chrono::duration<float>(replay_end - replay_start).count();
        if (elapsed >= opts.max_time_s) {
            exit_code = EXIT_TIMEOUT;
            status_str = "TIMEOUT";
        } else {
            exit_code = EXIT_FAILED;
            status_str = "END_OF_FILE";
        }
    }

    // Extract final pose
    float pose_x = static_cast<float>(final_pose(0, 3));
    float pose_y = static_cast<float>(final_pose(1, 3));
    float pose_yaw_deg = static_cast<float>(
        std::atan2(final_pose(1, 0), final_pose(0, 0)) * 180.0 / 3.14159265);

    // ================================================================
    // Output results
    // ================================================================
    if (opts.json) {
        std::cout << "{\n";
        std::cout << "  \"status\": \"" << status_str << "\",\n";
        std::cout << "  \"confidence\": " << std::fixed << std::setprecision(3) << final_confidence << ",\n";
        std::cout << "  \"pose\": {\"x\": " << std::setprecision(3) << pose_x
                  << ", \"y\": " << pose_y
                  << ", \"yaw_deg\": " << pose_yaw_deg << "},\n";
        std::cout << "  \"replay_time_s\": " << std::setprecision(1) << replay_time << ",\n";
        std::cout << "  \"recording_time_s\": " << recording_duration_s << ",\n";
        std::cout << "  \"imu_count\": " << imu_count << ",\n";
        std::cout << "  \"scan_count\": " << scan_count << ",\n";
        std::cout << "  \"vesc_count\": " << vesc_count << ",\n";
        std::cout << "  \"attempts\": " << attempts << ",\n";
        std::cout << "  \"exit_code\": " << exit_code << "\n";
        std::cout << "}\n";
    } else {
        std::cout << "\n=====================================\n";
        std::cout << "  REPLAY LOCALIZATION RESULT\n";
        std::cout << "=====================================\n";
        std::cout << "  Status:     " << status_str << "\n";
        std::cout << "  Confidence: " << std::fixed << std::setprecision(1)
                  << (final_confidence * 100) << "%\n";
        std::cout << "  Pose:       x=" << std::setprecision(3) << pose_x
                  << " y=" << pose_y
                  << " yaw=" << std::setprecision(1) << pose_yaw_deg << " deg\n";
        std::cout << "  Time:       " << std::setprecision(1) << replay_time
                  << "s (replay) / " << recording_duration_s << "s (recording)\n";
        std::cout << "  IMU:        " << imu_count << " samples\n";
        std::cout << "  Scans:      " << scan_count << "\n";
        std::cout << "  VESC:       " << vesc_count << "\n";
        std::cout << "  Attempts:   " << attempts << "\n";
        std::cout << "=====================================\n";
    }

    return exit_code;
}
