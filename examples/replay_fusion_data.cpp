/**
 * Replay Fusion Data
 *
 * Replays recorded sensor data to test sensor fusion algorithm offline.
 * No hardware needed - reads from binary file created by record_fusion_data.cpp
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
#include <thread>
#include <vector>
#include <cmath>
#include <iomanip>

#include "slam/sensor_fusion.hpp"
#include "slam/slam_engine.hpp"
#include "slam/types.hpp"

using namespace slam;

// ============================================================================
// Record Format (must match record_fusion_data.cpp)
// ============================================================================

#pragma pack(push, 1)

enum class RecordType : uint8_t {
    IMU = 1,
    POINT_CLOUD_START = 2,
    POINT_CLOUD_POINT = 3,
    POINT_CLOUD_END = 4,
    VESC_STATUS = 5,
    MOTION_COMMAND = 6,
    WHEEL_ODOM = 7,
};

struct RecordHeader {
    uint64_t timestamp_us;
    RecordType type;
    uint32_t size;
};

struct ImuRecord {
    float acc_x, acc_y, acc_z;
    float gyro_x, gyro_y, gyro_z;
};

struct PointRecord {
    float x, y, z;
    float intensity;
    float time_offset_ms;
    uint8_t tag;
};

struct VescStatusRecord {
    int32_t erpm_left, erpm_right;
    int32_t tach_left, tach_right;
    float duty_left, duty_right;
    float voltage;
};

struct MotionCommandRecord {
    float linear_duty;
    float angular_duty;
};

struct WheelOdomRecord {
    float x, y, theta;
    float linear_vel, angular_vel;
};

#pragma pack(pop)

// ============================================================================
// Result logging
// ============================================================================

struct FusionResult {
    double timestamp_s;
    // Wheel odom
    float odom_x, odom_y, odom_theta;
    float odom_linear_vel, odom_angular_vel;
    // VESC
    int32_t erpm_left, erpm_right;
    // SLAM
    float slam_x, slam_y, slam_yaw;
    // Fused
    float fused_x, fused_y, fused_yaw;
    MotionState motion_state;
};

std::vector<FusionResult> g_results;

constexpr float RAD_TO_DEG = 180.0f / 3.14159265f;

const char* stateToStr(MotionState s) {
    switch (s) {
        case MotionState::STATIONARY: return "STAT";
        case MotionState::STRAIGHT_LINE: return "STRT";
        case MotionState::TURNING: return "TURN";
        default: return "????";
    }
}

int main(int argc, char* argv[]) {
    std::cout << "================================================\n";
    std::cout << "  Replay Fusion Data\n";
    std::cout << "================================================\n";

    std::string input_file = "fusion_recording.bin";
    std::string output_csv = "fusion_replay_results.csv";
    float playback_speed = 1.0f;
    bool realtime = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_csv = argv[++i];
        else if (arg == "--speed" && i + 1 < argc) playback_speed = std::stof(argv[++i]);
        else if (arg == "--realtime") realtime = true;
        else if (arg == "--help") {
            std::cout << "Usage: replay_fusion_data [options]\n"
                      << "  --input FILE    Input recording (default: fusion_recording.bin)\n"
                      << "  --output FILE   Output CSV (default: fusion_replay_results.csv)\n"
                      << "  --speed X       Playback speed multiplier (default: 1.0)\n"
                      << "  --realtime      Play in realtime (otherwise as fast as possible)\n";
            return 0;
        }
    }

    // Open input file
    std::ifstream input(input_file, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "Failed to open: " << input_file << std::endl;
        return 1;
    }

    // Read header
    char magic[4];
    uint32_t version;
    input.read(magic, 4);
    input.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (std::string(magic, 4) != "FUSN" || version != 1) {
        std::cerr << "Invalid file format!\n";
        return 1;
    }

    std::cout << "Loading: " << input_file << "\n";

    // ================================================================
    // Initialize SLAM Engine
    // ================================================================
    std::cout << "\nInitializing SLAM engine...\n";
    SlamEngine slam;

    SlamConfig slam_config;
    slam_config.filter_size_surf = 0.1;
    slam_config.filter_size_map = 0.1;
    slam_config.deskew_enabled = true;
    slam_config.gyr_cov = 0.1;
    slam_config.acc_cov = 0.1;
    slam_config.max_iterations = 4;

    if (!slam.init(slam_config)) {
        std::cerr << "Failed to initialize SLAM!\n";
        return 1;
    }

    // ================================================================
    // Initialize Sensor Fusion
    // ================================================================
    std::cout << "Initializing sensor fusion...\n";
    SensorFusion fusion;

    FusionConfig fusion_config;
    fusion_config.stationary_erpm_threshold = 50;
    fusion_config.turning_angular_threshold = 0.2f;
    // SLAM smoothing - OPTIMAL: 0.60/0.65 gave 0.1cm error
    fusion_config.slam_position_alpha = 0.60f;
    fusion_config.slam_heading_alpha = 0.65f;
    // Correction rates (per second) - OPTIMAL: 4.0/4.0/12.0 gave 0.1cm error
    fusion_config.straight_heading_correction = 4.0f;
    fusion_config.straight_position_correction = 4.0f;
    fusion_config.turning_position_correction = 12.0f;

    fusion.init(fusion_config);

    fusion.setMotionStateCallback([](MotionState old_s, MotionState new_s) {
        std::cout << "  [Fusion] " << stateToStr(old_s) << " -> " << stateToStr(new_s) << "\n";
    });

    // ================================================================
    // Process recording
    // ================================================================
    std::cout << "\nProcessing recording...\n";

    // Current state
    uint64_t current_cloud_ts = 0;
    std::vector<LidarPoint> cloud_points;
    cloud_points.reserve(20000);

    WheelOdomRecord latest_odom = {};
    VescStatusRecord latest_vesc = {};
    MotionCommandRecord latest_cmd = {};

    uint64_t last_timestamp_us = 0;
    uint64_t imu_count = 0, cloud_count = 0, odom_count = 0;
    bool slam_initialized = false;
    bool fusion_initialized = false;

    // Generate monotonic timestamps for SLAM (avoid loop-back issues)
    uint64_t mono_timestamp_ns = 0;
    const uint64_t IMU_PERIOD_NS = 5000000;  // 5ms = 200Hz
    const uint64_t SCAN_PERIOD_NS = 100000000;  // 100ms = 10Hz

    auto replay_start = std::chrono::steady_clock::now();

    while (input.good() && !input.eof()) {
        RecordHeader header;
        input.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!input.good()) break;

        // Realtime playback
        if (realtime && last_timestamp_us > 0) {
            uint64_t delta_us = header.timestamp_us - last_timestamp_us;
            if (delta_us > 0 && delta_us < 1000000) {  // Sanity check
                auto target_delay = std::chrono::microseconds(static_cast<long long>(delta_us / playback_speed));
                std::this_thread::sleep_for(target_delay);
            }
        }
        last_timestamp_us = header.timestamp_us;

        float dt = 0.02f;  // Assume 50Hz update rate

        switch (header.type) {
            case RecordType::IMU: {
                ImuRecord rec;
                input.read(reinterpret_cast<char*>(&rec), sizeof(rec));

                // Use monotonic timestamp to avoid SLAM "loop back" issues
                mono_timestamp_ns += IMU_PERIOD_NS;

                ImuData imu;
                imu.timestamp_ns = mono_timestamp_ns;
                imu.acc = V3D(rec.acc_x, rec.acc_y, rec.acc_z);
                imu.gyro = V3D(rec.gyro_x, rec.gyro_y, rec.gyro_z);
                slam.addImuData(imu);
                imu_count++;

                // Process SLAM periodically
                if (imu_count % 20 == 0) {
                    slam.process();

                    if (!slam_initialized && slam.isInitialized()) {
                        slam_initialized = true;
                        std::cout << "  SLAM initialized!\n";

                        // Initialize fusion with SLAM pose
                        auto slam_pose = slam.getPose();
                        Pose3D init_pose;
                        init_pose.x = static_cast<float>(slam_pose(0, 3));
                        init_pose.y = static_cast<float>(slam_pose(1, 3));
                        init_pose.z = static_cast<float>(slam_pose(2, 3));
                        init_pose.yaw = std::atan2(static_cast<float>(slam_pose(1, 0)),
                                                    static_cast<float>(slam_pose(0, 0)));
                        fusion.reset(init_pose);
                        fusion_initialized = true;
                    }
                }
                break;
            }

            case RecordType::POINT_CLOUD_START: {
                uint64_t cloud_ts;
                uint32_t num_points;
                input.read(reinterpret_cast<char*>(&cloud_ts), sizeof(cloud_ts));
                input.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
                // Use current monotonic timestamp for cloud
                current_cloud_ts = mono_timestamp_ns;
                cloud_points.clear();
                break;
            }

            case RecordType::POINT_CLOUD_POINT: {
                PointRecord rec;
                input.read(reinterpret_cast<char*>(&rec), sizeof(rec));

                LidarPoint pt;
                pt.x = rec.x;
                pt.y = rec.y;
                pt.z = rec.z;
                pt.intensity = rec.intensity;
                pt.time_offset_ms = rec.time_offset_ms;
                pt.tag = rec.tag;
                pt.line = 0;
                cloud_points.push_back(pt);
                break;
            }

            case RecordType::POINT_CLOUD_END: {
                if (!cloud_points.empty()) {
                    PointCloud cloud;
                    cloud.timestamp_ns = current_cloud_ts;
                    cloud.points = cloud_points;
                    slam.addPointCloud(cloud);
                    cloud_count++;
                }
                break;
            }

            case RecordType::VESC_STATUS: {
                input.read(reinterpret_cast<char*>(&latest_vesc), sizeof(latest_vesc));
                break;
            }

            case RecordType::MOTION_COMMAND: {
                input.read(reinterpret_cast<char*>(&latest_cmd), sizeof(latest_cmd));
                break;
            }

            case RecordType::WHEEL_ODOM: {
                input.read(reinterpret_cast<char*>(&latest_odom), sizeof(latest_odom));
                odom_count++;

                if (fusion_initialized) {
                    // Update fusion with wheel odometry
                    fusion.updateWheelOdometry(
                        latest_odom.linear_vel,
                        latest_odom.angular_vel,
                        latest_vesc.erpm_left,
                        latest_vesc.erpm_right,
                        dt
                    );

                    // Update fusion with SLAM pose
                    auto slam_pose_d = slam.getPose();
                    Eigen::Matrix4f slam_pose = slam_pose_d.cast<float>();
                    fusion.updateSlamPose(slam_pose, header.timestamp_us);

                    // Log result
                    FusionResult result;
                    result.timestamp_s = header.timestamp_us / 1e6;
                    result.odom_x = latest_odom.x;
                    result.odom_y = latest_odom.y;
                    result.odom_theta = latest_odom.theta;
                    result.odom_linear_vel = latest_odom.linear_vel;
                    result.odom_angular_vel = latest_odom.angular_vel;
                    result.erpm_left = latest_vesc.erpm_left;
                    result.erpm_right = latest_vesc.erpm_right;

                    float slam_yaw = std::atan2(slam_pose(1, 0), slam_pose(0, 0));
                    result.slam_x = slam_pose(0, 3);
                    result.slam_y = slam_pose(1, 3);
                    result.slam_yaw = slam_yaw;

                    Pose3D fused = fusion.getFusedPose();
                    result.fused_x = fused.x;
                    result.fused_y = fused.y;
                    result.fused_yaw = fused.yaw;
                    result.motion_state = fusion.getMotionState();

                    g_results.push_back(result);

                    // Print progress
                    if (odom_count % 50 == 0) {
                        std::cout << std::fixed << std::setprecision(3);
                        std::cout << "[" << stateToStr(result.motion_state) << "] "
                                  << "t=" << result.timestamp_s << "s "
                                  << "Odom(" << result.odom_x << "," << result.odom_y << ") "
                                  << "SLAM(" << result.slam_x << "," << result.slam_y << ") "
                                  << "Fused(" << result.fused_x << "," << result.fused_y << ")\n";
                    }
                }
                break;
            }

            default:
                // Skip unknown records
                if (header.size > 0) {
                    input.seekg(header.size, std::ios::cur);
                }
                break;
        }
    }

    auto replay_end = std::chrono::steady_clock::now();
    float replay_time = std::chrono::duration<float>(replay_end - replay_start).count();

    // ================================================================
    // Statistics
    // ================================================================
    std::cout << "\n========================================\n";
    std::cout << "  REPLAY STATISTICS\n";
    std::cout << "========================================\n";
    std::cout << "  Replay time: " << replay_time << "s\n";
    std::cout << "  IMU samples: " << imu_count << "\n";
    std::cout << "  Point clouds: " << cloud_count << "\n";
    std::cout << "  Odom samples: " << odom_count << "\n";

    if (!g_results.empty()) {
        const auto& final_r = g_results.back();

        // Count motion states
        int stat_count = 0, strt_count = 0, turn_count = 0;
        for (const auto& r : g_results) {
            if (r.motion_state == MotionState::STATIONARY) stat_count++;
            else if (r.motion_state == MotionState::STRAIGHT_LINE) strt_count++;
            else if (r.motion_state == MotionState::TURNING) turn_count++;
        }

        std::cout << "\n  Motion State Distribution:\n";
        std::cout << "    STATIONARY:   " << stat_count << "\n";
        std::cout << "    STRAIGHT_LINE: " << strt_count << "\n";
        std::cout << "    TURNING:      " << turn_count << "\n";

        // Calculate jitter (frame-to-frame position changes)
        // Jitter during STATIONARY shows noise rejection quality
        float slam_jitter_stat = 0, fused_jitter_stat = 0;
        float slam_jitter_all = 0, fused_jitter_all = 0;
        int jitter_stat_count = 0, jitter_all_count = 0;
        for (size_t i = 1; i < g_results.size(); i++) {
            const auto& prev = g_results[i - 1];
            const auto& curr = g_results[i];

            float slam_delta = std::sqrt(
                (curr.slam_x - prev.slam_x) * (curr.slam_x - prev.slam_x) +
                (curr.slam_y - prev.slam_y) * (curr.slam_y - prev.slam_y)
            );
            float fused_delta = std::sqrt(
                (curr.fused_x - prev.fused_x) * (curr.fused_x - prev.fused_x) +
                (curr.fused_y - prev.fused_y) * (curr.fused_y - prev.fused_y)
            );

            slam_jitter_all += slam_delta;
            fused_jitter_all += fused_delta;
            jitter_all_count++;

            if (curr.motion_state == MotionState::STATIONARY) {
                slam_jitter_stat += slam_delta;
                fused_jitter_stat += fused_delta;
                jitter_stat_count++;
            }
        }

        std::cout << "\n  Jitter (avg frame-to-frame delta):\n";
        std::cout << std::fixed << std::setprecision(2);
        if (jitter_stat_count > 0) {
            std::cout << "    STATIONARY - SLAM:  " << (slam_jitter_stat / jitter_stat_count * 1000) << " mm\n";
            std::cout << "    STATIONARY - Fused: " << (fused_jitter_stat / jitter_stat_count * 1000) << " mm\n";
        }
        if (jitter_all_count > 0) {
            std::cout << "    ALL -        SLAM:  " << (slam_jitter_all / jitter_all_count * 1000) << " mm\n";
            std::cout << "    ALL -        Fused: " << (fused_jitter_all / jitter_all_count * 1000) << " mm\n";
        }

        std::cout << "\n  Final Poses:\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "    Wheel odom: x=" << final_r.odom_x << " y=" << final_r.odom_y
                  << " yaw=" << (final_r.odom_theta * RAD_TO_DEG) << "deg\n";
        std::cout << "    SLAM:       x=" << final_r.slam_x << " y=" << final_r.slam_y
                  << " yaw=" << (final_r.slam_yaw * RAD_TO_DEG) << "deg\n";
        std::cout << "    Fused:      x=" << final_r.fused_x << " y=" << final_r.fused_y
                  << " yaw=" << (final_r.fused_yaw * RAD_TO_DEG) << "deg\n";

        float odom_err = std::sqrt(final_r.odom_x * final_r.odom_x + final_r.odom_y * final_r.odom_y);
        float slam_err = std::sqrt(final_r.slam_x * final_r.slam_x + final_r.slam_y * final_r.slam_y);
        float fused_err = std::sqrt(final_r.fused_x * final_r.fused_x + final_r.fused_y * final_r.fused_y);

        std::cout << "\n  Return-to-Origin Error:\n";
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "    Wheel odom: " << (odom_err * 100) << " cm\n";
        std::cout << "    SLAM:       " << (slam_err * 100) << " cm\n";
        std::cout << "    Fused:      " << (fused_err * 100) << " cm\n";
    }
    std::cout << "========================================\n";

    // ================================================================
    // Save results to CSV
    // ================================================================
    std::ofstream csv(output_csv);
    csv << "timestamp_s,odom_x,odom_y,odom_theta,odom_linear_vel,odom_angular_vel,"
        << "erpm_left,erpm_right,slam_x,slam_y,slam_yaw,fused_x,fused_y,fused_yaw,motion_state\n";
    for (const auto& r : g_results) {
        csv << std::fixed << std::setprecision(6)
            << r.timestamp_s << "," << r.odom_x << "," << r.odom_y << "," << r.odom_theta << ","
            << r.odom_linear_vel << "," << r.odom_angular_vel << ","
            << r.erpm_left << "," << r.erpm_right << ","
            << r.slam_x << "," << r.slam_y << "," << r.slam_yaw << ","
            << r.fused_x << "," << r.fused_y << "," << r.fused_yaw << ","
            << stateToStr(r.motion_state) << "\n";
    }
    std::cout << "\nResults saved to: " << output_csv << "\n";

    return 0;
}
