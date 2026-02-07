/**
 * Record Fusion Data
 *
 * Records raw sensor data for offline sensor fusion development:
 * - LiDAR point clouds (with timestamps)
 * - IMU data (accelerometer, gyroscope)
 * - VESC status (ERPM, tachometer, duty)
 * - Motion commands (linear, angular duty)
 *
 * Output: Binary file that can be replayed with replay_fusion_data.cpp
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
#include <atomic>
#include <csignal>
#include <mutex>
#include <vector>
#include <cmath>
#include <cstring>
#include <iomanip>

#include "slam/vesc_driver.hpp"
#include "slam/motion_controller.hpp"
#include "slam/livox_mid360.hpp"
#include "slam/types.hpp"
#include "slam/gamepad.hpp"

using namespace slam;

// ============================================================================
// Record Format
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
    uint32_t size;  // Size of payload following this header
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
// Global State
// ============================================================================

std::atomic<bool> g_running{true};
std::ofstream g_output;
std::mutex g_output_mutex;
uint64_t g_start_time_us = 0;

// Motion parameters (same as test_sensor_fusion)
constexpr float DRIVE_DUTY = 0.8f;
constexpr float TURN_DUTY = 0.6f;
constexpr float MAX_DUTY = 0.12f;
constexpr float RAMP_RATE = 0.4f;
constexpr float TARGET_DISTANCE_M = 1.0f;
constexpr float TARGET_ANGLE_DEG = 90.0f;
constexpr float DEG_TO_RAD = 3.14159265f / 180.0f;

void signalHandler(int) {
    g_running = false;
}

uint64_t nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t relativeUs() {
    return nowUs() - g_start_time_us;
}

template<typename T>
void writeRecord(RecordType type, const T& data) {
    std::lock_guard<std::mutex> lock(g_output_mutex);
    RecordHeader header;
    header.timestamp_us = relativeUs();
    header.type = type;
    header.size = sizeof(T);
    g_output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    g_output.write(reinterpret_cast<const char*>(&data), sizeof(T));
}

void writePointCloudStart(uint64_t cloud_timestamp_ns, uint32_t num_points) {
    std::lock_guard<std::mutex> lock(g_output_mutex);
    RecordHeader header;
    header.timestamp_us = relativeUs();
    header.type = RecordType::POINT_CLOUD_START;
    header.size = sizeof(uint64_t) + sizeof(uint32_t);
    g_output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    g_output.write(reinterpret_cast<const char*>(&cloud_timestamp_ns), sizeof(uint64_t));
    g_output.write(reinterpret_cast<const char*>(&num_points), sizeof(uint32_t));
}

void writePointCloudEnd() {
    std::lock_guard<std::mutex> lock(g_output_mutex);
    RecordHeader header;
    header.timestamp_us = relativeUs();
    header.type = RecordType::POINT_CLOUD_END;
    header.size = 0;
    g_output.write(reinterpret_cast<const char*>(&header), sizeof(header));
}

float normalizeAngle(float angle) {
    while (angle > 3.14159265f) angle -= 2.0f * 3.14159265f;
    while (angle < -3.14159265f) angle += 2.0f * 3.14159265f;
    return angle;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================\n";
    std::cout << "  Record Fusion Data\n";
    std::cout << "================================================\n";

    // Configuration
    std::string can_port = "COM3";
    std::string cal_file = "vesc_calibration.ini";
    std::string device_ip = "";
    std::string host_ip = "192.168.1.50";
    std::string output_file = "fusion_recording.bin";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--can" && i + 1 < argc) can_port = argv[++i];
        else if (arg == "--cal" && i + 1 < argc) cal_file = argv[++i];
        else if (arg == "--device" && i + 1 < argc) device_ip = argv[++i];
        else if (arg == "--host" && i + 1 < argc) host_ip = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "--help") {
            std::cout << "Usage: record_fusion_data [options]\n"
                      << "  --can PORT      CAN port (default: COM3)\n"
                      << "  --cal FILE      Calibration file\n"
                      << "  --device IP     LiDAR IP (default: auto-discover)\n"
                      << "  --host IP       Host IP (default: 192.168.1.50)\n"
                      << "  --output FILE   Output file (default: fusion_recording.bin)\n";
            return 0;
        }
    }

    std::signal(SIGINT, signalHandler);

    // Open output file
    g_output.open(output_file, std::ios::binary);
    if (!g_output.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return 1;
    }

    // Write file header
    const char magic[] = "FUSN";
    uint32_t version = 1;
    g_output.write(magic, 4);
    g_output.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // ================================================================
    // Initialize VESC Driver + Motion Controller
    // ================================================================
    std::cout << "Initializing VESC driver on " << can_port << "...\n";
    VescDriver vesc;
    if (!vesc.init(can_port, 1, 126)) {
        std::cerr << "Failed to connect to VESCs!\n";
        return 1;
    }

    MotionController motion;
    if (!motion.init(&vesc, cal_file)) {
        std::cerr << "  Warning: Using default calibration\n";
    }
    motion.setMaxDuty(MAX_DUTY);
    motion.setRampRate(RAMP_RATE);
    std::cout << "  Motion controller ready\n";

    // ================================================================
    // Initialize LiDAR
    // ================================================================
    std::cout << "\nInitializing LiDAR...\n";
    LivoxMid360 lidar;

    if (device_ip.empty()) {
        auto devices = lidar.discover(3000, host_ip);
        if (devices.empty()) {
            std::cerr << "  No LiDAR found! Use --device IP\n";
            vesc.shutdown();
            return 1;
        }
        device_ip = devices[0].ip_address;
        std::cout << "  Found: " << device_ip << "\n";
    }

    if (!lidar.connect(device_ip, host_ip)) {
        std::cerr << "  Failed to connect to LiDAR!\n";
        vesc.shutdown();
        return 1;
    }

    // Point cloud accumulation
    std::mutex scan_mutex;
    std::vector<PointRecord> accumulated_points;
    uint64_t scan_start_time = 0;
    const double SCAN_PERIOD_MS = 100.0;
    accumulated_points.reserve(20000);

    // ================================================================
    // Set up callbacks
    // ================================================================
    std::atomic<uint64_t> imu_count{0}, point_count{0};

    // IMU callback
    lidar.setIMUCallback([&](const LivoxIMUFrame& frame) {
        constexpr double G_M_S2 = 9.81;
        ImuRecord rec;
        rec.acc_x = static_cast<float>(frame.accel.x() * G_M_S2);
        rec.acc_y = static_cast<float>(frame.accel.y() * G_M_S2);
        rec.acc_z = static_cast<float>(frame.accel.z() * G_M_S2);
        rec.gyro_x = static_cast<float>(frame.gyro.x());
        rec.gyro_y = static_cast<float>(frame.gyro.y());
        rec.gyro_z = static_cast<float>(frame.gyro.z());
        writeRecord(RecordType::IMU, rec);
        imu_count++;
    });

    // Point cloud callback
    lidar.setPointCloudCallback([&](const LivoxPointCloudFrame& frame) {
        std::lock_guard<std::mutex> lock(scan_mutex);

        if (accumulated_points.empty()) {
            scan_start_time = frame.timestamp_ns;
        }

        double frame_offset_ms = (frame.timestamp_ns - scan_start_time) / 1e6;

        static size_t valid_counter = 0;
        for (size_t i = 0; i < frame.points.size(); i++) {
            const V3D& p = frame.points[i];

            // Tag filtering
            if (!frame.tags.empty()) {
                uint8_t tag = frame.tags[i];
                uint8_t return_type = tag & 0x30;
                if (return_type != 0x00 && return_type != 0x10) continue;
            }

            // Range check
            double range = p.norm();
            if (range < 0.5 || range > 50.0) continue;

            // Point filtering (keep every 3rd)
            valid_counter++;
            if ((valid_counter % 3) != 0) continue;

            PointRecord rec;
            rec.x = static_cast<float>(p.x());
            rec.y = static_cast<float>(p.y());
            rec.z = static_cast<float>(p.z());
            rec.intensity = frame.reflectivities.empty() ? 100.0f : static_cast<float>(frame.reflectivities[i]);
            float point_offset_ms = (i < frame.time_offsets_us.size()) ?
                (frame.time_offsets_us[i] / 1000.0f) : 0.0f;
            rec.time_offset_ms = static_cast<float>(frame_offset_ms) + point_offset_ms;
            rec.tag = frame.tags.empty() ? 0 : frame.tags[i];

            accumulated_points.push_back(rec);
        }
        point_count += frame.points.size();

        // Write complete scan
        if (frame_offset_ms >= SCAN_PERIOD_MS && !accumulated_points.empty()) {
            writePointCloudStart(scan_start_time, static_cast<uint32_t>(accumulated_points.size()));
            for (const auto& pt : accumulated_points) {
                writeRecord(RecordType::POINT_CLOUD_POINT, pt);
            }
            writePointCloudEnd();

            accumulated_points.clear();
            accumulated_points.reserve(20000);
        }
    });

    // Start streaming
    std::cout << "\nStarting LiDAR streaming...\n";
    if (!lidar.startStreaming()) {
        std::cerr << "Failed to start LiDAR!\n";
        vesc.shutdown();
        return 1;
    }

    // Wait for IMU data
    std::cout << "Waiting for sensor data (keep robot still)...\n";
    g_start_time_us = nowUs();

    auto init_start = std::chrono::steady_clock::now();
    while (imu_count < 200 && g_running) {  // ~1 second of IMU at 200Hz
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - init_start).count();
        if (elapsed > 5.0f) {
            std::cerr << "Timeout waiting for IMU data!\n";
            break;
        }
        std::cout << "  IMU: " << imu_count << " | Points: " << point_count << "\r" << std::flush;
    }
    std::cout << "\nSensors ready!                          \n";

    // ================================================================
    // Initialize Gamepad
    // ================================================================
    std::cout << "\nInitializing gamepad...\n";
    Gamepad gamepad;
    if (!gamepad.init()) {
        std::cerr << "No gamepad found! Connect PS5 controller.\n";
        lidar.stop();
        vesc.shutdown();
        return 1;
    }
    std::cout << "  Gamepad: " << gamepad.getControllerName() << "\n";
    gamepad.setLEDColor(0, 255, 0);  // Green = ready

    // Gamepad drive config (same as manual_drive)
    GamepadDriveConfig drive_config;
    drive_config.max_linear_velocity = 0.6f;
    drive_config.max_angular_velocity = 3.5f;
    drive_config.stick_deadzone = 0.12f;
    drive_config.max_speed_scale = 0.8f;
    drive_config.use_trigger_boost = false;
    drive_config.right_stick_steer = false;  // Single stick mode

    // Allow higher duty for faster driving
    motion.setMaxDuty(0.30f);

    // Lambda to record VESC status and wheel odom
    auto recordState = [&](float linear_vel, float angular_vel) {
        auto status_left = vesc.getStatus(1);
        auto status_right = vesc.getStatus(126);

        VescStatusRecord vesc_rec;
        vesc_rec.erpm_left = status_left.erpm;
        vesc_rec.erpm_right = status_right.erpm;
        vesc_rec.tach_left = status_left.tachometer;
        vesc_rec.tach_right = status_right.tachometer;
        vesc_rec.duty_left = status_left.duty;
        vesc_rec.duty_right = status_right.duty;
        vesc_rec.voltage = status_left.voltage;
        writeRecord(RecordType::VESC_STATUS, vesc_rec);

        MotionCommandRecord cmd_rec;
        cmd_rec.linear_duty = linear_vel;
        cmd_rec.angular_duty = angular_vel;
        writeRecord(RecordType::MOTION_COMMAND, cmd_rec);

        Pose2D odom = motion.getPose();
        Velocity2D vel = motion.getVelocity();
        WheelOdomRecord odom_rec;
        odom_rec.x = odom.x;
        odom_rec.y = odom.y;
        odom_rec.theta = odom.theta;
        odom_rec.linear_vel = vel.linear;
        odom_rec.angular_vel = vel.angular;
        writeRecord(RecordType::WHEEL_ODOM, odom_rec);
    };

    // ================================================================
    // Ready to record - Gamepad controlled
    // ================================================================
    std::cout << "\n========================================\n";
    std::cout << "  MANUAL RECORDING MODE\n";
    std::cout << "========================================\n";
    std::cout << "  Left Stick: Drive & Steer\n";
    std::cout << "  L1: Emergency stop\n";
    std::cout << "  Options: Stop recording\n";
    std::cout << "\n  Recording to: " << output_file << "\n";
    std::cout << "========================================\n\n";

    motion.resetPose();
    auto last_time = std::chrono::steady_clock::now();
    auto start_time = last_time;
    bool e_stop_active = false;

    gamepad.setLEDColor(255, 0, 0);  // Red = recording
    gamepad.rumble(0.3f, 0.3f, 200);

    while (g_running && gamepad.isConnected()) {
        gamepad.update();
        auto state = gamepad.getState();

        // Exit on Options button
        if (state.button_start) {
            std::cout << "\nOptions pressed - stopping recording.\n";
            break;
        }

        // Emergency stop
        if (state.left_shoulder) {
            if (!e_stop_active) {
                motion.emergencyStop();
                gamepad.setLEDColor(255, 255, 0);  // Yellow
                e_stop_active = true;
            }
        } else {
            if (e_stop_active) {
                gamepad.setLEDColor(255, 0, 0);  // Back to red
                e_stop_active = false;
            }
        }

        // Get drive command from gamepad
        auto cmd = gamepad.getDriveCommand(drive_config);

        // Calculate dt
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        float elapsed = std::chrono::duration<float>(now - start_time).count();
        last_time = now;

        // Update motion (unless e-stopped)
        if (!e_stop_active) {
            motion.setVelocity(cmd.linear_velocity, cmd.angular_velocity);
            motion.update(dt);
        }

        // Record state
        recordState(cmd.linear_velocity, cmd.angular_velocity);

        // Print status
        Pose2D pose = motion.getPose();
        static int print_counter = 0;
        if (++print_counter >= 12) {  // ~4Hz
            print_counter = 0;
            std::cout << "\r  Time:" << std::fixed << std::setprecision(0) << std::setw(4) << elapsed << "s"
                      << "  Lin:" << std::setprecision(2) << std::setw(6) << cmd.linear_velocity << " m/s"
                      << "  Pos:(" << std::setw(6) << pose.x << "," << std::setw(6) << pose.y << ") m"
                      << "  IMU:" << imu_count << "  Pts:" << point_count/1000 << "k     " << std::flush;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // Stop motors
    motion.emergencyStop();
    gamepad.setLEDColor(0, 255, 0);  // Green = done
    // Final pose
    Pose2D final_pose = motion.getPose();
    float return_error = std::sqrt(final_pose.x*final_pose.x + final_pose.y*final_pose.y);
    std::cout << "\nFinal wheel odom: x=" << final_pose.x << " y=" << final_pose.y
              << " return_error=" << (return_error * 100) << "cm\n";

    // Cleanup
    std::cout << "\nStopping LiDAR...\n";
    lidar.stop();

    std::cout << "Stopping motors...\n";
    vesc.shutdown();

    gamepad.shutdown();

    // Close output file
    g_output.close();
    std::cout << "\nRecording saved to: " << output_file << "\n";
    std::cout << "  IMU samples: " << imu_count << "\n";
    std::cout << "  Point samples: " << point_count << "\n";

    return 0;
}
