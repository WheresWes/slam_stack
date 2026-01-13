/**
 * Sensor Fusion Test
 *
 * Tests the SensorFusion class that combines wheel odometry and SLAM:
 * - STATIONARY: Freeze pose (reject SLAM jitter)
 * - STRAIGHT_LINE: Wheel dead-reckoning + slow SLAM correction
 * - TURNING: Trust SLAM for rotation (wheel slip unreliable)
 *
 * Runs the same motion sequence as test_odom_vs_slam.cpp and compares:
 * 1. Raw wheel odometry
 * 2. Raw SLAM pose
 * 3. Fused pose (our output)
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
#include <iomanip>

#include "slam/vesc_driver.hpp"
#include "slam/motion_controller.hpp"
#include "slam/sensor_fusion.hpp"
#include "slam/livox_mid360.hpp"
#include "slam/slam_engine.hpp"
#include "slam/types.hpp"

#ifdef HAS_RERUN
#include "slam/visualization_interface.hpp"
#endif

using namespace slam;

// Constants
constexpr float DEG_TO_RAD = static_cast<float>(PI) / 180.0f;
constexpr float RAD_TO_DEG = 180.0f / static_cast<float>(PI);

// Motion parameters
constexpr float DRIVE_DUTY = 0.8f;
constexpr float TURN_DUTY = 0.6f;
constexpr float MAX_DUTY = 0.12f;
constexpr float RAMP_RATE = 0.4f;
constexpr float UPDATE_RATE_HZ = 50.0f;

// Targets
constexpr float TARGET_DISTANCE_M = 1.0f;
constexpr float TARGET_ANGLE_DEG = 90.0f;
constexpr float DISTANCE_TOLERANCE_M = 0.02f;
constexpr float ANGLE_TOLERANCE_DEG = 3.0f;

// Timeouts
constexpr float DRIVE_TIMEOUT_S = 20.0f;
constexpr float TURN_TIMEOUT_S = 15.0f;

// Global state
std::atomic<bool> g_running{true};
std::unique_ptr<SlamEngine> g_slam;
std::unique_ptr<SensorFusion> g_fusion;

// SLAM data buffers
std::mutex g_scan_mutex;
std::vector<LidarPoint> g_accumulated_points;
uint64_t g_scan_start_time = 0;
const double SCAN_PERIOD_MS = 100.0;

// Pose log entry (extended for fusion)
struct FusionLogEntry {
    double timestamp_s;
    std::string phase;
    MotionState motion_state;
    // Wheel odometry
    float odom_x, odom_y, odom_theta;
    // Raw SLAM
    float slam_x, slam_y, slam_z, slam_yaw;
    // Smoothed SLAM
    float slam_smooth_x, slam_smooth_y, slam_smooth_yaw;
    // Fused output
    float fused_x, fused_y, fused_z, fused_yaw;
    // Errors vs truth (using SLAM as ground truth)
    float odom_error, fused_error;
};

std::vector<FusionLogEntry> g_log;
std::mutex g_log_mutex;

void signalHandler(int) {
    g_running = false;
}

float normalizeAngle(float angle) {
    while (angle > static_cast<float>(PI)) angle -= 2.0f * static_cast<float>(PI);
    while (angle < -static_cast<float>(PI)) angle += 2.0f * static_cast<float>(PI);
    return angle;
}

const char* motionStateToString(MotionState state) {
    switch (state) {
        case MotionState::STATIONARY: return "STAT";
        case MotionState::STRAIGHT_LINE: return "STRT";
        case MotionState::TURNING: return "TURN";
        default: return "????";
    }
}

void logPoses(const std::string& phase, const Pose2D& odom) {
    if (!g_slam || !g_fusion) return;

    static auto start_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    double timestamp = std::chrono::duration<double>(now - start_time).count();

    // Get SLAM pose
    auto slam_pose = g_slam->getPose();
    float slam_x = slam_pose(0, 3);
    float slam_y = slam_pose(1, 3);
    float slam_z = slam_pose(2, 3);
    float slam_yaw = std::atan2(slam_pose(1,0), slam_pose(0,0));

    // Get fusion data
    Pose3D fused = g_fusion->getFusedPose();
    Pose3D slam_smooth = g_fusion->getSmoothedSlamPose();
    MotionState state = g_fusion->getMotionState();

    // Compute errors (using smoothed SLAM as reference)
    float odom_error = std::sqrt(std::pow(slam_smooth.x - odom.x, 2) +
                                  std::pow(slam_smooth.y - odom.y, 2));
    float fused_error = std::sqrt(std::pow(slam_smooth.x - fused.x, 2) +
                                   std::pow(slam_smooth.y - fused.y, 2));

    FusionLogEntry entry;
    entry.timestamp_s = timestamp;
    entry.phase = phase;
    entry.motion_state = state;
    entry.odom_x = odom.x;
    entry.odom_y = odom.y;
    entry.odom_theta = odom.theta;
    entry.slam_x = slam_x;
    entry.slam_y = slam_y;
    entry.slam_z = slam_z;
    entry.slam_yaw = slam_yaw;
    entry.slam_smooth_x = slam_smooth.x;
    entry.slam_smooth_y = slam_smooth.y;
    entry.slam_smooth_yaw = slam_smooth.yaw;
    entry.fused_x = fused.x;
    entry.fused_y = fused.y;
    entry.fused_z = fused.z;
    entry.fused_yaw = fused.yaw;
    entry.odom_error = odom_error;
    entry.fused_error = fused_error;

    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log.push_back(entry);
}

void printComparison(const Pose2D& odom) {
    if (!g_slam || !g_fusion) return;

    auto slam_pose = g_slam->getPose();
    Pose3D fused = g_fusion->getFusedPose();
    MotionState state = g_fusion->getMotionState();

    float slam_x = slam_pose(0, 3);
    float slam_y = slam_pose(1, 3);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[" << motionStateToString(state) << "] ";
    std::cout << "Odom(" << odom.x << "," << odom.y << ") ";
    std::cout << "SLAM(" << slam_x << "," << slam_y << ") ";
    std::cout << "Fused(" << fused.x << "," << fused.y << ")\n";
}

void saveLog(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to save log to " << filename << std::endl;
        return;
    }

    // Header
    file << "timestamp_s,phase,motion_state,";
    file << "odom_x,odom_y,odom_theta,";
    file << "slam_x,slam_y,slam_z,slam_yaw,";
    file << "slam_smooth_x,slam_smooth_y,slam_smooth_yaw,";
    file << "fused_x,fused_y,fused_z,fused_yaw,";
    file << "odom_error,fused_error\n";

    std::lock_guard<std::mutex> lock(g_log_mutex);
    for (const auto& e : g_log) {
        file << std::fixed << std::setprecision(6)
             << e.timestamp_s << "," << e.phase << "," << motionStateToString(e.motion_state) << ","
             << e.odom_x << "," << e.odom_y << "," << e.odom_theta << ","
             << e.slam_x << "," << e.slam_y << "," << e.slam_z << "," << e.slam_yaw << ","
             << e.slam_smooth_x << "," << e.slam_smooth_y << "," << e.slam_smooth_yaw << ","
             << e.fused_x << "," << e.fused_y << "," << e.fused_z << "," << e.fused_yaw << ","
             << e.odom_error << "," << e.fused_error << "\n";
    }

    std::cout << "Log saved to " << filename << " (" << g_log.size() << " entries)\n";
}

void printStatistics() {
    std::lock_guard<std::mutex> lock(g_log_mutex);

    if (g_log.empty()) {
        std::cout << "No data logged.\n";
        return;
    }

    // Compute jitter during stationary periods
    float max_stationary_jitter = 0;
    int stationary_count = 0;
    Pose3D first_stationary;
    bool found_first = false;

    // Compute errors by motion state
    float sum_odom_error_straight = 0, sum_fused_error_straight = 0;
    int straight_count = 0;
    float sum_odom_error_turn = 0, sum_fused_error_turn = 0;
    int turn_count = 0;

    for (const auto& e : g_log) {
        if (e.motion_state == MotionState::STATIONARY) {
            if (!found_first) {
                first_stationary.x = e.fused_x;
                first_stationary.y = e.fused_y;
                found_first = true;
            } else {
                float jitter = std::sqrt(std::pow(e.fused_x - first_stationary.x, 2) +
                                         std::pow(e.fused_y - first_stationary.y, 2));
                if (jitter > max_stationary_jitter) max_stationary_jitter = jitter;
            }
            stationary_count++;
        } else if (e.motion_state == MotionState::STRAIGHT_LINE) {
            sum_odom_error_straight += e.odom_error;
            sum_fused_error_straight += e.fused_error;
            straight_count++;
        } else if (e.motion_state == MotionState::TURNING) {
            sum_odom_error_turn += e.odom_error;
            sum_fused_error_turn += e.fused_error;
            turn_count++;
        }
    }

    // Final pose comparison
    const auto& final_entry = g_log.back();

    std::cout << "\n========================================\n";
    std::cout << "  SENSOR FUSION STATISTICS\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "\n  Motion State Distribution:\n";
    std::cout << "    STATIONARY:   " << stationary_count << " samples\n";
    std::cout << "    STRAIGHT_LINE: " << straight_count << " samples\n";
    std::cout << "    TURNING:      " << turn_count << " samples\n";

    std::cout << "\n  Stationary Jitter Rejection:\n";
    std::cout << "    Max fused deviation: " << (max_stationary_jitter * 1000) << " mm\n";
    std::cout << "    (Should be ~0 if fusion is working)\n";

    if (straight_count > 0) {
        std::cout << "\n  Straight-Line Mean Errors:\n";
        std::cout << "    Wheel odom: " << ((sum_odom_error_straight / straight_count) * 100) << " cm\n";
        std::cout << "    Fused:      " << ((sum_fused_error_straight / straight_count) * 100) << " cm\n";
    }

    if (turn_count > 0) {
        std::cout << "\n  Turning Mean Errors:\n";
        std::cout << "    Wheel odom: " << ((sum_odom_error_turn / turn_count) * 100) << " cm\n";
        std::cout << "    Fused:      " << ((sum_fused_error_turn / turn_count) * 100) << " cm\n";
    }

    std::cout << "\n  Final Poses:\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "    Wheel odom: x=" << final_entry.odom_x << " y=" << final_entry.odom_y
              << " yaw=" << (final_entry.odom_theta * RAD_TO_DEG) << "deg\n";
    std::cout << "    SLAM:       x=" << final_entry.slam_x << " y=" << final_entry.slam_y
              << " yaw=" << (final_entry.slam_yaw * RAD_TO_DEG) << "deg\n";
    std::cout << "    Fused:      x=" << final_entry.fused_x << " y=" << final_entry.fused_y
              << " yaw=" << (final_entry.fused_yaw * RAD_TO_DEG) << "deg\n";

    // Return-to-origin errors
    float odom_return = std::sqrt(final_entry.odom_x*final_entry.odom_x + final_entry.odom_y*final_entry.odom_y);
    float slam_return = std::sqrt(final_entry.slam_x*final_entry.slam_x + final_entry.slam_y*final_entry.slam_y);
    float fused_return = std::sqrt(final_entry.fused_x*final_entry.fused_x + final_entry.fused_y*final_entry.fused_y);

    std::cout << "\n  Return-to-Origin Error:\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "    Wheel odom: " << (odom_return * 100) << " cm\n";
    std::cout << "    SLAM:       " << (slam_return * 100) << " cm\n";
    std::cout << "    Fused:      " << (fused_return * 100) << " cm\n";

    std::cout << "========================================\n";
}

int main(int argc, char* argv[]) {
    std::cout << "================================================\n";
    std::cout << "  Sensor Fusion Test\n";
    std::cout << "================================================\n";
    std::cout << "Tests: STATIONARY (freeze) | STRAIGHT (wheel DR) | TURNING (SLAM)\n\n";

    // Configuration
    std::string can_port = "COM3";
    std::string cal_file = "vesc_calibration.ini";
    std::string device_ip = "";
    std::string host_ip = "192.168.1.50";
    std::string output_prefix = "sensor_fusion";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--can" && i + 1 < argc) can_port = argv[++i];
        else if (arg == "--cal" && i + 1 < argc) cal_file = argv[++i];
        else if (arg == "--device" && i + 1 < argc) device_ip = argv[++i];
        else if (arg == "--host" && i + 1 < argc) host_ip = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_prefix = argv[++i];
        else if (arg == "--help") {
            std::cout << "Usage: test_sensor_fusion [options]\n"
                      << "  --can PORT      CAN port (default: COM3)\n"
                      << "  --cal FILE      Calibration file (default: vesc_calibration.ini)\n"
                      << "  --device IP     LiDAR IP (default: auto-discover)\n"
                      << "  --host IP       Host IP (default: 192.168.1.50)\n"
                      << "  --output NAME   Output prefix (default: sensor_fusion)\n";
            return 0;
        }
    }

    std::signal(SIGINT, signalHandler);

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
    // Initialize Sensor Fusion
    // ================================================================
    std::cout << "\nInitializing sensor fusion...\n";
    g_fusion = std::make_unique<SensorFusion>();

    FusionConfig fusion_config;
    fusion_config.stationary_erpm_threshold = 50;
    fusion_config.turning_angular_threshold = 0.2f;  // Increased: 0.2 rad/s = ~11 deg/s
    fusion_config.slam_position_alpha = 0.15f;
    fusion_config.slam_heading_alpha = 0.20f;
    fusion_config.straight_heading_correction = 1.0f;  // Faster heading correction
    fusion_config.straight_position_correction = 0.5f; // Faster position correction
    fusion_config.turning_position_correction = 3.0f;

    g_fusion->init(fusion_config);

    // Set motion state callback for debugging
    g_fusion->setMotionStateCallback([](MotionState old_state, MotionState new_state) {
        std::cout << "  [Fusion] " << motionStateToString(old_state)
                  << " -> " << motionStateToString(new_state) << "\n";
    });

    std::cout << "  Fusion config:\n";
    std::cout << "    Position smoothing: " << fusion_config.slam_position_alpha << "\n";
    std::cout << "    Heading smoothing: " << fusion_config.slam_heading_alpha << "\n";
    std::cout << "    Straight correction: " << fusion_config.straight_position_correction << "/s\n";

    // ================================================================
    // Initialize SLAM Engine
    // ================================================================
    std::cout << "\nInitializing SLAM engine...\n";
    g_slam = std::make_unique<SlamEngine>();

    SlamConfig slam_config;
    slam_config.filter_size_surf = 0.1;
    slam_config.filter_size_map = 0.1;
    slam_config.deskew_enabled = true;
    slam_config.gyr_cov = 0.1;
    slam_config.acc_cov = 0.1;
    slam_config.max_iterations = 4;

    if (!g_slam->init(slam_config)) {
        std::cerr << "Failed to initialize SLAM engine!\n";
        vesc.shutdown();
        return 1;
    }

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
        std::cout << "  Found: " << devices[0].getTypeName() << " at " << device_ip << "\n";
    }

    if (!lidar.connect(device_ip, host_ip)) {
        std::cerr << "  Failed to connect to LiDAR!\n";
        vesc.shutdown();
        return 1;
    }

    // Preprocessor config
    PreprocessConfig pre_config;
    pre_config.lidar_type = LidarType::LIVOX_MID360;
    pre_config.blind_distance = 0.5;
    pre_config.max_distance = 50.0;
    pre_config.point_filter_num = 3;

    g_accumulated_points.reserve(20000);

    // ================================================================
    // Set up callbacks
    // ================================================================
    std::atomic<uint64_t> imu_count{0};

    // IMU callback
    lidar.setIMUCallback([&](const LivoxIMUFrame& frame) {
        constexpr double G_M_S2 = 9.81;

        ImuData imu;
        imu.timestamp_ns = frame.timestamp_ns;
        imu.acc = frame.accel * G_M_S2;
        imu.gyro = frame.gyro;

        g_slam->addImuData(imu);
        imu_count++;
    });

    // Point cloud callback
    lidar.setPointCloudCallback([&](const LivoxPointCloudFrame& frame) {
        std::lock_guard<std::mutex> lock(g_scan_mutex);

        if (g_accumulated_points.empty()) {
            g_scan_start_time = frame.timestamp_ns;
        }

        double frame_offset_ms = (frame.timestamp_ns - g_scan_start_time) / 1e6;

        static size_t valid_counter = 0;
        for (size_t i = 0; i < frame.points.size(); i++) {
            const V3D& p = frame.points[i];

            if (!frame.tags.empty()) {
                uint8_t tag = frame.tags[i];
                uint8_t return_type = tag & 0x30;
                if (return_type != 0x00 && return_type != 0x10) continue;
            }

            double range = p.norm();
            if (range < pre_config.blind_distance || range > pre_config.max_distance) continue;

            valid_counter++;
            if ((valid_counter % pre_config.point_filter_num) != 0) continue;

            LidarPoint lp;
            lp.x = static_cast<float>(p.x());
            lp.y = static_cast<float>(p.y());
            lp.z = static_cast<float>(p.z());
            lp.intensity = frame.reflectivities.empty() ? 100.0f : static_cast<float>(frame.reflectivities[i]);
            float point_offset_ms = (i < frame.time_offsets_us.size()) ?
                (frame.time_offsets_us[i] / 1000.0f) : 0.0f;
            lp.time_offset_ms = static_cast<float>(frame_offset_ms) + point_offset_ms;
            lp.tag = frame.tags.empty() ? 0 : frame.tags[i];
            lp.line = 0;

            g_accumulated_points.push_back(lp);
        }

        if (frame_offset_ms >= SCAN_PERIOD_MS && !g_accumulated_points.empty()) {
            PointCloud cloud;
            cloud.timestamp_ns = g_scan_start_time;
            cloud.points = std::move(g_accumulated_points);
            g_slam->addPointCloud(cloud);

            g_accumulated_points.clear();
            g_accumulated_points.reserve(20000);
        }
    });

    // Start streaming
    std::cout << "\nStarting LiDAR streaming...\n";
    if (!lidar.startStreaming()) {
        std::cerr << "Failed to start LiDAR!\n";
        vesc.shutdown();
        return 1;
    }

    // Wait for SLAM initialization
    std::cout << "Waiting for SLAM initialization (keep robot still)...\n";
    auto init_start = std::chrono::steady_clock::now();
    while (!g_slam->isInitialized() && g_running) {
        g_slam->process();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - init_start).count();
        if (elapsed > 10.0f) {
            std::cerr << "SLAM initialization timeout!\n";
            lidar.stop();
            vesc.shutdown();
            return 1;
        }
    }
    std::cout << "SLAM initialized!\n";

    // Initialize fusion with current SLAM pose
    auto initial_slam = g_slam->getPose();
    Pose3D initial_pose;
    initial_pose.x = initial_slam(0, 3);
    initial_pose.y = initial_slam(1, 3);
    initial_pose.z = initial_slam(2, 3);
    initial_pose.yaw = std::atan2(initial_slam(1,0), initial_slam(0,0));
    g_fusion->reset(initial_pose);

    // ================================================================
    // Ready to start
    // ================================================================
    std::cout << "\n========================================\n";
    std::cout << "  Press Enter to start motion sequence\n";
    std::cout << "========================================\n";
    std::cin.get();

    motion.resetPose();

    auto last_time = std::chrono::steady_clock::now();
    auto phase_start = last_time;
    int log_counter = 0;

    // Helper lambda for motion update with fusion
    auto updateAll = [&](float dt) {
        motion.update(dt);
        g_slam->process();

        // Get wheel state for fusion
        Pose2D odom = motion.getPose();
        Velocity2D vel = motion.getVelocity();
        auto status_left = vesc.getStatus(1);
        auto status_right = vesc.getStatus(126);

        // Update fusion with wheel odometry
        g_fusion->updateWheelOdometry(
            vel.linear,
            vel.angular,
            status_left.erpm,
            status_right.erpm,
            dt
        );

        // Update fusion with SLAM pose
        auto slam_pose_d = g_slam->getPose();  // Matrix4d (double)
        Eigen::Matrix4f slam_pose = slam_pose_d.cast<float>();  // Convert to float
        uint64_t timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        g_fusion->updateSlamPose(slam_pose, timestamp_us);
    };

    // ================================================================
    // PHASE 1: Forward 1m
    // ================================================================
    std::cout << "\n>>> PHASE 1: Forward " << TARGET_DISTANCE_M << "m <<<\n";

    Pose2D start_pose = motion.getPose();
    float start_x = start_pose.x;
    float start_y = start_pose.y;

    motion.setDuty(DRIVE_DUTY, 0.0f);
    phase_start = std::chrono::steady_clock::now();

    while (g_running) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        float elapsed = std::chrono::duration<float>(now - phase_start).count();
        last_time = now;

        updateAll(dt);

        Pose2D pose = motion.getPose();
        float dx = pose.x - start_x;
        float dy = pose.y - start_y;
        float distance = std::sqrt(dx*dx + dy*dy);

        if (++log_counter % 5 == 0) {
            logPoses("forward", pose);
        }
        if (log_counter % 25 == 0) {
            printComparison(pose);
        }

        if (distance >= TARGET_DISTANCE_M - DISTANCE_TOLERANCE_M) {
            std::cout << "  Target reached!\n";
            break;
        }
        if (elapsed > DRIVE_TIMEOUT_S) {
            std::cout << "  Timeout\n";
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    motion.stop(false);
    for (int i = 0; i < 25 && g_running; i++) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        updateAll(dt);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    logPoses("forward_end", motion.getPose());
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    if (!g_running) goto cleanup;

    // ================================================================
    // PHASE 2: Turn left 90
    // ================================================================
    std::cout << "\n>>> PHASE 2: Turn LEFT 90 <<<\n";

    {
        float start_theta = motion.getPose().theta;
        motion.setDuty(0.0f, TURN_DUTY);
        phase_start = std::chrono::steady_clock::now();

        while (g_running) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_time).count();
            float elapsed = std::chrono::duration<float>(now - phase_start).count();
            last_time = now;

            updateAll(dt);

            Pose2D pose = motion.getPose();
            float delta_theta = normalizeAngle(pose.theta - start_theta);
            float progress_deg = delta_theta * RAD_TO_DEG;

            if (++log_counter % 5 == 0) {
                logPoses("turn_left", pose);
            }

            if (delta_theta >= (TARGET_ANGLE_DEG - ANGLE_TOLERANCE_DEG) * DEG_TO_RAD) {
                std::cout << "  Target reached!\n";
                break;
            }
            if (elapsed > TURN_TIMEOUT_S) {
                std::cout << "  Timeout at " << progress_deg << "\n";
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        motion.stop(false);
        for (int i = 0; i < 25 && g_running; i++) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_time).count();
            last_time = now;
            updateAll(dt);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        logPoses("turn_left_end", motion.getPose());
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    if (!g_running) goto cleanup;

    // ================================================================
    // PHASE 3: Turn right 90
    // ================================================================
    std::cout << "\n>>> PHASE 3: Turn RIGHT 90 <<<\n";

    {
        float start_theta = motion.getPose().theta;
        motion.setDuty(0.0f, -TURN_DUTY);
        phase_start = std::chrono::steady_clock::now();

        while (g_running) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_time).count();
            float elapsed = std::chrono::duration<float>(now - phase_start).count();
            last_time = now;

            updateAll(dt);

            Pose2D pose = motion.getPose();
            float delta_theta = normalizeAngle(start_theta - pose.theta);
            float progress_deg = delta_theta * RAD_TO_DEG;

            if (++log_counter % 5 == 0) {
                logPoses("turn_right", pose);
            }

            if (delta_theta >= (TARGET_ANGLE_DEG - ANGLE_TOLERANCE_DEG) * DEG_TO_RAD) {
                std::cout << "  Target reached!\n";
                break;
            }
            if (elapsed > TURN_TIMEOUT_S) {
                std::cout << "  Timeout at " << progress_deg << "\n";
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        motion.stop(false);
        for (int i = 0; i < 25 && g_running; i++) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_time).count();
            last_time = now;
            updateAll(dt);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        logPoses("turn_right_end", motion.getPose());
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    if (!g_running) goto cleanup;

    // ================================================================
    // PHASE 4: Reverse 1m
    // ================================================================
    std::cout << "\n>>> PHASE 4: Reverse " << TARGET_DISTANCE_M << "m <<<\n";

    {
        Pose2D reverse_start = motion.getPose();
        start_x = reverse_start.x;
        start_y = reverse_start.y;

        motion.setDuty(-DRIVE_DUTY, 0.0f);
        phase_start = std::chrono::steady_clock::now();

        while (g_running) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_time).count();
            float elapsed = std::chrono::duration<float>(now - phase_start).count();
            last_time = now;

            updateAll(dt);

            Pose2D pose = motion.getPose();
            float dx = pose.x - start_x;
            float dy = pose.y - start_y;
            float distance = std::sqrt(dx*dx + dy*dy);

            if (++log_counter % 5 == 0) {
                logPoses("reverse", pose);
            }
            if (log_counter % 25 == 0) {
                printComparison(pose);
            }

            if (distance >= TARGET_DISTANCE_M - DISTANCE_TOLERANCE_M) {
                std::cout << "  Target reached!\n";
                break;
            }
            if (elapsed > DRIVE_TIMEOUT_S) {
                std::cout << "  Timeout\n";
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        motion.stop(false);
        for (int i = 0; i < 25 && g_running; i++) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_time).count();
            last_time = now;
            updateAll(dt);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    logPoses("final", motion.getPose());

cleanup:
    // Print statistics
    printStatistics();

    // Save log
    saveLog(output_prefix + "_log.csv");

    // Cleanup
    std::cout << "\nStopping LiDAR...\n";
    lidar.stop();

    std::cout << "Stopping motors...\n";
    vesc.shutdown();

    std::cout << "\nTest complete!\n";
    return 0;
}
