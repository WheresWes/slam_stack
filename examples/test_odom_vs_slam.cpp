/**
 * Wheel Odometry vs SLAM Comparison Test
 *
 * Runs the same motion sequence while tracking with both:
 * 1. Wheel odometry (MotionController using VESC encoders)
 * 2. LiDAR SLAM (FAST-LIO2 with Livox Mid-360)
 *
 * Logs both poses throughout the sequence and computes error statistics.
 *
 * Motion sequence: Forward 1m → Turn left 90° → Turn right 90° → Reverse 1m
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
#include "slam/livox_mid360.hpp"
#include "slam/slam_engine.hpp"
#include "slam/types.hpp"

#ifdef HAS_RERUN
#include "slam/visualization_interface.hpp"
#endif

using namespace slam;

// Constants - use slam::PI from types.hpp
constexpr float DEG_TO_RAD = static_cast<float>(PI) / 180.0f;
constexpr float RAD_TO_DEG = 180.0f / static_cast<float>(PI);

// Motion parameters
constexpr float DRIVE_DUTY = 0.8f;      // 80% of max
constexpr float TURN_DUTY = 0.6f;       // 60% for turning
constexpr float MAX_DUTY = 0.12f;       // 12% actual duty
constexpr float RAMP_RATE = 0.4f;       // Duty/sec
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
std::mutex g_data_mutex;

// SLAM data buffers
std::mutex g_scan_mutex;
std::vector<LidarPoint> g_accumulated_points;
uint64_t g_scan_start_time = 0;
const double SCAN_PERIOD_MS = 100.0;

// Pose log entry
struct PoseLogEntry {
    double timestamp_s;
    std::string phase;
    // Wheel odometry (2D)
    float odom_x, odom_y, odom_theta;
    // SLAM (3D)
    float slam_x, slam_y, slam_z;
    float slam_roll, slam_pitch, slam_yaw;
    // Error (2D projection)
    float error_x, error_y, error_dist;
};

std::vector<PoseLogEntry> g_pose_log;
std::mutex g_log_mutex;

void signalHandler(int) {
    g_running = false;
}

float normalizeAngle(float angle) {
    while (angle > static_cast<float>(PI)) angle -= 2.0f * static_cast<float>(PI);
    while (angle < -static_cast<float>(PI)) angle += 2.0f * static_cast<float>(PI);
    return angle;
}

// Extract yaw from SLAM pose (rotation matrix to euler)
float getYawFromSLAM() {
    if (!g_slam) return 0;
    auto pose = g_slam->getPose();
    // Rotation matrix to yaw (rotation around Z axis)
    // yaw = atan2(R(1,0), R(0,0))
    return std::atan2(pose(1,0), pose(0,0));
}

void logPose(const std::string& phase, const Pose2D& odom) {
    if (!g_slam) return;

    auto slam_pose = g_slam->getPose();
    float slam_x = slam_pose(0, 3);
    float slam_y = slam_pose(1, 3);
    float slam_z = slam_pose(2, 3);

    // Extract euler angles from rotation matrix
    float slam_roll = std::atan2(slam_pose(2,1), slam_pose(2,2));
    float slam_pitch = std::asin(-slam_pose(2,0));
    float slam_yaw = std::atan2(slam_pose(1,0), slam_pose(0,0));

    // Compute 2D error
    float error_x = slam_x - odom.x;
    float error_y = slam_y - odom.y;
    float error_dist = std::sqrt(error_x*error_x + error_y*error_y);

    static auto start_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    double timestamp = std::chrono::duration<double>(now - start_time).count();

    PoseLogEntry entry;
    entry.timestamp_s = timestamp;
    entry.phase = phase;
    entry.odom_x = odom.x;
    entry.odom_y = odom.y;
    entry.odom_theta = odom.theta;
    entry.slam_x = slam_x;
    entry.slam_y = slam_y;
    entry.slam_z = slam_z;
    entry.slam_roll = slam_roll;
    entry.slam_pitch = slam_pitch;
    entry.slam_yaw = slam_yaw;
    entry.error_x = error_x;
    entry.error_y = error_y;
    entry.error_dist = error_dist;

    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_pose_log.push_back(entry);
}

void printComparison(const Pose2D& odom) {
    if (!g_slam) return;

    auto slam_pose = g_slam->getPose();
    float slam_x = slam_pose(0, 3);
    float slam_y = slam_pose(1, 3);
    float slam_yaw = std::atan2(slam_pose(1,0), slam_pose(0,0));

    float error_dist = std::sqrt(std::pow(slam_x - odom.x, 2) + std::pow(slam_y - odom.y, 2));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  ODOM: x=" << odom.x << " y=" << odom.y << " θ=" << (odom.theta * RAD_TO_DEG) << "°"
              << " | SLAM: x=" << slam_x << " y=" << slam_y << " θ=" << (slam_yaw * RAD_TO_DEG) << "°"
              << " | Δ=" << (error_dist * 100) << "cm" << std::endl;
}

void savePoseLog(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to save pose log to " << filename << std::endl;
        return;
    }

    // Header
    file << "timestamp_s,phase,odom_x,odom_y,odom_theta,slam_x,slam_y,slam_z,slam_roll,slam_pitch,slam_yaw,error_x,error_y,error_dist\n";

    std::lock_guard<std::mutex> lock(g_log_mutex);
    for (const auto& e : g_pose_log) {
        file << std::fixed << std::setprecision(6)
             << e.timestamp_s << "," << e.phase << ","
             << e.odom_x << "," << e.odom_y << "," << e.odom_theta << ","
             << e.slam_x << "," << e.slam_y << "," << e.slam_z << ","
             << e.slam_roll << "," << e.slam_pitch << "," << e.slam_yaw << ","
             << e.error_x << "," << e.error_y << "," << e.error_dist << "\n";
    }

    std::cout << "Pose log saved to " << filename << " (" << g_pose_log.size() << " entries)\n";
}

void printErrorStatistics() {
    std::lock_guard<std::mutex> lock(g_log_mutex);

    if (g_pose_log.empty()) {
        std::cout << "No pose data logged.\n";
        return;
    }

    // Compute statistics
    float sum_error = 0, max_error = 0;
    float sum_error_sq = 0;

    for (const auto& e : g_pose_log) {
        sum_error += e.error_dist;
        sum_error_sq += e.error_dist * e.error_dist;
        if (e.error_dist > max_error) max_error = e.error_dist;
    }

    float mean_error = sum_error / g_pose_log.size();
    float rms_error = std::sqrt(sum_error_sq / g_pose_log.size());

    // Final pose comparison
    const auto& final_entry = g_pose_log.back();

    std::cout << "\n========================================\n";
    std::cout << "  ERROR STATISTICS (Wheel Odom vs SLAM)\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Samples:     " << g_pose_log.size() << "\n";
    std::cout << "  Mean error:  " << (mean_error * 100) << " cm\n";
    std::cout << "  RMS error:   " << (rms_error * 100) << " cm\n";
    std::cout << "  Max error:   " << (max_error * 100) << " cm\n";
    std::cout << "\n  Final poses:\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "    Wheel:  x=" << final_entry.odom_x << " y=" << final_entry.odom_y
              << " θ=" << (final_entry.odom_theta * RAD_TO_DEG) << "°\n";
    std::cout << "    SLAM:   x=" << final_entry.slam_x << " y=" << final_entry.slam_y
              << " θ=" << (final_entry.slam_yaw * RAD_TO_DEG) << "°\n";
    std::cout << "    Error:  " << (final_entry.error_dist * 100) << " cm\n";
    std::cout << "========================================\n";
}

int main(int argc, char* argv[]) {
    std::cout << "================================================\n";
    std::cout << "  Wheel Odometry vs SLAM Comparison Test\n";
    std::cout << "================================================\n";
    std::cout << "Sequence: Forward 1m → Left 90° → Right 90° → Reverse 1m\n\n";

    // Configuration
    std::string can_port = "COM3";
    std::string cal_file = "vesc_calibration.ini";
    std::string device_ip = "";
    std::string host_ip = "192.168.1.50";
    std::string output_prefix = "odom_vs_slam";
    bool enable_visualization = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--can" && i + 1 < argc) can_port = argv[++i];
        else if (arg == "--cal" && i + 1 < argc) cal_file = argv[++i];
        else if (arg == "--device" && i + 1 < argc) device_ip = argv[++i];
        else if (arg == "--host" && i + 1 < argc) host_ip = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_prefix = argv[++i];
        else if (arg == "--visualize") enable_visualization = true;
        else if (arg == "--help") {
            std::cout << "Usage: test_odom_vs_slam [options]\n"
                      << "  --can PORT      CAN port (default: COM3)\n"
                      << "  --cal FILE      Calibration file (default: vesc_calibration.ini)\n"
                      << "  --device IP     LiDAR IP (default: auto-discover)\n"
                      << "  --host IP       Host IP (default: 192.168.1.50)\n"
                      << "  --output NAME   Output prefix (default: odom_vs_slam)\n"
                      << "  --visualize     Enable Rerun visualization\n";
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
    std::cout << "  VESCs connected!\n";

    std::cout << "Loading calibration from " << cal_file << "...\n";
    MotionController motion;
    if (!motion.init(&vesc, cal_file)) {
        std::cerr << "  Warning: Using default calibration\n";
    }
    motion.setMaxDuty(MAX_DUTY);
    motion.setRampRate(RAMP_RATE);

    const auto& cal = motion.getCalibration();
    std::cout << "  Effective track: " << (cal.effective_track_m * 1000) << " mm\n";
    std::cout << "  Ticks/meter: " << cal.ticks_per_meter << "\n";

    // ================================================================
    // Initialize SLAM Engine
    // ================================================================
    std::cout << "\nInitializing SLAM engine...\n";
    g_slam = std::make_unique<SlamEngine>();

    SlamConfig slam_config;
    slam_config.filter_size_surf = 0.1;  // 10cm voxel
    slam_config.filter_size_map = 0.1;
    slam_config.deskew_enabled = true;
    slam_config.save_map = true;
    slam_config.map_save_path = output_prefix + "_map.ply";
    slam_config.gyr_cov = 0.1;
    slam_config.acc_cov = 0.1;
    slam_config.max_iterations = 4;

    if (!g_slam->init(slam_config)) {
        std::cerr << "Failed to initialize SLAM engine!\n";
        vesc.shutdown();
        return 1;
    }
    std::cout << "  SLAM engine initialized\n";

    // ================================================================
    // Initialize Visualization (optional)
    // ================================================================
#ifdef HAS_RERUN
    std::unique_ptr<Visualizer> visualizer;
    if (enable_visualization) {
        VisualizerConfig viz_config;
        viz_config.application_id = "odom_vs_slam";
        viz_config.spawn_viewer = true;
        visualizer = std::make_unique<Visualizer>(viz_config);
        if (visualizer->isInitialized()) {
            std::cout << "  Visualization: enabled\n";
        }
    }
#endif

    // ================================================================
    // Initialize LiDAR
    // ================================================================
    std::cout << "\nInitializing LiDAR...\n";
    LivoxMid360 lidar;

    if (device_ip.empty()) {
        std::cout << "  Scanning for devices...\n";
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
    std::cout << "  LiDAR connected!\n";

    // Preprocessor
    PreprocessConfig pre_config;
    pre_config.lidar_type = LidarType::LIVOX_MID360;
    pre_config.n_scans = 4;
    pre_config.blind_distance = 0.5;
    pre_config.max_distance = 50.0;
    pre_config.point_filter_num = 3;
    Preprocessor preprocessor(pre_config);

    g_accumulated_points.reserve(20000);

    // ================================================================
    // Set up LiDAR callbacks
    // ================================================================
    std::atomic<uint64_t> point_count{0};
    std::atomic<uint64_t> imu_count{0};

    // IMU callback
    lidar.setIMUCallback([&](const LivoxIMUFrame& frame) {
        constexpr double G_M_S2 = 9.81;

        ImuData imu;
        imu.timestamp_ns = frame.timestamp_ns;
        // Livox Mid-360 accelerometer outputs in g units, convert to m/s²
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

        // Convert and filter points
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
            if (range < pre_config.blind_distance || range > pre_config.max_distance) continue;

            // Point filtering
            valid_counter++;
            if ((valid_counter % pre_config.point_filter_num) != 0) continue;

            LidarPoint lp;
            lp.x = static_cast<float>(p.x());
            lp.y = static_cast<float>(p.y());
            lp.z = static_cast<float>(p.z());
            lp.intensity = frame.reflectivities.empty() ? 100.0f : static_cast<float>(frame.reflectivities[i]);
            // Per-point timestamp offset in ms
            float point_offset_ms = (i < frame.time_offsets_us.size()) ?
                (frame.time_offsets_us[i] / 1000.0f) : 0.0f;
            lp.time_offset_ms = static_cast<float>(frame_offset_ms) + point_offset_ms;
            lp.tag = frame.tags.empty() ? 0 : frame.tags[i];
            lp.line = 0;

            g_accumulated_points.push_back(lp);
        }
        point_count += frame.points.size();

        // Check if scan complete
        if (frame_offset_ms >= SCAN_PERIOD_MS && !g_accumulated_points.empty()) {
            // Create point cloud from accumulated points
            PointCloud cloud;
            cloud.timestamp_ns = g_scan_start_time;
            cloud.points = std::move(g_accumulated_points);

            // Add to SLAM (will be buffered internally)
            g_slam->addPointCloud(cloud);

            // Reset accumulator
            g_accumulated_points.clear();
            g_accumulated_points.reserve(20000);
        }
    });

    // Start LiDAR streaming
    std::cout << "\nStarting LiDAR streaming...\n";
    if (!lidar.startStreaming()) {
        std::cerr << "Failed to start LiDAR!\n";
        vesc.shutdown();
        return 1;
    }

    // Wait for SLAM initialization (gravity alignment)
    std::cout << "Waiting for SLAM initialization (keep robot still)...\n";
    auto init_start = std::chrono::steady_clock::now();
    while (!g_slam->isInitialized() && g_running) {
        g_slam->process();  // Process buffered data
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - init_start).count();
        if (elapsed > 10.0f) {
            std::cerr << "SLAM initialization timeout!\n";
            lidar.stop();
            vesc.shutdown();
            return 1;
        }
        std::cout << "  IMU: " << imu_count << " | Points: " << point_count << "\r" << std::flush;
    }
    std::cout << "\nSLAM initialized!                          \n";

    // ================================================================
    // Ready to start
    // ================================================================
    std::cout << "\n========================================\n";
    std::cout << "  Press Enter to start motion sequence\n";
    std::cout << "  (Robot will move!)\n";
    std::cout << "========================================\n";
    std::cin.get();

    // Reset both tracking systems
    motion.resetPose();
    // Note: SLAM doesn't have a reset - its initial pose becomes origin

    auto last_time = std::chrono::steady_clock::now();
    auto phase_start = last_time;
    int log_counter = 0;

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

        motion.update(dt);

        Pose2D pose = motion.getPose();
        float dx = pose.x - start_x;
        float dy = pose.y - start_y;
        float distance = std::sqrt(dx*dx + dy*dy);

        // Log every 5th iteration
        if (++log_counter % 5 == 0) {
            logPose("forward", pose);
        }

        // Print every 25th iteration
        if (log_counter % 25 == 0) {
            printComparison(pose);
        }

        if (distance >= TARGET_DISTANCE_M - DISTANCE_TOLERANCE_M) {
            std::cout << "  Target reached!\n";
            break;
        }

        if (elapsed > DRIVE_TIMEOUT_S) {
            std::cout << "  Timeout at " << distance << "m\n";
            break;
        }

        // Process SLAM data
        g_slam->process();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // Stop and settle
    motion.stop(false);
    for (int i = 0; i < 25 && g_running; i++) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        motion.update(dt);
        g_slam->process();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    Pose2D after_forward = motion.getPose();
    std::cout << "  After forward:\n";
    printComparison(after_forward);
    logPose("forward_end", after_forward);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    if (!g_running) goto cleanup;

    // ================================================================
    // PHASE 2: Turn left 90°
    // ================================================================
    std::cout << "\n>>> PHASE 2: Turn LEFT " << TARGET_ANGLE_DEG << "° <<<\n";

    {
        float start_theta = motion.getPose().theta;

        motion.setDuty(0.0f, TURN_DUTY);
        phase_start = std::chrono::steady_clock::now();

        while (g_running) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_time).count();
            float elapsed = std::chrono::duration<float>(now - phase_start).count();
            last_time = now;

            motion.update(dt);

            Pose2D pose = motion.getPose();
            float delta_theta = normalizeAngle(pose.theta - start_theta);
            float progress_deg = delta_theta * RAD_TO_DEG;

            if (++log_counter % 5 == 0) {
                logPose("turn_left", pose);
            }

            if (log_counter % 25 == 0) {
                std::cout << "  angle=" << progress_deg << "°\n";
            }

            if (delta_theta >= (TARGET_ANGLE_DEG - ANGLE_TOLERANCE_DEG) * DEG_TO_RAD) {
                std::cout << "  Target reached!\n";
                break;
            }

            if (elapsed > TURN_TIMEOUT_S) {
                std::cout << "  Timeout at " << progress_deg << "°\n";
                break;
            }

            g_slam->process();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        motion.stop(false);
        for (int i = 0; i < 25 && g_running; i++) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_time).count();
            last_time = now;
            motion.update(dt);
            g_slam->process();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        Pose2D after_left = motion.getPose();
        std::cout << "  After left turn:\n";
        printComparison(after_left);
        logPose("turn_left_end", after_left);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    if (!g_running) goto cleanup;

    // ================================================================
    // PHASE 3: Turn right 90°
    // ================================================================
    std::cout << "\n>>> PHASE 3: Turn RIGHT " << TARGET_ANGLE_DEG << "° <<<\n";

    {
        float start_theta = motion.getPose().theta;

        motion.setDuty(0.0f, -TURN_DUTY);
        phase_start = std::chrono::steady_clock::now();

        while (g_running) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_time).count();
            float elapsed = std::chrono::duration<float>(now - phase_start).count();
            last_time = now;

            motion.update(dt);

            Pose2D pose = motion.getPose();
            float delta_theta = normalizeAngle(start_theta - pose.theta);
            float progress_deg = delta_theta * RAD_TO_DEG;

            if (++log_counter % 5 == 0) {
                logPose("turn_right", pose);
            }

            if (log_counter % 25 == 0) {
                std::cout << "  angle=" << progress_deg << "°\n";
            }

            if (delta_theta >= (TARGET_ANGLE_DEG - ANGLE_TOLERANCE_DEG) * DEG_TO_RAD) {
                std::cout << "  Target reached!\n";
                break;
            }

            if (elapsed > TURN_TIMEOUT_S) {
                std::cout << "  Timeout at " << progress_deg << "°\n";
                break;
            }

            g_slam->process();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        motion.stop(false);
        for (int i = 0; i < 25 && g_running; i++) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_time).count();
            last_time = now;
            motion.update(dt);
            g_slam->process();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        Pose2D after_right = motion.getPose();
        std::cout << "  After right turn:\n";
        printComparison(after_right);
        logPose("turn_right_end", after_right);
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

            motion.update(dt);

            Pose2D pose = motion.getPose();
            float dx = pose.x - start_x;
            float dy = pose.y - start_y;
            float distance = std::sqrt(dx*dx + dy*dy);

            if (++log_counter % 5 == 0) {
                logPose("reverse", pose);
            }

            if (log_counter % 25 == 0) {
                printComparison(pose);
            }

            if (distance >= TARGET_DISTANCE_M - DISTANCE_TOLERANCE_M) {
                std::cout << "  Target reached!\n";
                break;
            }

            if (elapsed > DRIVE_TIMEOUT_S) {
                std::cout << "  Timeout at " << distance << "m\n";
                break;
            }

            g_slam->process();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        motion.stop(false);
        for (int i = 0; i < 25 && g_running; i++) {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - last_time).count();
            last_time = now;
            motion.update(dt);
            g_slam->process();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    // ================================================================
    // Results
    // ================================================================
    {
        Pose2D final_pose = motion.getPose();
        std::cout << "\n  Final position:\n";
        printComparison(final_pose);
        logPose("final", final_pose);

        float return_error = std::sqrt(final_pose.x*final_pose.x + final_pose.y*final_pose.y);
        std::cout << "\n  Wheel odom return-to-start error: " << (return_error * 100) << " cm\n";

        auto slam_pose = g_slam->getPose();
        float slam_return = std::sqrt(slam_pose(0,3)*slam_pose(0,3) + slam_pose(1,3)*slam_pose(1,3));
        std::cout << "  SLAM return-to-start error: " << (slam_return * 100) << " cm\n";
    }

cleanup:
    // Print statistics
    printErrorStatistics();

    // Save logs
    savePoseLog(output_prefix + "_poses.csv");

    // Save SLAM map
    std::cout << "\nSaving SLAM map...\n";
    g_slam->saveMap(output_prefix + "_map.ply");

    // Cleanup
    std::cout << "Stopping LiDAR...\n";
    lidar.stop();

    std::cout << "Stopping motors...\n";
    vesc.shutdown();

    std::cout << "\nTest complete!\n";
    return 0;
}
