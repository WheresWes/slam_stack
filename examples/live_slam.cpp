/**
 * @file live_slam.cpp
 * @brief Live SLAM with native Livox Mid-360 driver
 *
 * Uses our native UDP driver (no Livox SDK dependency) for real-time SLAM.
 * Supports the Mid-360 with both point cloud and IMU data.
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
#include <atomic>
#include <csignal>
#include <mutex>
#include <deque>

#include "slam/livox_mid360.hpp"
#include "slam/slam_engine.hpp"
#include "slam/types.hpp"

using namespace slam;

// Global state
std::atomic<bool> g_running{true};
std::unique_ptr<SlamEngine> g_slam;
std::mutex g_data_mutex;

// Statistics
std::atomic<uint64_t> g_point_count{0};
std::atomic<uint64_t> g_imu_count{0};
std::atomic<uint64_t> g_scan_count{0};

void signalHandler(int) {
    g_running = false;
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --serial <num>   Last 2-3 digits of serial number (e.g., 544)\n";
    std::cout << "  --device <ip>    Device IP directly (default: auto-discover)\n";
    std::cout << "  --host <ip>      Host IP address (default: 192.168.1.50)\n";
    std::cout << "  --time <sec>     Run duration in seconds (default: 30, 0=unlimited)\n";
    std::cout << "  --output <name>  Output file prefix (default: live_slam)\n";
    std::cout << "  --voxel <size>   Map voxel size in meters (default: 0.1)\n";
    std::cout << "  --help           Show this help\n";
}

int main(int argc, char** argv) {
    std::cout << "============================================\n";
    std::cout << "  Live SLAM - Native Livox Mid-360 Driver\n";
    std::cout << "============================================\n\n";

    // Parse arguments
    std::string host_ip = "192.168.1.50";
    std::string device_ip = "";
    std::string serial_suffix = "";
    std::string output_prefix = "live_slam";
    int run_seconds = 30;
    double voxel_size = 0.1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) {
            host_ip = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device_ip = argv[++i];
        } else if (arg == "--serial" && i + 1 < argc) {
            serial_suffix = argv[++i];
            if (serial_suffix.length() >= 2) {
                std::string last2 = serial_suffix.substr(serial_suffix.length() - 2);
                device_ip = "192.168.1.1" + last2;
            }
        } else if (arg == "--time" && i + 1 < argc) {
            run_seconds = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_prefix = argv[++i];
        } else if (arg == "--voxel" && i + 1 < argc) {
            voxel_size = std::atof(argv[++i]);
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Register signal handler
    std::signal(SIGINT, signalHandler);

    // Initialize SLAM engine
    std::cout << "Initializing SLAM engine...\n";
    g_slam = std::make_unique<SlamEngine>();

    SlamConfig config;
    config.filter_size_surf = voxel_size;
    config.filter_size_map = voxel_size;
    config.deskew_enabled = true;
    config.save_map = true;
    config.map_save_path = output_prefix + "_map.ply";

    // Mid-360 IMU noise parameters (from datasheet)
    config.gyr_cov = 0.01;
    config.acc_cov = 0.1;
    config.b_gyr_cov = 0.0001;
    config.b_acc_cov = 0.0001;

    if (!g_slam->init(config)) {
        std::cerr << "Failed to initialize SLAM engine!\n";
        return 1;
    }

    std::cout << "  Voxel size: " << voxel_size << "m\n";
    std::cout << "  Deskew: enabled\n\n";

    // Initialize Livox driver
    LivoxMid360 lidar;
    std::string target_ip = device_ip;

    // Discover or use specified device
    if (device_ip.empty()) {
        std::cout << "Scanning for Livox devices...\n";
        auto devices = lidar.discover(3000, host_ip);

        if (devices.empty()) {
            std::cout << "\n[!] No devices found via broadcast.\n";
            std::cout << "    Try specifying the device IP directly:\n";
            std::cout << "    --device 192.168.1.1XX  (XX = last 2 digits of serial)\n";
            std::cout << "    --serial 544  (for serial ending in 544)\n\n";
            return 1;
        }
        target_ip = devices[0].ip_address;
        std::cout << "Found: " << devices[0].getTypeName() << " at " << target_ip << "\n";
    } else {
        std::cout << "Using device IP: " << device_ip << "\n";
    }

    // Connect to device
    std::cout << "Connecting to " << target_ip << "...\n";
    if (!lidar.connect(target_ip, host_ip)) {
        std::cerr << "Failed to connect to LiDAR!\n";
        return 1;
    }
    std::cout << "Connected!\n\n";

    // Preprocessor for point cloud
    PreprocessConfig pre_config;
    pre_config.lidar_type = LidarType::LIVOX_MID360;
    pre_config.n_scans = 4;
    pre_config.blind_distance = 0.5;
    pre_config.max_distance = 50.0;
    pre_config.point_filter_num = 1;
    Preprocessor preprocessor(pre_config);

    // Accumulator for building complete scans (~100ms at 10Hz)
    std::mutex scan_mutex;
    std::vector<LidarPoint> accumulated_points;
    uint64_t scan_start_time = 0;
    const double SCAN_PERIOD_MS = 100.0;  // 10 Hz scan rate

    accumulated_points.reserve(20000);

    // Point cloud callback - accumulate into scans
    lidar.setPointCloudCallback([&](const LivoxPointCloudFrame& frame) {
        std::lock_guard<std::mutex> lock(scan_mutex);

        // Start new scan if empty
        if (accumulated_points.empty()) {
            scan_start_time = frame.timestamp_ns;
        }

        // Calculate time offset from scan start
        double time_offset_ms = (frame.timestamp_ns - scan_start_time) / 1e6;

        // Convert frame points to LidarPoint format
        for (size_t i = 0; i < frame.points.size(); i++) {
            const V3D& p = frame.points[i];

            // Skip invalid points (too close or too far)
            double dist = p.norm();
            if (dist < 0.3 || dist > 50.0) continue;

            // Skip zero points
            if (std::abs(p.x()) < 0.001 && std::abs(p.y()) < 0.001 && std::abs(p.z()) < 0.001) continue;

            LidarPoint lp;
            lp.x = static_cast<float>(p.x());
            lp.y = static_cast<float>(p.y());
            lp.z = static_cast<float>(p.z());
            lp.intensity = static_cast<float>(frame.reflectivities[i]);
            lp.time_offset_ms = static_cast<float>(time_offset_ms);
            lp.tag = frame.tags.empty() ? 0 : frame.tags[i];
            lp.line = 0;

            accumulated_points.push_back(lp);
        }

        g_point_count += frame.points.size();

        // Check if we have accumulated enough for a scan (~100ms)
        if (time_offset_ms >= SCAN_PERIOD_MS && accumulated_points.size() > 100) {
            // Create point cloud from accumulated points
            PointCloud cloud;
            cloud.timestamp_ns = scan_start_time;
            cloud.points = std::move(accumulated_points);

            // Reset accumulator
            accumulated_points.clear();
            accumulated_points.reserve(20000);

            // Add to SLAM (will be buffered internally)
            g_slam->addPointCloud(cloud);
            g_scan_count++;
        }
    });

    // IMU callback
    lidar.setIMUCallback([&](const LivoxIMUFrame& frame) {
        ImuData imu;
        imu.timestamp_ns = frame.timestamp_ns;
        imu.acc = frame.accel;
        imu.gyro = frame.gyro;

        g_slam->addImuData(imu);
        g_imu_count++;
    });

    // Start streaming
    std::cout << "Starting SLAM...\n";
    if (run_seconds > 0) {
        std::cout << "Will run for " << run_seconds << " seconds (Ctrl+C to stop early)\n\n";
    } else {
        std::cout << "Running until Ctrl+C...\n\n";
    }

    if (!lidar.startStreaming()) {
        std::cerr << "Failed to start streaming!\n";
        return 1;
    }

    // Main loop - process SLAM and display status
    auto start_time = std::chrono::steady_clock::now();
    auto last_status_time = start_time;
    uint64_t last_points = 0;
    uint64_t last_imu = 0;

    while (g_running) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        if (run_seconds > 0 && elapsed >= run_seconds) break;

        // Process any available buffered sensor data
        // This syncs IMU + LiDAR and runs the SLAM optimization
        g_slam->process();

        // Sleep briefly to not spin too fast
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Display status every second
        auto status_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_status_time).count();
        if (status_elapsed >= 1000) {
            // Get current stats
            uint64_t cur_points = g_point_count.load();
            uint64_t cur_imu = g_imu_count.load();
            uint64_t cur_scans = g_scan_count.load();

            // Calculate rates
            double pts_rate = (cur_points - last_points);
            double imu_rate = (cur_imu - last_imu);
            last_points = cur_points;
            last_imu = cur_imu;

            // Get SLAM state
            SlamState state = g_slam->getState();
            size_t map_size = g_slam->getMapSize();

            // Display status
            std::cout << "\r  Time: " << elapsed << "s | "
                      << "Pts: " << (pts_rate / 1000) << "k/s | "
                      << "IMU: " << imu_rate << "Hz | "
                      << "Scans: " << cur_scans << " | "
                      << "Map: " << (map_size / 1000) << "k | "
                      << "Pos: [" << std::fixed << std::setprecision(2)
                      << state.pos.x() << ", " << state.pos.y() << ", " << state.pos.z() << "]    "
                      << std::flush;

            last_status_time = now;
        }
    }

    std::cout << "\n\nStopping...\n";
    lidar.stop();

    // Save results
    std::cout << "\n============================================\n";
    std::cout << "  Results\n";
    std::cout << "============================================\n";

    std::string map_file = output_prefix + "_map.ply";
    std::string traj_file = output_prefix + "_trajectory.ply";

    if (g_slam->saveMap(map_file)) {
        std::cout << "  Map saved: " << map_file << " (" << g_slam->getMapSize() << " points)\n";
    }

    if (g_slam->saveTrajectory(traj_file)) {
        std::cout << "  Trajectory saved: " << traj_file << " (" << g_slam->getTrajectory().size() << " poses)\n";
    }

    std::cout << "\n  Total points: " << g_point_count.load() << "\n";
    std::cout << "  Total IMU: " << g_imu_count.load() << "\n";
    std::cout << "  Total scans: " << g_scan_count.load() << "\n";
    std::cout << "============================================\n\n";

    return 0;
}
