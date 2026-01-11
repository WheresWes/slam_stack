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
#include "slam/visualization_interface.hpp"

using namespace slam;

// Debug timing statistics
struct DebugTiming {
    std::atomic<double> callback_time_us{0};
    std::atomic<double> callback_max_us{0};
    std::atomic<int> callback_count{0};
    std::atomic<int> points_queued{0};
    std::atomic<int> imu_queued{0};
    std::atomic<double> process_time_us{0};
    std::atomic<double> process_max_us{0};
    std::atomic<int> process_count{0};
    std::atomic<uint64_t> last_callback_time{0};
    std::atomic<double> callback_interval_us{0};
};
DebugTiming g_timing;

// Global state
std::atomic<bool> g_running{true};
std::unique_ptr<SlamEngine> g_slam;
std::mutex g_data_mutex;

// Statistics
std::atomic<uint64_t> g_point_count{0};
std::atomic<uint64_t> g_imu_count{0};
std::atomic<uint64_t> g_scan_count{0};

// Raw point cloud buffer for visualization
std::mutex g_raw_points_mutex;
std::vector<V3D> g_raw_points_buffer;
std::atomic<bool> g_raw_points_ready{false};

// Recording file
std::ofstream* g_record_file = nullptr;
std::mutex g_record_mutex;

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
    std::cout << "  --voxel <size>   Map voxel size in meters (default: 0.05)\n";
    std::cout << "  --iterations <n> IEKF iterations per scan (default: 5, range: 1-10)\n";
    std::cout << "  --gyr-cov <val>  Gyroscope noise covariance (default: 0.1)\n";
    std::cout << "  --blind <dist>   Minimum point range in meters (default: 0.5)\n";
    std::cout << "  --filter <n>     Keep every Nth point (default: 3, 1=keep all)\n";
    std::cout << "  --visualize      Enable real-time visualization (requires Rerun)\n";
    std::cout << "  --record [file]  Record raw data for offline replay (default: recorded_data.bin)\n";
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
    double voxel_size = 0.05;  // 5cm for high-detail indoor mapping
    int max_iterations = 5;    // IEKF iterations per scan (5 recommended for motion)
    double gyr_cov = 0.1;      // Gyroscope noise covariance
    double blind_dist = 0.5;   // Minimum point range
    int point_filter = 3;      // Keep every Nth point
    bool enable_visualization = false;
    bool record_data = false;
    std::string record_file = "recorded_data.bin";

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
        } else if (arg == "--iterations" && i + 1 < argc) {
            max_iterations = std::atoi(argv[++i]);
            if (max_iterations < 1) max_iterations = 1;
            if (max_iterations > 10) max_iterations = 10;
        } else if (arg == "--gyr-cov" && i + 1 < argc) {
            gyr_cov = std::atof(argv[++i]);
        } else if (arg == "--blind" && i + 1 < argc) {
            blind_dist = std::atof(argv[++i]);
        } else if (arg == "--filter" && i + 1 < argc) {
            point_filter = std::atoi(argv[++i]);
            if (point_filter < 1) point_filter = 1;
        } else if (arg == "--visualize") {
            enable_visualization = true;
        } else if (arg == "--record") {
            record_data = true;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                record_file = argv[++i];
            }
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
    config.deskew_enabled = true;   // CRITICAL: Enable motion compensation for proper tracking
    config.save_map = true;
    config.map_save_path = output_prefix + "_map.ply";

    // Mid-360 IMU noise parameters (matching original FAST-LIO mid360.yaml)
    // IMPORTANT: gyr_cov=0.1 is critical - lower values trust noisy gyro too much
    config.gyr_cov = gyr_cov;
    config.acc_cov = gyr_cov;  // Use same value for acc_cov
    config.b_gyr_cov = 0.0001;
    config.b_acc_cov = 0.0001;
    config.max_iterations = max_iterations;

    if (!g_slam->init(config)) {
        std::cerr << "Failed to initialize SLAM engine!\n";
        return 1;
    }

    std::cout << "  Voxel size: " << voxel_size << "m\n";
    std::cout << "  Iterations: " << max_iterations << "\n";
    std::cout << "  Gyr cov: " << gyr_cov << "\n";
    std::cout << "  Blind dist: " << blind_dist << "m\n";
    std::cout << "  Point filter: " << point_filter << " (keep 1/" << point_filter << ")\n";
    std::cout << "  Deskew: " << (config.deskew_enabled ? "enabled" : "disabled") << "\n";

    // Initialize visualization (if enabled)
    std::unique_ptr<Visualizer> visualizer;
    if (enable_visualization) {
        VisualizerConfig viz_config;
        viz_config.application_id = "live_slam";
        viz_config.spawn_viewer = true;  // Always spawn a new viewer
        viz_config.connect_addr = "";     // Don't try connecting to existing - just spawn fresh
        viz_config.color_by_height = true;
        viz_config.point_size = 8.0f;    // Doubled for visibility
        viz_config.map_point_size = 6.0f;
        viz_config.max_points_per_frame = 20000;  // Reduced for performance

        visualizer = std::make_unique<Visualizer>(viz_config);
        if (visualizer->isInitialized()) {
            std::cout << "  Visualization: enabled (Rerun)\n";
        } else {
            std::cout << "  Visualization: failed to initialize\n";
            visualizer.reset();
        }
    }

    // Initialize recording file
    if (record_data) {
        g_record_file = new std::ofstream(record_file, std::ios::binary);
        if (g_record_file->is_open()) {
            std::cout << "  Recording: enabled -> " << record_file << "\n";
            // Write header magic
            uint32_t magic = 0x534C414D;  // "SLAM"
            uint32_t version = 1;
            g_record_file->write(reinterpret_cast<char*>(&magic), sizeof(magic));
            g_record_file->write(reinterpret_cast<char*>(&version), sizeof(version));
        } else {
            std::cout << "  Recording: failed to open file\n";
            delete g_record_file;
            g_record_file = nullptr;
        }
    }
    std::cout << "\n";

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
    pre_config.blind_distance = 0.1;  // Mid-360 can see down to ~10cm
    pre_config.max_distance = 100.0;
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
        auto cb_start = std::chrono::high_resolution_clock::now();

        // Track callback interval
        uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        uint64_t last = g_timing.last_callback_time.exchange(now_ns);
        if (last > 0) {
            g_timing.callback_interval_us = (now_ns - last) / 1000.0;
        }

        std::lock_guard<std::mutex> lock(scan_mutex);

        // Record raw data if enabled
        if (g_record_file) {
            std::lock_guard<std::mutex> rec_lock(g_record_mutex);
            uint8_t msg_type = 1;  // 1 = point cloud
            uint32_t num_points = static_cast<uint32_t>(frame.points.size());
            uint64_t ts = frame.timestamp_ns;
            g_record_file->write(reinterpret_cast<char*>(&msg_type), 1);
            g_record_file->write(reinterpret_cast<char*>(&ts), sizeof(ts));
            g_record_file->write(reinterpret_cast<char*>(&num_points), sizeof(num_points));
            for (size_t i = 0; i < frame.points.size(); i++) {
                float x = static_cast<float>(frame.points[i].x());
                float y = static_cast<float>(frame.points[i].y());
                float z = static_cast<float>(frame.points[i].z());
                uint8_t refl = frame.reflectivities[i];
                g_record_file->write(reinterpret_cast<char*>(&x), sizeof(x));
                g_record_file->write(reinterpret_cast<char*>(&y), sizeof(y));
                g_record_file->write(reinterpret_cast<char*>(&z), sizeof(z));
                g_record_file->write(reinterpret_cast<char*>(&refl), sizeof(refl));
            }
        }

        // Debug: print first few frames
        static int frame_count = 0;
        if (frame_count < 5) {
            std::cout << "[DEBUG] Point cloud frame: " << frame.points.size() << " points" << std::endl;
            if (!frame.points.empty()) {
                // Print first point as sanity check
                const V3D& p = frame.points[0];
                std::cout << "[DEBUG] First point: [" << p.x() << ", " << p.y() << ", " << p.z() << "]" << std::endl;
            }
            frame_count++;
        }

        // Copy raw points for visualization (every frame for now)
        {
            std::lock_guard<std::mutex> raw_lock(g_raw_points_mutex);
            g_raw_points_buffer.clear();
            g_raw_points_buffer.reserve(frame.points.size());
            for (const auto& p : frame.points) {
                // Skip zero/invalid points
                if (std::abs(p.x()) < 0.001 && std::abs(p.y()) < 0.001 && std::abs(p.z()) < 0.001) continue;
                g_raw_points_buffer.push_back(p);
            }
            g_raw_points_ready = true;
        }

        // Start new scan if empty
        if (accumulated_points.empty()) {
            scan_start_time = frame.timestamp_ns;
        }

        // Calculate frame time offset from scan start (ms)
        double frame_offset_ms = (frame.timestamp_ns - scan_start_time) / 1e6;

        // Convert frame points to LidarPoint format
        // MATCHING ORIGINAL FAST-LIO Mid-360 settings:
        // - point_filter_num = configurable (default 3 = keep every 3rd valid point)
        // - blind = configurable (default 0.5m for indoor)
        // - tag filtering: only 0x00 (single return) and 0x10 (strongest return)
        static size_t valid_point_counter = 0;

        for (size_t i = 0; i < frame.points.size(); i++) {
            const V3D& p = frame.points[i];

            // Tag filtering - only keep valid returns (matching original FAST-LIO)
            // Tag bits 4-5: 0x00=single return, 0x10=strongest, 0x20=farthest, 0x30=multi
            if (!frame.tags.empty()) {
                uint8_t return_type = frame.tags[i] & 0x30;
                if (return_type != 0x00 && return_type != 0x10) continue;
            }

            // Skip invalid points (too close or too far)
            double dist = p.norm();
            if (dist < blind_dist || dist > 100.0) continue;

            // Skip zero points
            if (std::abs(p.x()) < 0.001 && std::abs(p.y()) < 0.001 && std::abs(p.z()) < 0.001) continue;

            // Point decimation - keep every Nth valid point (reduces noise, matches original)
            valid_point_counter++;
            if (point_filter > 1 && valid_point_counter % point_filter != 0) continue;

            LidarPoint lp;
            lp.x = static_cast<float>(p.x());
            lp.y = static_cast<float>(p.y());
            lp.z = static_cast<float>(p.z());
            lp.intensity = static_cast<float>(frame.reflectivities[i]);
            // Use per-point timestamp: frame offset + individual point offset within packet
            float point_offset_ms = (i < frame.time_offsets_us.size()) ?
                (frame.time_offsets_us[i] / 1000.0f) : 0.0f;
            lp.time_offset_ms = static_cast<float>(frame_offset_ms) + point_offset_ms;
            lp.tag = frame.tags.empty() ? 0 : frame.tags[i];
            lp.line = 0;

            accumulated_points.push_back(lp);
        }

        g_point_count += frame.points.size();

        // Check if we have accumulated enough for a scan (~100ms)
        if (frame_offset_ms >= SCAN_PERIOD_MS && accumulated_points.size() > 100) {
            // Debug: Check time offset distribution
            static int scan_debug_count = 0;
            if (scan_debug_count++ < 3) {
                float min_t = 1e9, max_t = -1e9;
                for (const auto& p : accumulated_points) {
                    if (p.time_offset_ms < min_t) min_t = p.time_offset_ms;
                    if (p.time_offset_ms > max_t) max_t = p.time_offset_ms;
                }
                std::cout << "[DEBUG scan] " << accumulated_points.size() << " pts, time_offset: "
                          << min_t << " - " << max_t << " ms (span=" << (max_t - min_t) << "ms)" << std::endl;
            }

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

        // Track callback timing
        auto cb_end = std::chrono::high_resolution_clock::now();
        double cb_us = std::chrono::duration<double, std::micro>(cb_end - cb_start).count();
        g_timing.callback_time_us = cb_us;
        if (cb_us > g_timing.callback_max_us) g_timing.callback_max_us = cb_us;
        g_timing.callback_count++;
        g_timing.points_queued = accumulated_points.size();
    });

    // IMU callback
    lidar.setIMUCallback([&](const LivoxIMUFrame& frame) {
        // Record IMU data if enabled
        if (g_record_file) {
            std::lock_guard<std::mutex> rec_lock(g_record_mutex);
            uint8_t msg_type = 2;  // 2 = IMU
            uint64_t ts = frame.timestamp_ns;
            g_record_file->write(reinterpret_cast<char*>(&msg_type), 1);
            g_record_file->write(reinterpret_cast<char*>(&ts), sizeof(ts));
            // Write accel (3 floats)
            float ax = static_cast<float>(frame.accel.x());
            float ay = static_cast<float>(frame.accel.y());
            float az = static_cast<float>(frame.accel.z());
            g_record_file->write(reinterpret_cast<char*>(&ax), sizeof(ax));
            g_record_file->write(reinterpret_cast<char*>(&ay), sizeof(ay));
            g_record_file->write(reinterpret_cast<char*>(&az), sizeof(az));
            // Write gyro (3 floats)
            float gx = static_cast<float>(frame.gyro.x());
            float gy = static_cast<float>(frame.gyro.y());
            float gz = static_cast<float>(frame.gyro.z());
            g_record_file->write(reinterpret_cast<char*>(&gx), sizeof(gx));
            g_record_file->write(reinterpret_cast<char*>(&gy), sizeof(gy));
            g_record_file->write(reinterpret_cast<char*>(&gz), sizeof(gz));
        }

        ImuData imu;
        imu.timestamp_ns = frame.timestamp_ns;
        // Livox Mid-360 accelerometer outputs in g units, convert to m/s²
        constexpr double G_M_S2 = 9.81;
        imu.acc = frame.accel * G_M_S2;
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
    auto last_viz_time = start_time;
    uint64_t last_points = 0;
    uint64_t last_imu = 0;
    std::vector<V3D> trajectory_positions;

    while (g_running) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

        if (run_seconds > 0 && elapsed >= run_seconds) break;

        // Process any available buffered sensor data
        // This syncs IMU + LiDAR and runs the SLAM optimization
        auto proc_start = std::chrono::high_resolution_clock::now();
        int processed = g_slam->process();
        auto proc_end = std::chrono::high_resolution_clock::now();
        if (processed > 0) {
            double proc_us = std::chrono::duration<double, std::micro>(proc_end - proc_start).count();
            g_timing.process_time_us = proc_us;
            if (proc_us > g_timing.process_max_us) g_timing.process_max_us = proc_us;
            g_timing.process_count++;
        }

        // Visualize RAW point cloud (before SLAM processing)
        if (visualizer && g_raw_points_ready.load()) {
            std::vector<V3D> raw_points;
            {
                std::lock_guard<std::mutex> lock(g_raw_points_mutex);
                raw_points = g_raw_points_buffer;
                g_raw_points_ready = false;
            }
            if (!raw_points.empty()) {
                visualizer->setTime("slam_time", elapsed_ms / 1000.0);
                // Log raw points in GREEN to distinguish from processed data
                visualizer->logPointCloud("world/raw_lidar", raw_points, slam::colors::GREEN);
            }
        }

        // Update visualization after processing
        if (visualizer && processed > 0) {
            // Set timeline
            visualizer->setTime("slam_time", elapsed_ms / 1000.0);

            // Log current pose
            SlamState state = g_slam->getState();
            visualizer->logPose("world/robot", state);

            // Log coordinate frame at robot position
            visualizer->logCoordinateFrame("world/robot/frame", 0.3f);

            // Add to trajectory and log
            trajectory_positions.push_back(state.pos);
            if (trajectory_positions.size() > 1) {
                visualizer->logTrajectory("world/trajectory", trajectory_positions);
            }
        }

        // Update map visualization periodically (every 1000ms to reduce overhead)
        auto viz_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_viz_time).count();
        if (visualizer && viz_elapsed >= 1000) {
            // Get map points from SLAM and visualize
            auto map_points = g_slam->getMapPoints();
            if (!map_points.empty()) {
                // Convert WorldPoint to V3D for visualization
                std::vector<V3D> map_v3d;
                map_v3d.reserve(map_points.size());
                for (const auto& wp : map_points) {
                    map_v3d.emplace_back(wp.x, wp.y, wp.z);
                }
                visualizer->logPointCloud("world/map", map_v3d, slam::colors::CYAN);
            }
            last_viz_time = now;
        }

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

            // Display status with timing debug
            std::cout << "\r  Time: " << elapsed << "s | "
                      << "Pts: " << (pts_rate / 1000) << "k/s | "
                      << "Scans: " << cur_scans << " | "
                      << "Map: " << (map_size / 1000) << "k | "
                      << "Pos: [" << std::fixed << std::setprecision(2)
                      << state.pos.x() << ", " << state.pos.y() << ", " << state.pos.z() << "]    "
                      << std::flush;

            // Debug timing line (every 5 seconds)
            if (elapsed % 5 == 0) {
                auto slam_timing = g_slam->getDebugTiming();
                std::cout << "\n  [DEBUG] CB: " << std::fixed << std::setprecision(0)
                          << g_timing.callback_time_us.load() << "us (max:" << g_timing.callback_max_us.load() << ") | "
                          << "CB interval: " << g_timing.callback_interval_us.load() << "us | "
                          << "Queued: " << g_timing.points_queued.load() << " pts" << std::endl;
                std::cout << "  [SLAM] Total: " << slam_timing.total_us << "us | "
                          << "IMU: " << slam_timing.imu_process_us << "us | "
                          << "DS: " << slam_timing.downsample_us << "us | "
                          << "ICP: " << slam_timing.icp_us << "us | "
                          << "Map: " << slam_timing.map_update_us << "us | "
                          << "pts: " << slam_timing.points_in << "->" << slam_timing.points_after_ds
                          << std::endl;
            }

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

    // Close recording file
    if (g_record_file) {
        g_record_file->close();
        delete g_record_file;
        g_record_file = nullptr;
        std::cout << "  Recording saved: " << record_file << "\n";
    }

    std::cout << "============================================\n\n";

    return 0;
}
