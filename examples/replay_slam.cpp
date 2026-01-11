/**
 * @file replay_slam.cpp
 * @brief Replay recorded LiDAR/IMU data for offline SLAM testing
 *
 * Reads binary recordings from live_slam --record and replays them
 * through the SLAM engine for debugging and development.
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
#include <vector>
#include <algorithm>

#include "slam/slam_engine.hpp"
#include "slam/types.hpp"
#include "slam/visualization_interface.hpp"

using namespace slam;

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

    file.close();
    return true;
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <recording.bin> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --output <name>  Output file prefix (default: replay)\n";
    std::cout << "  --voxel <size>   Map voxel size in meters (default: 0.05)\n";
    std::cout << "  --iterations <n> IEKF iterations per scan (default: 5, range: 1-10)\n";
    std::cout << "  --gyr-cov <val>  Gyroscope noise covariance (default: 0.1)\n";
    std::cout << "  --blind <dist>   Minimum point range in meters (default: 0.5)\n";
    std::cout << "  --filter <n>     Keep every Nth point (default: 3, 1=keep all)\n";
    std::cout << "  --visualize      Enable visualization (requires Rerun)\n";
    std::cout << "  --speed <x>      Playback speed multiplier (default: 1.0, 0=fast)\n";
    std::cout << "  --help           Show this help\n";
}

int main(int argc, char** argv) {
    std::cout << "============================================\n";
    std::cout << "  SLAM Replay - Offline Testing\n";
    std::cout << "============================================\n\n";

    if (argc < 2 || std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return argc < 2 ? 1 : 0;
    }

    // Parse arguments
    std::string recording_file = argv[1];
    std::string output_prefix = "replay";
    double voxel_size = 0.05;
    int max_iterations = 5;
    double gyr_cov = 0.1;
    double blind_dist = 0.5;
    int point_filter = 3;
    bool enable_visualization = false;
    double playback_speed = 1.0;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
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
        } else if (arg == "--speed" && i + 1 < argc) {
            playback_speed = std::atof(argv[++i]);
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Load recording
    std::cout << "Loading recording: " << recording_file << std::endl;
    std::vector<RecordedPointCloud> clouds;
    std::vector<RecordedIMU> imus;

    if (!loadRecording(recording_file, clouds, imus)) {
        return 1;
    }

    std::cout << "  Point cloud frames: " << clouds.size() << std::endl;
    std::cout << "  IMU samples: " << imus.size() << std::endl;

    if (clouds.empty() || imus.empty()) {
        std::cerr << "No data in recording!" << std::endl;
        return 1;
    }

    // Calculate total points
    size_t total_points = 0;
    for (const auto& c : clouds) total_points += c.points.size();
    std::cout << "  Total points: " << total_points << std::endl;

    // Get time range
    uint64_t min_time = std::min(clouds[0].timestamp_ns, imus[0].timestamp_ns);
    uint64_t max_time = std::max(clouds.back().timestamp_ns, imus.back().timestamp_ns);
    double duration_sec = (max_time - min_time) / 1e9;
    std::cout << "  Duration: " << duration_sec << " seconds\n\n";

    // Initialize SLAM engine
    std::cout << "Initializing SLAM engine...\n";
    SlamEngine slam;

    SlamConfig config;
    config.filter_size_surf = voxel_size;
    config.filter_size_map = voxel_size;
    config.deskew_enabled = true;  // CRITICAL: Enable motion compensation for proper tracking
    config.save_map = true;
    config.map_save_path = output_prefix + "_map.ply";
    // IMU noise parameters
    config.gyr_cov = gyr_cov;
    config.acc_cov = gyr_cov;
    config.b_gyr_cov = 0.0001;
    config.b_acc_cov = 0.0001;
    config.max_iterations = max_iterations;

    if (!slam.init(config)) {
        std::cerr << "Failed to initialize SLAM engine!\n";
        return 1;
    }

    std::cout << "  Voxel size: " << voxel_size << "m\n";
    std::cout << "  Iterations: " << max_iterations << "\n";
    std::cout << "  Gyr cov: " << gyr_cov << "\n";
    std::cout << "  Blind dist: " << blind_dist << "m\n";
    std::cout << "  Point filter: " << point_filter << "\n";
    std::cout << "  Playback speed: " << (playback_speed > 0 ? std::to_string(playback_speed) + "x" : "max") << "\n";

    // Initialize visualization
    std::unique_ptr<Visualizer> visualizer;
    if (enable_visualization) {
        VisualizerConfig viz_config;
        viz_config.application_id = "slam_replay";
        viz_config.spawn_viewer = true;
        viz_config.color_by_height = true;
        viz_config.point_size = 4.0f;  // Doubled for visibility
        viz_config.max_points_per_frame = 200000;  // Increased for high density

        visualizer = std::make_unique<Visualizer>(viz_config);
        if (visualizer->isInitialized()) {
            std::cout << "  Visualization: enabled\n";
        } else {
            std::cout << "  Visualization: failed to initialize\n";
            visualizer.reset();
        }
    }
    std::cout << "\n";

    // Merge and sort all events by timestamp
    struct Event {
        uint64_t timestamp_ns;
        bool is_imu;  // true = IMU, false = point cloud
        size_t index;
    };

    std::vector<Event> events;
    events.reserve(clouds.size() + imus.size());

    for (size_t i = 0; i < clouds.size(); i++) {
        events.push_back({clouds[i].timestamp_ns, false, i});
    }
    for (size_t i = 0; i < imus.size(); i++) {
        events.push_back({imus[i].timestamp_ns, true, i});
    }

    std::sort(events.begin(), events.end(),
              [](const Event& a, const Event& b) { return a.timestamp_ns < b.timestamp_ns; });

    // Replay data
    std::cout << "Starting replay...\n\n";
    auto start_time = std::chrono::steady_clock::now();
    uint64_t base_timestamp = events[0].timestamp_ns;
    size_t cloud_count = 0;
    size_t imu_count = 0;

    // Accumulate points for complete scans
    std::vector<LidarPoint> accumulated_points;
    uint64_t scan_start_time = 0;
    const double SCAN_PERIOD_MS = 100.0;
    accumulated_points.reserve(20000);

    std::vector<V3D> trajectory_positions;

    for (size_t ei = 0; ei < events.size(); ei++) {
        const Event& event = events[ei];

        // Calculate expected playback time
        if (playback_speed > 0) {
            double event_time_sec = (event.timestamp_ns - base_timestamp) / 1e9;
            double target_time_sec = event_time_sec / playback_speed;

            auto elapsed = std::chrono::steady_clock::now() - start_time;
            double elapsed_sec = std::chrono::duration<double>(elapsed).count();

            if (elapsed_sec < target_time_sec) {
                std::this_thread::sleep_for(
                    std::chrono::duration<double>(target_time_sec - elapsed_sec));
            }
        }

        if (event.is_imu) {
            // Process IMU
            const RecordedIMU& rec_imu = imus[event.index];
            ImuData imu;
            imu.timestamp_ns = rec_imu.timestamp_ns;
            // Recorded data is in g units, convert to m/s²
            constexpr double G_M_S2 = 9.81;
            imu.acc = rec_imu.accel * G_M_S2;
            imu.gyro = rec_imu.gyro;
            slam.addImuData(imu);
            imu_count++;

        } else {
            // Process point cloud frame
            const RecordedPointCloud& rec_cloud = clouds[event.index];

            // Start new scan if empty
            if (accumulated_points.empty()) {
                scan_start_time = rec_cloud.timestamp_ns;
            }

            // Calculate time offset
            double time_offset_ms = (rec_cloud.timestamp_ns - scan_start_time) / 1e6;

            // Add points to accumulator
            static size_t valid_point_counter = 0;
            for (size_t i = 0; i < rec_cloud.points.size(); i++) {
                const V3D& p = rec_cloud.points[i];

                // Skip invalid points (too close or too far)
                double dist = p.norm();
                if (dist < blind_dist || dist > 100.0) continue;
                if (std::abs(p.x()) < 0.001 && std::abs(p.y()) < 0.001 && std::abs(p.z()) < 0.001) continue;

                // Point decimation - keep every Nth valid point
                valid_point_counter++;
                if (point_filter > 1 && valid_point_counter % point_filter != 0) continue;

                LidarPoint lp;
                lp.x = static_cast<float>(p.x());
                lp.y = static_cast<float>(p.y());
                lp.z = static_cast<float>(p.z());
                lp.intensity = static_cast<float>(rec_cloud.reflectivities[i]);
                lp.time_offset_ms = static_cast<float>(time_offset_ms);
                lp.tag = 0;
                lp.line = 0;

                accumulated_points.push_back(lp);
            }

            // Check if scan is complete
            if (time_offset_ms >= SCAN_PERIOD_MS && accumulated_points.size() > 100) {
                PointCloud cloud;
                cloud.timestamp_ns = scan_start_time;
                cloud.points = std::move(accumulated_points);

                slam.addPointCloud(cloud);
                cloud_count++;

                accumulated_points.clear();
                accumulated_points.reserve(20000);

                // Process SLAM
                int processed = slam.process();

                // Update visualization
                if (visualizer && processed > 0) {
                    double elapsed_sec = (event.timestamp_ns - base_timestamp) / 1e9;
                    visualizer->setTime("slam_time", elapsed_sec);

                    SlamState state = slam.getState();
                    visualizer->logPose("world/robot", state);
                    visualizer->logCoordinateFrame("world/robot/frame", 0.3f);

                    trajectory_positions.push_back(state.pos);
                    if (trajectory_positions.size() > 1) {
                        visualizer->logTrajectory("world/trajectory", trajectory_positions);
                    }

                    // Log map periodically
                    static int viz_counter = 0;
                    if (++viz_counter % 5 == 0) {
                        auto map_points = slam.getMapPoints();
                        if (!map_points.empty()) {
                            std::vector<V3D> map_v3d;
                            map_v3d.reserve(map_points.size());
                            for (const auto& wp : map_points) {
                                map_v3d.emplace_back(wp.x, wp.y, wp.z);
                            }
                            visualizer->logPointCloud("world/map", map_v3d, slam::colors::CYAN);
                        }
                    }
                }
            }
        }

        // Progress update
        if ((ei + 1) % 1000 == 0 || ei == events.size() - 1) {
            double progress = 100.0 * (ei + 1) / events.size();
            SlamState state = slam.getState();
            std::cout << "\r  Progress: " << std::fixed << std::setprecision(1) << progress << "% | "
                      << "Scans: " << cloud_count << " | "
                      << "IMU: " << imu_count << " | "
                      << "Map: " << slam.getMapSize() << " | "
                      << "Pos: [" << std::setprecision(2)
                      << state.pos.x() << ", " << state.pos.y() << ", " << state.pos.z() << "]    "
                      << std::flush;
        }
    }

    std::cout << "\n\nReplay complete!\n";

    // Save results
    std::cout << "\n============================================\n";
    std::cout << "  Results\n";
    std::cout << "============================================\n";

    std::string map_file = output_prefix + "_map.ply";
    std::string traj_file = output_prefix + "_trajectory.ply";

    if (slam.saveMap(map_file)) {
        std::cout << "  Map saved: " << map_file << " (" << slam.getMapSize() << " points)\n";
    }

    if (slam.saveTrajectory(traj_file)) {
        std::cout << "  Trajectory saved: " << traj_file << " (" << slam.getTrajectory().size() << " poses)\n";
    }

    std::cout << "\n  Processed scans: " << cloud_count << "\n";
    std::cout << "  Processed IMU: " << imu_count << "\n";
    std::cout << "============================================\n\n";

    return 0;
}
