/**
 * @file slam_main.cpp
 * @brief Main SLAM application with Livox SDK2 integration
 *
 * This is the standalone SLAM application for the Livox Mid-360.
 * It replaces the ROS node with direct Livox SDK2 callbacks.
 *
 * Features:
 * - Gravity alignment at startup (handles non-level start)
 * - Intensity preservation in all outputs
 * - PLY map export with intensity
 * - Real-time state output
 *
 * Build on Jetson: cmake .. && make
 * Run: ./slam_main [config.yaml]
 */

#include <iostream>
#include <csignal>
#include <atomic>
#include <chrono>
#include <thread>
#include <fstream>
#include <random>

// Livox SDK2 (conditional)
#ifdef LIVOX_SDK2_ENABLED
#include "livox_lidar_api.h"
#include "livox_lidar_def.h"
#else
// Stub types for compilation without Livox SDK
typedef struct {
    uint64_t time_stamp;
    uint32_t dot_num;
    uint8_t data_type;
    uint8_t data[1];
} LivoxLidarEthernetPacket;

typedef struct {
    int32_t x, y, z;
    uint8_t reflectivity;
    uint8_t tag;
    uint8_t line;
} LivoxLidarCartesianHighRawPoint;

typedef struct {
    int16_t x, y, z;
    uint8_t reflectivity;
    uint8_t tag;
} LivoxLidarCartesianLowRawPoint;

typedef struct {
    float gyro_x, gyro_y, gyro_z;
    float acc_x, acc_y, acc_z;
} LivoxLidarImuRawPoint;

enum { kLivoxLidarCartesianCoordinateHighData = 1, kLivoxLidarCartesianCoordinateLowData = 2, kLivoxLidarImuData = 3 };

inline bool LivoxLidarSdkInit(const char* path) { (void)path; return false; }
inline void LivoxLidarSdkUninit() {}
inline void SetLivoxLidarPointCloudCallBack(void* cb, void* data) { (void)cb; (void)data; }
inline void SetLivoxLidarImuDataCallback(void* cb, void* data) { (void)cb; (void)data; }
inline void SetLivoxLidarInfoCallback(void* cb, void* data) { (void)cb; (void)data; }
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// SLAM stack
#include "slam/types.hpp"
#include "slam/preprocess.hpp"
#include "slam/slam_engine.hpp"

using namespace slam;

//=============================================================================
// Global state
//=============================================================================

std::atomic<bool> g_running{true};
std::unique_ptr<SlamEngine> g_slam;
std::unique_ptr<Preprocessor> g_preprocessor;

// Statistics
std::atomic<uint64_t> g_point_count{0};
std::atomic<uint64_t> g_imu_count{0};
std::atomic<uint64_t> g_scan_count{0};

//=============================================================================
// Signal handler
//=============================================================================

void signalHandler(int sig) {
    std::cout << "\n[SLAM] Caught signal " << sig << ", shutting down..." << std::endl;
    g_running = false;
}

//=============================================================================
// Livox SDK2 Callbacks
//=============================================================================

/**
 * @brief Callback for Livox point cloud data
 *
 * This callback receives raw point data from the Mid-360.
 * The points are in LivoxPointXYZRTLT format with:
 * - x, y, z: Position in mm (int32_t)
 * - reflectivity: 0-255 (uint8_t)
 * - tag: Return type flags
 * - line: Scan line (0-3 for Mid-360)
 * - offset_time: Time offset in ns from packet timestamp
 */
void livoxPointCloudCallback(uint32_t handle, const uint8_t dev_type,
                              LivoxLidarEthernetPacket* data, void* client_data) {
    if (!data || !g_running) return;

    // Get timestamp (from packet)
    uint64_t timestamp_ns = data->time_stamp;

    // Parse point data
    if (data->data_type == kLivoxLidarCartesianCoordinateHighData) {
        // High-resolution Cartesian data
        LivoxLidarCartesianHighRawPoint* points =
            reinterpret_cast<LivoxLidarCartesianHighRawPoint*>(data->data);
        uint32_t point_num = data->dot_num;

        // Convert to our format
        std::vector<LivoxPointXYZRTLT> raw_points;
        raw_points.reserve(point_num);

        for (uint32_t i = 0; i < point_num; i++) {
            LivoxPointXYZRTLT pt;
            pt.x = points[i].x;          // Already in mm
            pt.y = points[i].y;
            pt.z = points[i].z;
            pt.reflectivity = points[i].reflectivity;
            pt.tag = points[i].tag;
            pt.line = points[i].line;
            pt.offset_time = i * 1000;   // Approximate timing
            raw_points.push_back(pt);
        }

        // Preprocess and add to SLAM
        PointCloud cloud;
        g_preprocessor->process(raw_points, timestamp_ns, cloud);

        if (!cloud.empty()) {
            g_slam->addPointCloud(cloud);
            g_point_count += cloud.size();
            g_scan_count++;
        }
    }
    else if (data->data_type == kLivoxLidarCartesianCoordinateLowData) {
        // Low-resolution Cartesian data (fallback)
        LivoxLidarCartesianLowRawPoint* points =
            reinterpret_cast<LivoxLidarCartesianLowRawPoint*>(data->data);
        uint32_t point_num = data->dot_num;

        std::vector<LivoxPointXYZRTLT> raw_points;
        raw_points.reserve(point_num);

        for (uint32_t i = 0; i < point_num; i++) {
            LivoxPointXYZRTLT pt;
            pt.x = points[i].x;
            pt.y = points[i].y;
            pt.z = points[i].z;
            pt.reflectivity = points[i].reflectivity;
            pt.tag = points[i].tag;
            pt.line = 0;
            pt.offset_time = i * 1000;
            raw_points.push_back(pt);
        }

        PointCloud cloud;
        g_preprocessor->process(raw_points, timestamp_ns, cloud);

        if (!cloud.empty()) {
            g_slam->addPointCloud(cloud);
            g_point_count += cloud.size();
            g_scan_count++;
        }
    }
}

/**
 * @brief Callback for Livox IMU data
 *
 * The Mid-360 has an integrated IMU running at 200Hz.
 * Data includes:
 * - gyro_x/y/z: Angular velocity (rad/s)
 * - acc_x/y/z: Linear acceleration (g, needs *9.81)
 */
void livoxImuCallback(uint32_t handle, const uint8_t dev_type,
                      LivoxLidarEthernetPacket* data, void* client_data) {
    if (!data || !g_running) return;

    if (data->data_type == kLivoxLidarImuData) {
        LivoxLidarImuRawPoint* imu_data =
            reinterpret_cast<LivoxLidarImuRawPoint*>(data->data);

        uint64_t timestamp_ns = data->time_stamp;

        // Convert to our IMU format
        ImuData imu;
        imu.timestamp_ns = timestamp_ns;
        imu.gyro = V3D(imu_data->gyro_x, imu_data->gyro_y, imu_data->gyro_z);
        imu.acc = V3D(imu_data->acc_x, imu_data->acc_y, imu_data->acc_z) * 9.81;

        g_slam->addImuData(imu);
        g_imu_count++;
    }
}

/**
 * @brief Callback for Livox device info
 */
void livoxInfoCallback(const uint32_t handle, const uint8_t dev_type,
                       const char* info, void* client_data) {
    std::cout << "[Livox] Device info: " << info << std::endl;
}

//=============================================================================
// Configuration loading
//=============================================================================

bool loadConfig(const std::string& filename, SlamConfig& config, PreprocessConfig& pre_config) {
    // Simple YAML parser for our config format
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[Config] Could not open: " << filename << std::endl;
        return false;
    }

    std::string line;
    std::string section;

    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        // Check for section headers
        if (line.find("common:") != std::string::npos) { section = "common"; continue; }
        if (line.find("preprocess:") != std::string::npos) { section = "preprocess"; continue; }
        if (line.find("imu:") != std::string::npos) { section = "imu"; continue; }
        if (line.find("slam:") != std::string::npos) { section = "slam"; continue; }
        if (line.find("extrinsic:") != std::string::npos) { section = "extrinsic"; continue; }
        if (line.find("output:") != std::string::npos) { section = "output"; continue; }

        // Parse key-value pairs
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;

        std::string key = line.substr(0, colon_pos);
        std::string value = line.substr(colon_pos + 1);

        // Trim whitespace
        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t"));
            s.erase(s.find_last_not_of(" \t") + 1);
        };
        trim(key);
        trim(value);

        // Parse based on section and key
        if (section == "preprocess") {
            if (key == "blind_distance") pre_config.blind_distance = std::stod(value);
            if (key == "max_distance") pre_config.max_distance = std::stod(value);
            if (key == "point_filter_num") pre_config.point_filter_num = std::stoi(value);
        }
        else if (section == "imu") {
            if (key == "acc_noise") config.acc_cov = std::stod(value);
            if (key == "gyro_noise") config.gyr_cov = std::stod(value);
            if (key == "acc_bias_noise") config.b_acc_cov = std::stod(value);
            if (key == "gyro_bias_noise") config.b_gyr_cov = std::stod(value);
        }
        else if (section == "slam") {
            if (key == "max_iterations") config.max_iterations = std::stoi(value);
            if (key == "map_resolution") {
                config.filter_size_map = std::stod(value);
                config.filter_size_surf = std::stod(value);
            }
            if (key == "deskew_enabled") config.deskew_enabled = (value == "true");
        }
        else if (section == "output") {
            if (key == "save_map") config.save_map = (value == "true");
        }
    }

    std::cout << "[Config] Loaded: " << filename << std::endl;
    return true;
}

//=============================================================================
// Status display
//=============================================================================

void printStatus() {
    static auto last_time = std::chrono::steady_clock::now();
    static uint64_t last_points = 0;
    static uint64_t last_scans = 0;

    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_time).count();

    if (dt >= 1.0) {
        uint64_t current_points = g_point_count;
        uint64_t current_scans = g_scan_count;

        double points_per_sec = (current_points - last_points) / dt;
        double scans_per_sec = (current_scans - last_scans) / dt;

        SlamState state = g_slam->getState();
        size_t map_size = g_slam->getMapSize();

        std::cout << "\r[SLAM] "
                  << "Scans: " << scans_per_sec << "/s | "
                  << "Points: " << (points_per_sec / 1000.0) << "k/s | "
                  << "Map: " << (map_size / 1000.0) << "k | "
                  << "Pos: [" << state.pos.x() << ", " << state.pos.y() << ", " << state.pos.z() << "]"
                  << std::flush;

        last_time = now;
        last_points = current_points;
        last_scans = current_scans;
    }
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Non-ROS SLAM Stack" << std::endl;
    std::cout << "  For Livox Mid-360" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Setup signal handler
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // Load configuration
    std::string config_file = "config/mid360.yaml";
    if (argc > 1) {
        config_file = argv[1];
    }

    SlamConfig slam_config;
    PreprocessConfig pre_config;
    pre_config.lidar_type = LidarType::LIVOX_MID360;
    pre_config.n_scans = 4;
    pre_config.time_unit = TimeUnit::NS;

    if (!loadConfig(config_file, slam_config, pre_config)) {
        std::cout << "[Config] Using default configuration" << std::endl;
    }

    // Initialize preprocessor
    g_preprocessor = std::make_unique<Preprocessor>(pre_config);
    std::cout << "[Preprocess] Initialized" << std::endl;
    std::cout << "  Blind distance: " << pre_config.blind_distance << "m" << std::endl;
    std::cout << "  Max distance: " << pre_config.max_distance << "m" << std::endl;

    // Initialize SLAM engine
    g_slam = std::make_unique<SlamEngine>();
    slam_config.save_map = true;
    slam_config.map_save_path = "slam_output_map.ply";

    if (!g_slam->init(slam_config)) {
        std::cerr << "[SLAM] Failed to initialize!" << std::endl;
        return 1;
    }

    // Set state callback for real-time output
    g_slam->setStateCallback([](const SlamState& state) {
        // Could publish to shared memory here for visualization
        // For now, just used for status display
    });

    std::cout << "\n[Livox] Initializing SDK2..." << std::endl;

    // Initialize Livox SDK2
    if (!LivoxLidarSdkInit(config_file.c_str())) {
        std::cerr << "[Livox] SDK initialization failed!" << std::endl;
        std::cerr << "        Make sure config file exists and LiDAR is connected" << std::endl;

        // Run in simulation mode for testing
        std::cout << "\n[SLAM] Running in SIMULATION mode for testing..." << std::endl;

        // Generate simulated data for testing
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> pos_noise(-0.01f, 0.01f);
        std::uniform_real_distribution<float> imu_noise(-0.001f, 0.001f);
        std::normal_distribution<float> refl_dist(128.0f, 30.0f);

        uint64_t timestamp_ns = 0;
        double dt_imu = 0.005;  // 200 Hz
        double dt_lidar = 0.1; // 10 Hz

        int frame = 0;
        while (g_running && frame < 100) {
            // Simulate IMU at 200 Hz for 100ms
            for (int i = 0; i < 20 && g_running; i++) {
                ImuData imu;
                imu.timestamp_ns = timestamp_ns;
                imu.gyro = V3D(0.01 + imu_noise(gen), imu_noise(gen), imu_noise(gen));
                imu.acc = V3D(imu_noise(gen), imu_noise(gen), -9.81 + imu_noise(gen));
                g_slam->addImuData(imu);
                g_imu_count++;
                timestamp_ns += static_cast<uint64_t>(dt_imu * 1e9);
            }

            // Simulate LiDAR scan
            std::vector<LivoxPointXYZRTLT> raw_points;
            for (int i = 0; i < 5000; i++) {
                LivoxPointXYZRTLT pt;
                float angle = static_cast<float>(i) / 5000.0f * 2.0f * M_PI;
                float r = 3000.0f + pos_noise(gen) * 100.0f;  // ~3m in mm
                pt.x = static_cast<int32_t>(r * std::cos(angle));
                pt.y = static_cast<int32_t>(r * std::sin(angle));
                pt.z = static_cast<int32_t>(500 + pos_noise(gen) * 1000);
                pt.reflectivity = static_cast<uint8_t>(std::clamp(refl_dist(gen), 1.0f, 255.0f));
                pt.tag = 0;
                pt.line = i % 4;
                pt.offset_time = i * 1000;
                raw_points.push_back(pt);
            }

            PointCloud cloud;
            g_preprocessor->process(raw_points, timestamp_ns, cloud);
            g_slam->addPointCloud(cloud);
            g_point_count += cloud.size();
            g_scan_count++;

            printStatus();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            frame++;
        }
    }
    else {
        // Real hardware mode
        std::cout << "[Livox] SDK initialized successfully" << std::endl;

        // Register callbacks
        SetLivoxLidarPointCloudCallBack(livoxPointCloudCallback, nullptr);
        SetLivoxLidarImuDataCallback(livoxImuCallback, nullptr);
        SetLivoxLidarInfoCallback(livoxInfoCallback, nullptr);

        std::cout << "[Livox] Callbacks registered" << std::endl;
        std::cout << "\n[SLAM] Running... Press Ctrl+C to stop\n" << std::endl;

        // Main loop
        while (g_running) {
            printStatus();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Cleanup Livox SDK
        LivoxLidarSdkUninit();
    }

    // Save results
    std::cout << "\n\n[SLAM] Saving results..." << std::endl;

    if (g_slam->saveMap("slam_output_map.ply")) {
        std::cout << "[SLAM] Map saved: slam_output_map.ply" << std::endl;
        std::cout << "       Points: " << g_slam->getMapSize() << std::endl;
    }

    if (g_slam->saveTrajectory("slam_output_trajectory.ply")) {
        std::cout << "[SLAM] Trajectory saved: slam_output_trajectory.ply" << std::endl;
        std::cout << "       Poses: " << g_slam->getTrajectory().size() << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Session Statistics" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Total scans processed: " << g_scan_count << std::endl;
    std::cout << "  Total points processed: " << g_point_count << std::endl;
    std::cout << "  Total IMU measurements: " << g_imu_count << std::endl;
    std::cout << "  Final map size: " << g_slam->getMapSize() << " points" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}
