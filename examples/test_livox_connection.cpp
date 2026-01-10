/**
 * @file test_livox_connection.cpp
 * @brief Test Livox Mid-360 discovery and connection
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

#include "slam/livox_mid360.hpp"
#include "slam/ply_export.hpp"

using namespace slam;

// Simple PLY export for V3D vectors
inline bool saveToPly(const std::string& filename, const std::vector<V3D>& points) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    // Header
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << points.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "end_header\n";

    // Data
    for (const auto& p : points) {
        float x = static_cast<float>(p.x());
        float y = static_cast<float>(p.y());
        float z = static_cast<float>(p.z());
        file.write(reinterpret_cast<const char*>(&x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&y), sizeof(float));
        file.write(reinterpret_cast<const char*>(&z), sizeof(float));
    }

    return true;
}

std::atomic<bool> g_running{true};

void signalHandler(int) {
    g_running = false;
}

int main(int argc, char** argv) {
    std::cout << "============================================\n";
    std::cout << "  Livox Mid-360 Connection Test\n";
    std::cout << "============================================\n\n";

    // Parse arguments
    std::string host_ip = "192.168.1.50";
    std::string device_ip = "";  // Auto-discover
    std::string serial_suffix = "";  // Last 2-3 digits of serial number
    int capture_seconds = 5;
    bool save_ply = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) {
            host_ip = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device_ip = argv[++i];
        } else if (arg == "--serial" && i + 1 < argc) {
            serial_suffix = argv[++i];
            // Extract last 2 digits and compute IP
            if (serial_suffix.length() >= 2) {
                std::string last2 = serial_suffix.substr(serial_suffix.length() - 2);
                device_ip = "192.168.1.1" + last2;
            }
        } else if (arg == "--time" && i + 1 < argc) {
            capture_seconds = std::atoi(argv[++i]);
        } else if (arg == "--save") {
            save_ply = true;
        } else if (arg == "--scan") {
            // Just scan and exit
            scanLivoxDevices();
            return 0;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --scan           Scan for devices and exit\n";
            std::cout << "  --serial <num>   Last 2-3 digits of serial number (e.g., 544 -> IP 192.168.1.144)\n";
            std::cout << "  --device <ip>    Device IP directly (default: auto-discover)\n";
            std::cout << "  --host <ip>      Host IP address (default: 192.168.1.50)\n";
            std::cout << "  --time <sec>     Capture duration in seconds (default: 5)\n";
            std::cout << "  --save           Save captured points to PLY file\n";
            std::cout << "  --help           Show this help\n";
            return 0;
        }
    }

    // Register signal handler for clean shutdown
    std::signal(SIGINT, signalHandler);

    // Step 1: Check network configuration
    std::cout << "Network Configuration:\n";
    std::cout << "  Host IP: " << host_ip << "\n";
    if (!serial_suffix.empty()) {
        std::cout << "  Serial #: ..." << serial_suffix << " -> IP " << device_ip << "\n\n";
    } else {
        std::cout << "  Device IP: " << (device_ip.empty() ? "(auto-discover)" : device_ip) << "\n\n";
    }

    LivoxMid360 lidar;
    std::string target_ip = device_ip;

    // Step 2: Discover or use specified device
    if (device_ip.empty()) {
        std::cout << "Step 1: Scanning for Livox devices...\n";
        auto devices = lidar.discover(3000, host_ip);

        if (devices.empty()) {
            std::cout << "\n[!] No devices found via broadcast.\n";
            std::cout << "    Try specifying the device IP directly:\n";
            std::cout << "    --device 192.168.1.1XX  (XX = last 2 digits of serial)\n\n";
            return 1;
        }
        target_ip = devices[0].ip_address;
        std::cout << "Found: " << devices[0].getTypeName() << " at " << target_ip << "\n";
    } else {
        std::cout << "Step 1: Using specified device IP: " << device_ip << "\n";
    }

    // Step 3: Connect to device
    std::cout << "\nStep 2: Connecting to " << target_ip << "...\n";

    if (!lidar.connect(target_ip, host_ip)) {
        std::cerr << "Failed to connect!\n";
        return 1;
    }

    std::cout << "Connected!\n\n";

    // Step 4: Start streaming
    std::cout << "Step 3: Starting point cloud stream...\n";

    std::vector<V3D> all_points;
    std::mutex points_mutex;
    std::atomic<uint64_t> packet_count{0};
    std::atomic<uint64_t> point_count{0};
    std::atomic<uint64_t> imu_count{0};
    V3D last_accel{0, 0, 0};
    V3D last_gyro{0, 0, 0};

    lidar.setPointCloudCallback([&](const LivoxPointCloudFrame& frame) {
        packet_count++;
        point_count += frame.points.size();

        if (save_ply) {
            std::lock_guard<std::mutex> lock(points_mutex);
            all_points.insert(all_points.end(), frame.points.begin(), frame.points.end());
        }
    });

    lidar.setIMUCallback([&](const LivoxIMUFrame& frame) {
        imu_count++;
        last_accel = frame.accel;
        last_gyro = frame.gyro;
    });

    if (!lidar.startStreaming()) {
        std::cerr << "Failed to start streaming!\n";
        return 1;
    }

    std::cout << "Streaming started! Capturing for " << capture_seconds << " seconds...\n\n";

    // Step 5: Monitor for specified duration
    auto start_time = std::chrono::steady_clock::now();
    uint64_t last_points = 0;

    while (g_running) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        if (elapsed >= capture_seconds) break;

        // Print stats every second
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        uint64_t current_points = point_count.load();
        uint64_t points_per_sec = current_points - last_points;
        last_points = current_points;

        std::cout << "\r  Time: " << elapsed + 1 << "s | "
                  << "Packets: " << packet_count.load() << " | "
                  << "Points: " << current_points << " | "
                  << "Rate: " << (points_per_sec / 1000) << "k pts/s    " << std::flush;
    }

    std::cout << "\n\n";

    // Step 6: Stop and report
    lidar.stop();

    std::cout << "============================================\n";
    std::cout << "  Capture Summary\n";
    std::cout << "============================================\n";
    std::cout << "  Duration: " << capture_seconds << " seconds\n";
    std::cout << "  Point cloud packets: " << packet_count.load() << "\n";
    std::cout << "  Total points: " << point_count.load() << "\n";
    std::cout << "  Average rate: " << (point_count.load() / capture_seconds / 1000) << " k points/sec\n";
    std::cout << "  IMU samples: " << imu_count.load() << " (" << (imu_count.load() / capture_seconds) << " Hz)\n";
    if (imu_count.load() > 0) {
        std::cout << "  Last accel: (" << last_accel.x() << ", " << last_accel.y() << ", " << last_accel.z() << ") m/s^2\n";
        std::cout << "  Last gyro:  (" << last_gyro.x() << ", " << last_gyro.y() << ", " << last_gyro.z() << ") rad/s\n";
    }

    // Step 7: Save PLY if requested
    if (save_ply && !all_points.empty()) {
        std::lock_guard<std::mutex> lock(points_mutex);

        std::string filename = "livox_capture_" +
            std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) +
            ".ply";

        std::cout << "\nSaving " << all_points.size() << " points to " << filename << "...\n";

        if (saveToPly(filename, all_points)) {
            std::cout << "Saved successfully!\n";
        } else {
            std::cerr << "Failed to save PLY file!\n";
        }
    }

    std::cout << "\nTest complete!\n";
    return 0;
}
