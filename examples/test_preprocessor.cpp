/**
 * @file test_preprocessor.cpp
 * @brief Test the Livox SDK2 preprocessor
 *
 * Simulates Livox Mid-360 point data and tests preprocessing.
 */

#include <iostream>
#include <cmath>
#include <random>

#include "slam/types.hpp"
#include "slam/preprocess.hpp"
#include "slam/ply_export.hpp"

using namespace slam;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Preprocessor Test" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Create preprocessor with Mid-360 config
    PreprocessConfig config;
    config.lidar_type = LidarType::LIVOX_MID360;
    config.n_scans = 4;
    config.blind_distance = 0.5;
    config.max_distance = 50.0;
    config.point_filter_num = 1;  // Keep all points
    config.time_unit = TimeUnit::NS;

    Preprocessor preprocessor(config);

    // Generate simulated Livox SDK2 points (wall + floor)
    std::vector<LivoxPointXYZRTLT> raw_points;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise(-10.0f, 10.0f);  // mm noise
    std::uniform_int_distribution<int> refl_dist(50, 200);

    // Simulate a wall at x = 3m
    for (int y_idx = -100; y_idx <= 100; y_idx++) {
        for (int z_idx = 0; z_idx <= 50; z_idx++) {
            LivoxPointXYZRTLT pt;
            pt.x = 3000 + static_cast<int32_t>(noise(gen));  // 3m in mm
            pt.y = y_idx * 20 + static_cast<int32_t>(noise(gen));  // -2m to 2m
            pt.z = z_idx * 20 + static_cast<int32_t>(noise(gen));  // 0 to 1m
            pt.reflectivity = static_cast<uint8_t>(refl_dist(gen));
            pt.tag = 0x00;  // Valid single return
            pt.line = static_cast<uint8_t>(z_idx % 4);  // 4 scan lines
            pt.offset_time = static_cast<uint32_t>((y_idx + 100) * 1000);  // Nanoseconds

            raw_points.push_back(pt);
        }
    }

    // Simulate a floor at z = 0
    for (int x_idx = 10; x_idx <= 50; x_idx++) {
        for (int y_idx = -30; y_idx <= 30; y_idx++) {
            LivoxPointXYZRTLT pt;
            pt.x = x_idx * 100 + static_cast<int32_t>(noise(gen));  // 1-5m
            pt.y = y_idx * 100 + static_cast<int32_t>(noise(gen));  // -3m to 3m
            pt.z = static_cast<int32_t>(noise(gen));  // Near 0
            pt.reflectivity = static_cast<uint8_t>(refl_dist(gen) - 30);  // Floor is darker
            pt.tag = 0x00;
            pt.line = 0;
            pt.offset_time = static_cast<uint32_t>((x_idx * 100) + y_idx + 5000) * 1000;

            raw_points.push_back(pt);
        }
    }

    // Add some points in blind zone (should be filtered out)
    for (int i = 0; i < 100; i++) {
        LivoxPointXYZRTLT pt;
        pt.x = 200;   // 0.2m - in blind zone
        pt.y = static_cast<int32_t>(i * 10);
        pt.z = 100;
        pt.reflectivity = 100;
        pt.tag = 0x00;
        pt.line = 0;
        pt.offset_time = 0;
        raw_points.push_back(pt);
    }

    std::cout << "Generated " << raw_points.size() << " raw points" << std::endl;

    // Process points
    uint64_t scan_timestamp = 1000000000;  // 1 second in nanoseconds
    PointCloud processed;

    preprocessor.process(raw_points, scan_timestamp, processed);

    std::cout << "After preprocessing: " << processed.size() << " points" << std::endl;

    // Check intensity preservation
    float min_intensity = 999.0f, max_intensity = 0.0f;
    float sum_intensity = 0.0f;

    for (const auto& pt : processed.points) {
        min_intensity = std::min(min_intensity, pt.intensity);
        max_intensity = std::max(max_intensity, pt.intensity);
        sum_intensity += pt.intensity;
    }

    std::cout << "\nIntensity statistics:" << std::endl;
    std::cout << "  Min: " << min_intensity << std::endl;
    std::cout << "  Max: " << max_intensity << std::endl;
    std::cout << "  Avg: " << (sum_intensity / processed.size()) << std::endl;

    // Export to PLY
    PlyExportOptions opts;
    opts.format = PlyFormat::BINARY_LITTLE_ENDIAN;
    opts.include_intensity = true;

    if (exportToPly(processed, "test_preprocessed.ply", opts)) {
        std::cout << "\nExported: test_preprocessed.ply" << std::endl;
    }

    // Test IMU conversion
    LivoxImuPoint raw_imu;
    raw_imu.gyro_x = 0.01f;
    raw_imu.gyro_y = -0.02f;
    raw_imu.gyro_z = 0.005f;
    raw_imu.acc_x = 0.05f;
    raw_imu.acc_y = -0.03f;
    raw_imu.acc_z = -0.98f;  // Approximately -g in Z

    ImuData imu = Preprocessor::processImu(raw_imu, scan_timestamp);

    std::cout << "\nIMU conversion test:" << std::endl;
    std::cout << "  Gyro: [" << imu.gyro.transpose() << "] rad/s" << std::endl;
    std::cout << "  Acc:  [" << imu.acc.transpose() << "] m/s^2" << std::endl;
    std::cout << "  Acc magnitude: " << imu.acc.norm() << " m/s^2 (expected ~9.81)" << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test complete!" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}
