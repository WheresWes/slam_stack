/**
 * @file test_ply_export.cpp
 * @brief Test PLY export with intensity
 *
 * Creates a sample point cloud and exports it to PLY format
 * with intensity values preserved.
 */

#include <iostream>
#include <cmath>
#include <random>

#include "slam/types.hpp"
#include "slam/ply_export.hpp"

using namespace slam;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  PLY Export Test" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Create a sample point cloud (sphere with varying intensity)
    PointCloud cloud;
    cloud.reserve(10000);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise(-0.05f, 0.05f);

    float radius = 5.0f;
    int points_per_ring = 100;
    int num_rings = 100;

    for (int ring = 0; ring < num_rings; ring++) {
        float phi = static_cast<float>(M_PI) * ring / (num_rings - 1);

        for (int i = 0; i < points_per_ring; i++) {
            float theta = 2.0f * static_cast<float>(M_PI) * i / points_per_ring;

            LidarPoint pt;
            pt.x = radius * std::sin(phi) * std::cos(theta) + noise(gen);
            pt.y = radius * std::sin(phi) * std::sin(theta) + noise(gen);
            pt.z = radius * std::cos(phi) + noise(gen);

            // Intensity varies with position (for visualization)
            pt.intensity = 128.0f + 127.0f * std::sin(2.0f * phi) * std::cos(3.0f * theta);

            cloud.push_back(pt);
        }
    }

    std::cout << "Generated " << cloud.size() << " points" << std::endl;

    // Export as ASCII PLY with intensity
    PlyExportOptions ascii_opts;
    ascii_opts.format = PlyFormat::ASCII;
    ascii_opts.include_intensity = true;
    ascii_opts.precision = 4;

    if (exportToPly(cloud, "test_sphere_ascii.ply", ascii_opts)) {
        std::cout << "Exported: test_sphere_ascii.ply (ASCII with intensity)" << std::endl;
    }

    // Export as binary PLY with intensity
    PlyExportOptions binary_opts;
    binary_opts.format = PlyFormat::BINARY_LITTLE_ENDIAN;
    binary_opts.include_intensity = true;

    if (exportToPly(cloud, "test_sphere_binary.ply", binary_opts)) {
        std::cout << "Exported: test_sphere_binary.ply (binary with intensity)" << std::endl;
    }

    // Export with RGB (intensity as grayscale)
    PlyExportOptions rgb_opts;
    rgb_opts.format = PlyFormat::BINARY_LITTLE_ENDIAN;
    rgb_opts.include_intensity = true;
    rgb_opts.include_rgb = true;

    if (exportToPly(cloud, "test_sphere_rgb.ply", rgb_opts)) {
        std::cout << "Exported: test_sphere_rgb.ply (binary with intensity + RGB)" << std::endl;
    }

    // Create and export a trajectory
    std::vector<M4D> trajectory;
    for (int i = 0; i < 100; i++) {
        float t = static_cast<float>(i) / 99.0f;
        M4D pose = M4D::Identity();
        pose(0, 3) = 10.0 * t;                        // X moves forward
        pose(1, 3) = 2.0 * std::sin(4.0 * M_PI * t);  // Y oscillates
        pose(2, 3) = 0.5;                             // Z constant
        trajectory.push_back(pose);
    }

    if (exportTrajectoryToPly(trajectory, "test_trajectory.ply")) {
        std::cout << "Exported: test_trajectory.ply (trajectory)" << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Export complete!" << std::endl;
    std::cout << "  Open PLY files in CloudCompare to verify" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}
