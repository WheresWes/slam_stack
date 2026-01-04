/**
 * @file test_real_icp.cpp
 * @brief Test ICP algorithms against real-world Stanford Bunny dataset
 *
 * Uses the Stanford Bunny scans to validate ICP alignment accuracy
 * by applying known transformations and checking recovery.
 */

#include <iostream>
#include <cmath>
#include <chrono>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "slam/types.hpp"
#include "slam/icp.hpp"
#include "slam/ply_reader.hpp"
#include "slam/so3_math.hpp"

using namespace slam;

//=============================================================================
// Transform Points
//=============================================================================

std::vector<V3D> transformPoints(const std::vector<V3D>& points, const M4D& T) {
    std::vector<V3D> result;
    result.reserve(points.size());
    M3D R = T.block<3,3>(0,0);
    V3D t = T.block<3,1>(0,3);
    for (const auto& p : points) {
        result.push_back(R * p + t);
    }
    return result;
}

//=============================================================================
// Add Noise to Points
//=============================================================================

std::vector<V3D> addNoise(const std::vector<V3D>& points, double sigma) {
    std::vector<V3D> result;
    result.reserve(points.size());
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<double> noise(0.0, sigma);
    for (const auto& p : points) {
        result.emplace_back(p.x() + noise(gen),
                           p.y() + noise(gen),
                           p.z() + noise(gen));
    }
    return result;
}

//=============================================================================
// Single Point Cloud ICP Test
//=============================================================================

bool runSingleCloudTest(const std::string& name,
                        const std::vector<V3D>& cloud,
                        const V3D& test_translation,
                        const V3D& test_rotation_euler,  // radians
                        ICPMethod method,
                        double noise_sigma = 0.0,
                        bool verbose = true) {

    std::cout << "\n--- " << name << " ("
              << (method == ICPMethod::POINT_TO_POINT ? "P2P" :
                  method == ICPMethod::POINT_TO_PLANE ? "P2L" : "GICP")
              << ") ---\n";

    // Downsample for faster processing
    double voxel_size = 0.003;  // 3mm voxel
    std::vector<V3D> target = voxelDownsample(cloud, voxel_size);

    // Create transformation
    M3D R_true = Exp(test_rotation_euler);
    V3D t_true = test_translation;
    M4D T_true = M4D::Identity();
    T_true.block<3,3>(0,0) = R_true;
    T_true.block<3,1>(0,3) = t_true;

    // Transform source (inverse transform to get from source to target)
    M4D T_inverse = T_true.inverse();
    std::vector<V3D> source = transformPoints(target, T_inverse);

    // Optionally add noise
    if (noise_sigma > 0) {
        source = addNoise(source, noise_sigma);
    }

    std::cout << "Points: " << target.size() << " (after voxel downsample)\n";
    std::cout << "True translation: [" << t_true.x() << ", " << t_true.y() << ", " << t_true.z() << "] m\n";
    std::cout << "True rotation: " << (test_rotation_euler.norm() * 180.0 / M_PI) << " deg\n";
    if (noise_sigma > 0) {
        std::cout << "Noise sigma: " << (noise_sigma * 1000) << " mm\n";
    }

    // Configure ICP
    ICPConfig config;
    config.method = method;
    config.max_iterations = 100;
    config.convergence_threshold = 1e-8;
    config.max_correspondence_dist = 0.02;  // 20mm max correspondence
    config.normal_estimation_k = 20;

    ICP icp(config);

    // Run ICP
    auto start = std::chrono::high_resolution_clock::now();
    ICPResult result = icp.align(source, target, M4D::Identity());
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Check accuracy - ICP should recover T_true
    V3D t_recovered = result.transformation.block<3,1>(0,3);
    M3D R_recovered = result.transformation.block<3,3>(0,0);

    double t_error = (t_recovered - t_true).norm();
    M3D R_diff = R_true.transpose() * R_recovered;
    double r_error = Log(R_diff).norm();

    if (verbose) {
        std::cout << "  Iterations: " << result.num_iterations << "\n";
        std::cout << "  Inliers: " << result.num_inliers << "/" << source.size() << "\n";
        std::cout << "  Fitness: " << result.fitness_score << "\n";
        std::cout << "  RMSE: " << result.rmse << "\n";
        std::cout << "  Time: " << elapsed_ms << " ms\n";
    }

    std::cout << "  Translation error: " << (t_error * 1000.0) << " mm\n";
    std::cout << "  Rotation error: " << (r_error * 180.0 / M_PI) << " deg\n";

    // Tolerances based on perturbation magnitude
    double t_tolerance = std::max(0.001, 0.1 * t_true.norm());  // 10% or 1mm min
    double r_tolerance = std::max(0.01, 0.1 * test_rotation_euler.norm());  // 10% or ~0.5deg min

    bool success = t_error < t_tolerance && r_error < r_tolerance;
    std::cout << "  Result: " << (success ? "PASSED" : "FAILED") << "\n";

    return success;
}

//=============================================================================
// Multi-Scale ICP Test
//=============================================================================

bool runMultiScaleTest(const std::string& name,
                       const std::vector<V3D>& cloud,
                       const V3D& test_translation,
                       const V3D& test_rotation_euler,
                       bool verbose = true) {

    std::cout << "\n--- " << name << " (Multi-Scale P2L) ---\n";

    // Light downsample for multi-scale test
    std::vector<V3D> target = voxelDownsample(cloud, 0.002);  // 2mm

    // Create transformation
    M3D R_true = Exp(test_rotation_euler);
    V3D t_true = test_translation;
    M4D T_true = M4D::Identity();
    T_true.block<3,3>(0,0) = R_true;
    T_true.block<3,1>(0,3) = t_true;

    // Transform source
    M4D T_inverse = T_true.inverse();
    std::vector<V3D> source = transformPoints(target, T_inverse);

    std::cout << "Points: " << target.size() << "\n";
    std::cout << "True translation: [" << t_true.x() << ", " << t_true.y() << ", " << t_true.z() << "] m\n";
    std::cout << "True rotation: " << (test_rotation_euler.norm() * 180.0 / M_PI) << " deg\n";

    // Run multi-scale ICP
    MultiScaleICP ms_icp;

    auto start = std::chrono::high_resolution_clock::now();
    ICPResult result = ms_icp.align(source, target, M4D::Identity(),
                                    {8.0, 4.0, 2.0, 1.0},  // Voxel scale factors
                                    ICPMethod::POINT_TO_PLANE);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Check accuracy
    V3D t_recovered = result.transformation.block<3,1>(0,3);
    M3D R_recovered = result.transformation.block<3,3>(0,0);

    double t_error = (t_recovered - t_true).norm();
    M3D R_diff = R_true.transpose() * R_recovered;
    double r_error = Log(R_diff).norm();

    if (verbose) {
        std::cout << "  Total iterations: " << result.num_iterations << "\n";
        std::cout << "  Final RMSE: " << result.rmse << "\n";
        std::cout << "  Time: " << elapsed_ms << " ms\n";
    }

    std::cout << "  Translation error: " << (t_error * 1000.0) << " mm\n";
    std::cout << "  Rotation error: " << (r_error * 180.0 / M_PI) << " deg\n";

    double t_tolerance = std::max(0.002, 0.15 * t_true.norm());  // 15% or 2mm
    double r_tolerance = std::max(0.02, 0.15 * test_rotation_euler.norm());  // 15% or ~1deg

    bool success = t_error < t_tolerance && r_error < r_tolerance;
    std::cout << "  Result: " << (success ? "PASSED" : "FAILED") << "\n";

    return success;
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    std::cout << "=== Real-World ICP Test (Stanford Bunny) ===\n";
    std::cout << "Testing ICP transformation recovery on real 3D scans\n";

    std::string data_dir = "test_data/bunny/data";
    if (argc > 1) {
        data_dir = argv[1];
    }

    // Load the reconstructed bunny (complete model)
    std::vector<V3D> bunny;
    std::string bunny_file = data_dir + "/bun000.ply";
    if (!loadFromPly(bunny_file, bunny)) {
        // Try reconstruction folder
        bunny_file = "test_data/bunny/reconstruction/bun_zipper_res2.ply";
        if (!loadFromPly(bunny_file, bunny)) {
            std::cerr << "Cannot find Stanford Bunny data.\n";
            std::cerr << "Usage: " << argv[0] << " <data_dir>\n";
            return 1;
        }
    }

    std::cout << "Loaded bunny with " << bunny.size() << " points\n";

    int passed = 0;
    int total = 0;

    //=========================================================================
    std::cout << "\n========================================\n";
    std::cout << "Part 1: Small Perturbation Tests\n";
    std::cout << "========================================\n";
    //=========================================================================

    // Test 1: Very small translation only
    if (runSingleCloudTest("Small Translation (2mm)",
                           bunny,
                           V3D(0.002, 0.001, -0.0015),
                           V3D(0, 0, 0),
                           ICPMethod::POINT_TO_POINT)) passed++;
    total++;

    // Test 2: Very small rotation only
    if (runSingleCloudTest("Small Rotation (1 deg)",
                           bunny,
                           V3D(0, 0, 0),
                           V3D(0.01, 0.008, 0.005),  // ~1 deg
                           ICPMethod::POINT_TO_POINT)) passed++;
    total++;

    // Test 3: Small combined transform - Point-to-Point
    if (runSingleCloudTest("Small Transform P2P",
                           bunny,
                           V3D(0.003, -0.002, 0.001),
                           V3D(0.015, -0.01, 0.008),  // ~1.5 deg
                           ICPMethod::POINT_TO_POINT)) passed++;
    total++;

    // Test 4: Small combined transform - Point-to-Plane
    if (runSingleCloudTest("Small Transform P2L",
                           bunny,
                           V3D(0.003, -0.002, 0.001),
                           V3D(0.015, -0.01, 0.008),  // ~1.5 deg
                           ICPMethod::POINT_TO_PLANE)) passed++;
    total++;

    //=========================================================================
    std::cout << "\n========================================\n";
    std::cout << "Part 2: Medium Perturbation Tests\n";
    std::cout << "========================================\n";
    //=========================================================================

    // Test 5: Medium translation
    if (runSingleCloudTest("Medium Translation (5mm) P2P",
                           bunny,
                           V3D(0.005, -0.003, 0.002),
                           V3D(0, 0, 0),
                           ICPMethod::POINT_TO_POINT)) passed++;
    total++;

    // Test 6: Medium rotation
    if (runSingleCloudTest("Medium Rotation (3 deg) P2L",
                           bunny,
                           V3D(0, 0, 0),
                           V3D(0.03, -0.02, 0.025),  // ~3 deg
                           ICPMethod::POINT_TO_PLANE)) passed++;
    total++;

    // Test 7: Medium combined transform
    if (runSingleCloudTest("Medium Transform P2L",
                           bunny,
                           V3D(0.005, -0.004, 0.003),
                           V3D(0.025, -0.02, 0.015),  // ~2.5 deg
                           ICPMethod::POINT_TO_PLANE)) passed++;
    total++;

    //=========================================================================
    std::cout << "\n========================================\n";
    std::cout << "Part 3: Larger Perturbation Tests\n";
    std::cout << "========================================\n";
    //=========================================================================

    // Test 8: Larger translation with Point-to-Plane
    if (runSingleCloudTest("Large Translation (10mm) P2L",
                           bunny,
                           V3D(0.01, -0.005, 0.003),
                           V3D(0, 0, 0),
                           ICPMethod::POINT_TO_PLANE)) passed++;
    total++;

    // Test 9: Larger rotation with Point-to-Plane
    if (runSingleCloudTest("Large Rotation (5 deg) P2L",
                           bunny,
                           V3D(0, 0, 0),
                           V3D(0.05, -0.04, 0.03),  // ~5 deg
                           ICPMethod::POINT_TO_PLANE)) passed++;
    total++;

    // Test 10: Larger combined transform
    if (runSingleCloudTest("Large Transform P2L",
                           bunny,
                           V3D(0.008, -0.006, 0.004),
                           V3D(0.04, -0.035, 0.025),  // ~4 deg
                           ICPMethod::POINT_TO_PLANE)) passed++;
    total++;

    //=========================================================================
    std::cout << "\n========================================\n";
    std::cout << "Part 4: Noise Robustness Tests\n";
    std::cout << "========================================\n";
    //=========================================================================

    // Test 11: Small transform with noise
    if (runSingleCloudTest("Transform with 0.5mm noise P2L",
                           bunny,
                           V3D(0.004, -0.003, 0.002),
                           V3D(0.02, -0.015, 0.01),
                           ICPMethod::POINT_TO_PLANE,
                           0.0005)) passed++;  // 0.5mm noise
    total++;

    // Test 12: Transform with more noise
    if (runSingleCloudTest("Transform with 1mm noise P2L",
                           bunny,
                           V3D(0.005, -0.004, 0.003),
                           V3D(0.025, -0.02, 0.015),
                           ICPMethod::POINT_TO_PLANE,
                           0.001)) passed++;  // 1mm noise
    total++;

    //=========================================================================
    std::cout << "\n========================================\n";
    std::cout << "Part 5: Multi-Scale ICP Tests\n";
    std::cout << "========================================\n";
    //=========================================================================

    // Test 13: Multi-scale with medium perturbation
    if (runMultiScaleTest("Multi-Scale Medium Transform",
                          bunny,
                          V3D(0.008, -0.005, 0.003),
                          V3D(0.04, -0.03, 0.02))) passed++;  // ~4 deg
    total++;

    // Test 14: Multi-scale with larger perturbation
    if (runMultiScaleTest("Multi-Scale Large Transform",
                          bunny,
                          V3D(0.015, -0.01, 0.008),
                          V3D(0.08, -0.06, 0.04))) passed++;  // ~8 deg
    total++;

    //=========================================================================
    // Summary
    //=========================================================================
    std::cout << "\n========================================\n";
    std::cout << "SUMMARY: " << passed << "/" << total << " tests passed";
    if (passed == total) {
        std::cout << " - ALL PASSED!\n";
    } else {
        std::cout << " (" << (100 * passed / total) << "%)\n";
    }
    std::cout << "========================================\n";

    return (passed >= total * 0.8) ? 0 : 1;  // Pass if 80% succeed
}
