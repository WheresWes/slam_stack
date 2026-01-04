/**
 * @file test_localization.cpp
 * @brief Test ICP algorithm for localization
 *
 * This test demonstrates the native ICP implementation for
 * point cloud alignment, which is the core of the localization system.
 */

#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "slam/types.hpp"
#include "slam/icp.hpp"
#include "slam/so3_math.hpp"

using namespace slam;

//=============================================================================
// Main
//=============================================================================

int main() {
    std::cout << "=== ICP Algorithm Test ===" << std::endl;
    std::cout << "(Testing point-to-point ICP alignment)" << std::endl;
    std::cout.flush();

    // Create simple source and target point clouds
    std::vector<V3D> source, target;

    // Generate a 3D box structure (floor + 4 walls) for unambiguous ICP
    // Floor
    for (double x = 0; x < 5; x += 0.25) {
        for (double y = 0; y < 5; y += 0.25) {
            source.emplace_back(x, y, 0);
        }
    }
    // Left wall (y=0)
    for (double x = 0; x < 5; x += 0.25) {
        for (double z = 0; z < 2; z += 0.25) {
            source.emplace_back(x, 0, z);
        }
    }
    // Right wall (y=5)
    for (double x = 0; x < 5; x += 0.25) {
        for (double z = 0; z < 2; z += 0.25) {
            source.emplace_back(x, 5, z);
        }
    }
    // Back wall (x=0)
    for (double y = 0; y < 5; y += 0.25) {
        for (double z = 0; z < 2; z += 0.25) {
            source.emplace_back(0, y, z);
        }
    }
    // Front wall (x=5)
    for (double y = 0; y < 5; y += 0.25) {
        for (double z = 0; z < 2; z += 0.25) {
            source.emplace_back(5, y, z);
        }
    }
    std::cout << "Source points: " << source.size() << std::endl;

    // Target is source translated and rotated slightly
    M3D R = Exp(V3D(0, 0, 0.1));  // 0.1 rad around Z
    V3D t(0.3, 0.2, 0.0);

    for (const auto& pt : source) {
        target.push_back(R * pt + t);
    }
    std::cout << "Target points: " << target.size() << std::endl;

    // Test ICP
    std::cout << "\nRunning ICP..." << std::endl;
    ICPConfig config;
    config.max_iterations = 100;
    config.convergence_threshold = 1e-8;
    config.max_correspondence_dist = 0.5;

    ICP icp(config);
    ICPResult result = icp.align(source, target, M4D::Identity());

    std::cout << "  Iterations: " << result.num_iterations << std::endl;
    std::cout << "  Inliers: " << result.num_inliers << "/" << source.size() << std::endl;
    std::cout << "  Fitness: " << result.fitness_score << std::endl;
    std::cout << "  RMSE: " << result.rmse << std::endl;
    std::cout << "  Converged: " << (result.converged ? "YES" : "NO") << std::endl;

    // Check result
    V3D t_est = result.transformation.block<3, 1>(0, 3);
    M3D R_est = result.transformation.block<3, 3>(0, 0);

    std::cout << "\nExpected translation: [" << t.x() << ", " << t.y() << ", " << t.z() << "]" << std::endl;
    std::cout << "Estimated translation: [" << t_est.x() << ", " << t_est.y() << ", " << t_est.z() << "]" << std::endl;

    double t_error = (t_est - t).norm();
    M3D rot_diff = R.transpose() * R_est;
    V3D rot_error = Log(rot_diff);
    double r_error = rot_error.norm();

    std::cout << "\nTranslation error: " << t_error << " m" << std::endl;
    std::cout << "Rotation error: " << r_error * 180.0 / M_PI << " deg" << std::endl;

    bool success = t_error < 0.1 && r_error < 0.05;
    std::cout << "\n=== ICP Test " << (success ? "PASSED" : "FAILED") << " ===" << std::endl;

    return success ? 0 : 1;
}
