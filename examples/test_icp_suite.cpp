/**
 * @file test_icp_suite.cpp
 * @brief Extensive ICP test suite
 *
 * Tests all ICP variants and utilities:
 * - Point-to-Point ICP
 * - Point-to-Plane ICP
 * - Generalized ICP (GICP)
 * - Multi-scale ICP
 * - FOV cropping
 * - Normal estimation
 * - Voxel downsampling
 * - Noise robustness
 * - Various initial pose errors
 * - Different scene geometries
 */

#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "slam/types.hpp"
#include "slam/icp.hpp"
#include "slam/so3_math.hpp"

using namespace slam;

//=============================================================================
// Test Utilities
//=============================================================================

struct TestResult {
    std::string name;
    bool passed;
    double translation_error;
    double rotation_error_deg;
    double rmse;
    int iterations;
    double time_ms;
};

std::vector<TestResult> g_results;

void printTestHeader(const std::string& name) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << name << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void printResult(const TestResult& r) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << (r.passed ? "[PASS]" : "[FAIL]") << " " << r.name << std::endl;
    std::cout << "  Trans err: " << r.translation_error << " m, "
              << "Rot err: " << r.rotation_error_deg << " deg, "
              << "RMSE: " << r.rmse << ", "
              << "Iters: " << r.iterations << ", "
              << "Time: " << r.time_ms << " ms" << std::endl;
}

double computeTranslationError(const M4D& estimated, const M4D& gt) {
    V3D t_est = estimated.block<3, 1>(0, 3);
    V3D t_gt = gt.block<3, 1>(0, 3);
    return (t_est - t_gt).norm();
}

double computeRotationError(const M4D& estimated, const M4D& gt) {
    M3D R_est = estimated.block<3, 3>(0, 0);
    M3D R_gt = gt.block<3, 3>(0, 0);
    M3D R_diff = R_gt.transpose() * R_est;
    V3D axis_angle = Log(R_diff);
    return axis_angle.norm() * 180.0 / M_PI;  // degrees
}

//=============================================================================
// Scene Generators
//=============================================================================

// Generate a room (floor + 4 walls + ceiling)
std::vector<V3D> generateRoom(double length, double width, double height,
                               double point_spacing) {
    std::vector<V3D> points;

    // Floor
    for (double x = 0; x < length; x += point_spacing) {
        for (double y = 0; y < width; y += point_spacing) {
            points.emplace_back(x, y, 0);
        }
    }

    // Ceiling
    for (double x = 0; x < length; x += point_spacing) {
        for (double y = 0; y < width; y += point_spacing) {
            points.emplace_back(x, y, height);
        }
    }

    // Walls
    for (double x = 0; x < length; x += point_spacing) {
        for (double z = 0; z < height; z += point_spacing) {
            points.emplace_back(x, 0, z);      // Front wall
            points.emplace_back(x, width, z);  // Back wall
        }
    }
    for (double y = 0; y < width; y += point_spacing) {
        for (double z = 0; z < height; z += point_spacing) {
            points.emplace_back(0, y, z);       // Left wall
            points.emplace_back(length, y, z);  // Right wall
        }
    }

    return points;
}

// Generate a corridor (long room)
std::vector<V3D> generateCorridor(double length, double width, double height,
                                   double point_spacing) {
    return generateRoom(length, width, height, point_spacing);
}

// Generate a sphere (for testing curved surfaces)
std::vector<V3D> generateSphere(double radius, int num_points) {
    std::vector<V3D> points;
    points.reserve(num_points);

    // Fibonacci sphere distribution
    double phi = M_PI * (3.0 - std::sqrt(5.0));  // Golden angle

    for (int i = 0; i < num_points; ++i) {
        double y = 1 - (i / static_cast<double>(num_points - 1)) * 2;
        double r = std::sqrt(1 - y * y);
        double theta = phi * i;

        double x = std::cos(theta) * r;
        double z = std::sin(theta) * r;

        points.emplace_back(x * radius, y * radius, z * radius);
    }

    return points;
}

// Generate random point cloud (for stress testing)
std::vector<V3D> generateRandom(int num_points, double range,
                                 std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(-range, range);
    std::vector<V3D> points;
    points.reserve(num_points);

    for (int i = 0; i < num_points; ++i) {
        points.emplace_back(dist(gen), dist(gen), dist(gen));
    }

    return points;
}

// Transform point cloud
std::vector<V3D> transformPoints(const std::vector<V3D>& points,
                                  const M4D& transform) {
    std::vector<V3D> result;
    result.reserve(points.size());

    M3D R = transform.block<3, 3>(0, 0);
    V3D t = transform.block<3, 1>(0, 3);

    for (const auto& pt : points) {
        result.push_back(R * pt + t);
    }

    return result;
}

// Add noise to point cloud
std::vector<V3D> addNoise(const std::vector<V3D>& points, double noise_std,
                           std::mt19937& gen) {
    std::normal_distribution<double> dist(0, noise_std);
    std::vector<V3D> result;
    result.reserve(points.size());

    for (const auto& pt : points) {
        result.emplace_back(pt.x() + dist(gen),
                           pt.y() + dist(gen),
                           pt.z() + dist(gen));
    }

    return result;
}

// Create transformation from translation and euler angles (deg)
M4D createTransform(double tx, double ty, double tz,
                    double rx_deg, double ry_deg, double rz_deg) {
    V3D euler_rad(rx_deg * M_PI / 180.0,
                  ry_deg * M_PI / 180.0,
                  rz_deg * M_PI / 180.0);
    M3D R = Exp(euler_rad);

    M4D T = M4D::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = V3D(tx, ty, tz);
    return T;
}

//=============================================================================
// Test Cases
//=============================================================================

TestResult testBasicAlignment(const std::string& name,
                               const std::vector<V3D>& source,
                               const M4D& gt_transform,
                               ICPMethod method,
                               double noise_std = 0.0) {
    TestResult result;
    result.name = name;

    // Create target by transforming source
    std::vector<V3D> target = transformPoints(source, gt_transform);

    // Add noise if specified
    std::mt19937 gen(42);
    if (noise_std > 0) {
        target = addNoise(target, noise_std, gen);
    }

    // Configure ICP
    ICPConfig config;
    config.method = method;
    config.max_iterations = 100;
    config.convergence_threshold = 1e-8;
    config.max_correspondence_dist = 1.0;

    ICP icp(config);

    // Run ICP
    auto start = std::chrono::high_resolution_clock::now();
    ICPResult icp_result = icp.align(source, target, M4D::Identity());
    auto end = std::chrono::high_resolution_clock::now();

    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.translation_error = computeTranslationError(icp_result.transformation, gt_transform);
    result.rotation_error_deg = computeRotationError(icp_result.transformation, gt_transform);
    result.rmse = icp_result.rmse;
    result.iterations = icp_result.num_iterations;

    // Pass criteria
    double trans_threshold = (noise_std > 0) ? 0.1 : 0.01;
    double rot_threshold = (noise_std > 0) ? 1.0 : 0.1;
    result.passed = (result.translation_error < trans_threshold) &&
                    (result.rotation_error_deg < rot_threshold);

    return result;
}

TestResult testWithInitialGuess(const std::string& name,
                                 const std::vector<V3D>& source,
                                 const M4D& gt_transform,
                                 const M4D& initial_guess,
                                 ICPMethod method) {
    TestResult result;
    result.name = name;

    std::vector<V3D> target = transformPoints(source, gt_transform);

    ICPConfig config;
    config.method = method;
    config.max_iterations = 100;
    config.convergence_threshold = 1e-8;
    config.max_correspondence_dist = 2.0;

    ICP icp(config);

    auto start = std::chrono::high_resolution_clock::now();
    ICPResult icp_result = icp.align(source, target, initial_guess);
    auto end = std::chrono::high_resolution_clock::now();

    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.translation_error = computeTranslationError(icp_result.transformation, gt_transform);
    result.rotation_error_deg = computeRotationError(icp_result.transformation, gt_transform);
    result.rmse = icp_result.rmse;
    result.iterations = icp_result.num_iterations;

    result.passed = (result.translation_error < 0.1) && (result.rotation_error_deg < 1.0);

    return result;
}

TestResult testMultiScale(const std::string& name,
                           const std::vector<V3D>& source,
                           const M4D& gt_transform,
                           const M4D& initial_guess,
                           ICPMethod method) {
    TestResult result;
    result.name = name;

    std::vector<V3D> target = transformPoints(source, gt_transform);

    MultiScaleICP ms_icp;

    auto start = std::chrono::high_resolution_clock::now();
    ICPResult icp_result = ms_icp.align(source, target, initial_guess,
                                         {2.0, 1.0, 0.5, 0.25}, method);
    auto end = std::chrono::high_resolution_clock::now();

    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.translation_error = computeTranslationError(icp_result.transformation, gt_transform);
    result.rotation_error_deg = computeRotationError(icp_result.transformation, gt_transform);
    result.rmse = icp_result.rmse;
    result.iterations = icp_result.num_iterations;

    result.passed = (result.translation_error < 0.15) && (result.rotation_error_deg < 2.0);

    return result;
}

//=============================================================================
// Test Suites
//=============================================================================

void runPointToPointTests() {
    printTestHeader("Point-to-Point ICP Tests");

    auto room = generateRoom(5, 5, 3, 0.2);
    std::cout << "Scene: Room with " << room.size() << " points\n" << std::endl;

    // Test 1: Pure translation
    {
        M4D T = createTransform(0.3, 0.2, 0.1, 0, 0, 0);
        auto r = testBasicAlignment("Pure translation (0.3, 0.2, 0.1)m",
                                     room, T, ICPMethod::POINT_TO_POINT);
        printResult(r);
        g_results.push_back(r);
    }

    // Test 2: Pure rotation
    {
        M4D T = createTransform(0, 0, 0, 5, 3, 8);
        auto r = testBasicAlignment("Pure rotation (5, 3, 8) deg",
                                     room, T, ICPMethod::POINT_TO_POINT);
        printResult(r);
        g_results.push_back(r);
    }

    // Test 3: Combined transformation
    {
        M4D T = createTransform(0.2, 0.15, 0.05, 3, 2, 5);
        auto r = testBasicAlignment("Combined T(0.2,0.15,0.05) R(3,2,5)",
                                     room, T, ICPMethod::POINT_TO_POINT);
        printResult(r);
        g_results.push_back(r);
    }

    // Test 4: With noise
    {
        M4D T = createTransform(0.2, 0.1, 0, 0, 0, 5);
        auto r = testBasicAlignment("With 1cm noise",
                                     room, T, ICPMethod::POINT_TO_POINT, 0.01);
        printResult(r);
        g_results.push_back(r);
    }

    // Test 5: With poor initial guess
    {
        M4D T_gt = createTransform(0.3, 0.2, 0, 0, 0, 10);
        M4D T_init = createTransform(0.5, 0.4, 0.1, 5, 5, 15);
        auto r = testWithInitialGuess("Poor initial guess (0.2m, 5deg error)",
                                       room, T_gt, T_init, ICPMethod::POINT_TO_POINT);
        printResult(r);
        g_results.push_back(r);
    }
}

void runPointToPlaneTests() {
    printTestHeader("Point-to-Plane ICP Tests");

    auto room = generateRoom(5, 5, 3, 0.3);  // Coarser for speed
    std::cout << "Scene: Room with " << room.size() << " points\n" << std::endl;

    // Test 1: Pure translation
    {
        M4D T = createTransform(0.3, 0.2, 0.1, 0, 0, 0);
        auto r = testBasicAlignment("Pure translation (0.3, 0.2, 0.1)m",
                                     room, T, ICPMethod::POINT_TO_PLANE);
        printResult(r);
        g_results.push_back(r);
    }

    // Test 2: Pure rotation
    {
        M4D T = createTransform(0, 0, 0, 5, 3, 8);
        auto r = testBasicAlignment("Pure rotation (5, 3, 8) deg",
                                     room, T, ICPMethod::POINT_TO_PLANE);
        printResult(r);
        g_results.push_back(r);
    }

    // Test 3: Combined
    {
        M4D T = createTransform(0.2, 0.15, 0.05, 3, 2, 5);
        auto r = testBasicAlignment("Combined T(0.2,0.15,0.05) R(3,2,5)",
                                     room, T, ICPMethod::POINT_TO_PLANE);
        printResult(r);
        g_results.push_back(r);
    }

    // Test 4: Corridor (planar surfaces)
    {
        auto corridor = generateCorridor(20, 3, 3, 0.3);
        std::cout << "\nCorridor scene: " << corridor.size() << " points" << std::endl;

        M4D T = createTransform(0.5, 0.1, 0, 0, 0, 3);
        auto r = testBasicAlignment("Corridor - translation along axis",
                                     corridor, T, ICPMethod::POINT_TO_PLANE);
        printResult(r);
        g_results.push_back(r);
    }
}

void runGICPTests() {
    printTestHeader("Generalized ICP (GICP) Tests");

    auto room = generateRoom(5, 5, 3, 0.4);  // Coarser for speed
    std::cout << "Scene: Room with " << room.size() << " points\n" << std::endl;

    // Test 1: Basic alignment
    {
        M4D T = createTransform(0.3, 0.2, 0.1, 3, 2, 5);
        auto r = testBasicAlignment("Combined transformation",
                                     room, T, ICPMethod::GICP);
        printResult(r);
        g_results.push_back(r);
    }

    // Test 2: Curved surface (sphere)
    {
        auto sphere = generateSphere(2.0, 500);
        std::cout << "\nSphere scene: " << sphere.size() << " points" << std::endl;

        M4D T = createTransform(0.2, 0.15, 0.1, 5, 3, 8);
        auto r = testBasicAlignment("Sphere - curved surface",
                                     sphere, T, ICPMethod::GICP);
        printResult(r);
        g_results.push_back(r);
    }

    // Test 3: With noise
    {
        M4D T = createTransform(0.2, 0.1, 0.05, 2, 2, 3);
        auto r = testBasicAlignment("With 2cm noise",
                                     room, T, ICPMethod::GICP, 0.02);
        printResult(r);
        g_results.push_back(r);
    }
}

void runMultiScaleTests() {
    printTestHeader("Multi-Scale ICP Tests");

    auto corridor = generateCorridor(20, 4, 3, 0.2);
    std::cout << "Scene: Corridor with " << corridor.size() << " points\n" << std::endl;

    // Test 1: Large initial error with multi-scale
    {
        M4D T_gt = createTransform(1.0, 0.5, 0, 0, 0, 15);
        M4D T_init = createTransform(2.0, 1.0, 0.3, 5, 5, 25);

        auto r = testMultiScale("Large error (1m, 10deg) - Point-to-Point",
                                 corridor, T_gt, T_init, ICPMethod::POINT_TO_POINT);
        printResult(r);
        g_results.push_back(r);
    }

    // Test 2: Multi-scale Point-to-Plane
    {
        auto room = generateRoom(5, 5, 3, 0.3);
        M4D T_gt = createTransform(0.5, 0.3, 0.1, 5, 3, 10);
        M4D T_init = createTransform(0.8, 0.6, 0.2, 10, 8, 18);

        auto r = testMultiScale("Medium error - Point-to-Plane",
                                 room, T_gt, T_init, ICPMethod::POINT_TO_PLANE);
        printResult(r);
        g_results.push_back(r);
    }
}

void runFOVCroppingTests() {
    printTestHeader("FOV Cropping Tests");

    // Create a large map
    auto map = generateRoom(50, 50, 5, 0.5);
    std::cout << "Full map: " << map.size() << " points" << std::endl;

    // Define sensor pose in center of room
    M4D sensor_pose = createTransform(25, 25, 1.5, 0, 0, 0);

    // Test 1: Full 360 FOV
    {
        FOVConfig fov_config;
        fov_config.fov_horizontal = 2.0 * M_PI;
        fov_config.max_range = 20.0;

        auto cropped = cropToFOV(map, sensor_pose, fov_config);
        std::cout << "\n360 FOV, 20m range: " << cropped.size() << " points ("
                  << (100.0 * cropped.size() / map.size()) << "%)" << std::endl;
    }

    // Test 2: Limited FOV (like Livox)
    {
        FOVConfig fov_config;
        fov_config.fov_horizontal = 70.0 * M_PI / 180.0;  // 70 deg
        fov_config.fov_vertical = 77.0 * M_PI / 180.0;    // 77 deg
        fov_config.max_range = 15.0;

        auto cropped = cropToFOV(map, sensor_pose, fov_config);
        std::cout << "Livox-like FOV (70x77 deg, 15m): " << cropped.size() << " points ("
                  << (100.0 * cropped.size() / map.size()) << "%)" << std::endl;
    }

    // Test 3: Use cropped map for ICP
    {
        FOVConfig fov_config;
        fov_config.max_range = 10.0;

        auto local_map = cropToFOV(map, sensor_pose, fov_config);

        // Create a scan (subset of local map with small transform)
        M4D T_gt = createTransform(0.2, 0.1, 0, 0, 0, 3);
        auto scan = transformPoints(local_map, T_gt);

        // Add some noise
        std::mt19937 gen(42);
        scan = addNoise(scan, 0.01, gen);

        // Subsample scan
        scan = voxelDownsample(scan, 0.3);
        std::cout << "\nScan: " << scan.size() << " points" << std::endl;

        // Run ICP
        ICPConfig config;
        config.method = ICPMethod::POINT_TO_POINT;
        config.max_iterations = 50;

        ICP icp(config);
        auto result = icp.align(scan, local_map, M4D::Identity());

        double t_err = computeTranslationError(result.transformation, T_gt);
        double r_err = computeRotationError(result.transformation, T_gt);

        std::cout << "ICP on FOV-cropped map:" << std::endl;
        std::cout << "  Trans error: " << t_err << " m" << std::endl;
        std::cout << "  Rot error: " << r_err << " deg" << std::endl;
        std::cout << "  RMSE: " << result.rmse << std::endl;
        std::cout << "  " << (t_err < 0.05 && r_err < 0.5 ? "[PASS]" : "[FAIL]") << std::endl;

        TestResult tr;
        tr.name = "FOV-cropped ICP";
        tr.translation_error = t_err;
        tr.rotation_error_deg = r_err;
        tr.rmse = result.rmse;
        tr.iterations = result.num_iterations;
        tr.time_ms = 0;
        tr.passed = (t_err < 0.05 && r_err < 0.5);
        g_results.push_back(tr);
    }
}

void runVoxelDownsampleTests() {
    printTestHeader("Voxel Downsampling Tests");

    auto points = generateRoom(10, 10, 3, 0.1);
    std::cout << "Original: " << points.size() << " points" << std::endl;

    // Test different voxel sizes
    std::vector<double> voxel_sizes = {0.1, 0.2, 0.5, 1.0};

    for (double vs : voxel_sizes) {
        auto downsampled = voxelDownsample(points, vs);
        double reduction = 100.0 * (1.0 - static_cast<double>(downsampled.size()) / points.size());
        std::cout << "Voxel " << vs << "m: " << downsampled.size()
                  << " points (" << std::fixed << std::setprecision(1)
                  << reduction << "% reduction)" << std::endl;
    }
}

void runNormalEstimationTests() {
    printTestHeader("Normal Estimation Tests");

    // Test on a flat plane (normals should be ~(0,0,1))
    {
        std::vector<V3D> plane;
        for (double x = 0; x < 5; x += 0.2) {
            for (double y = 0; y < 5; y += 0.2) {
                plane.emplace_back(x, y, 0);
            }
        }

        auto with_normals = estimateNormals(plane, 10);

        // Check average normal
        V3D avg_normal = V3D::Zero();
        for (const auto& pwn : with_normals) {
            avg_normal += pwn.normal;
        }
        avg_normal /= static_cast<double>(with_normals.size());
        avg_normal.normalize();

        double dot_with_z = std::abs(avg_normal.dot(V3D::UnitZ()));
        std::cout << "Flat plane avg normal dot Z: " << dot_with_z
                  << " " << (dot_with_z > 0.99 ? "[PASS]" : "[FAIL]") << std::endl;
    }

    // Test on a vertical wall
    {
        std::vector<V3D> wall;
        for (double x = 0; x < 5; x += 0.2) {
            for (double z = 0; z < 3; z += 0.2) {
                wall.emplace_back(x, 0, z);
            }
        }

        auto with_normals = estimateNormals(wall, 10);

        V3D avg_normal = V3D::Zero();
        for (const auto& pwn : with_normals) {
            avg_normal += pwn.normal;
        }
        avg_normal /= static_cast<double>(with_normals.size());

        double dot_with_y = std::abs(avg_normal.dot(V3D::UnitY()));
        std::cout << "Vertical wall avg normal dot Y: " << dot_with_y
                  << " " << (dot_with_y > 0.95 ? "[PASS]" : "[FAIL]") << std::endl;
    }
}

void runStressTests() {
    printTestHeader("Stress Tests");

    std::mt19937 gen(12345);

    // Test with many points
    {
        auto large_scene = generateRoom(20, 20, 5, 0.15);
        std::cout << "Large scene: " << large_scene.size() << " points" << std::endl;

        M4D T = createTransform(0.3, 0.2, 0.1, 3, 2, 5);
        auto target = transformPoints(large_scene, T);

        ICPConfig config;
        config.method = ICPMethod::POINT_TO_POINT;
        config.max_iterations = 50;

        ICP icp(config);

        auto start = std::chrono::high_resolution_clock::now();
        auto result = icp.align(large_scene, target, M4D::Identity());
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double t_err = computeTranslationError(result.transformation, T);
        double r_err = computeRotationError(result.transformation, T);

        std::cout << "Large scene ICP: " << time_ms << " ms, "
                  << "t_err=" << t_err << "m, r_err=" << r_err << "deg"
                  << " " << (t_err < 0.05 ? "[PASS]" : "[FAIL]") << std::endl;

        TestResult tr;
        tr.name = "Large scene stress test";
        tr.translation_error = t_err;
        tr.rotation_error_deg = r_err;
        tr.rmse = result.rmse;
        tr.iterations = result.num_iterations;
        tr.time_ms = time_ms;
        tr.passed = t_err < 0.05;
        g_results.push_back(tr);
    }

    // Test convergence with challenging noise
    {
        auto scene = generateRoom(5, 5, 3, 0.2);
        M4D T = createTransform(0.2, 0.1, 0.05, 2, 2, 3);
        auto target = transformPoints(scene, T);
        target = addNoise(target, 0.03, gen);  // 3cm noise

        ICPConfig config;
        config.method = ICPMethod::POINT_TO_POINT;
        config.max_iterations = 100;

        ICP icp(config);
        auto result = icp.align(scene, target, M4D::Identity());

        double t_err = computeTranslationError(result.transformation, T);
        double r_err = computeRotationError(result.transformation, T);

        std::cout << "High noise (3cm) test: t_err=" << t_err
                  << "m, r_err=" << r_err << "deg"
                  << " " << (t_err < 0.15 ? "[PASS]" : "[FAIL]") << std::endl;

        TestResult tr;
        tr.name = "High noise stress test";
        tr.translation_error = t_err;
        tr.rotation_error_deg = r_err;
        tr.rmse = result.rmse;
        tr.iterations = result.num_iterations;
        tr.time_ms = 0;
        tr.passed = t_err < 0.15;
        g_results.push_back(tr);
    }
}

//=============================================================================
// Main
//=============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "     ICP Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    // Run all test suites
    runPointToPointTests();
    runPointToPlaneTests();
    runGICPTests();
    runMultiScaleTests();
    runFOVCroppingTests();
    runVoxelDownsampleTests();
    runNormalEstimationTests();
    runStressTests();

    // Summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    int passed = 0, failed = 0;
    for (const auto& r : g_results) {
        if (r.passed) passed++;
        else failed++;
    }

    std::cout << "\nTotal tests: " << g_results.size() << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;

    if (failed > 0) {
        std::cout << "\nFailed tests:" << std::endl;
        for (const auto& r : g_results) {
            if (!r.passed) {
                std::cout << "  - " << r.name << std::endl;
            }
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << (failed == 0 ? "  ALL TESTS PASSED!" : "  SOME TESTS FAILED") << std::endl;
    std::cout << "========================================" << std::endl;

    return (failed == 0) ? 0 : 1;
}
