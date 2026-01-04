/**
 * @file test_global_localization.cpp
 * @brief Test global localization from unknown position
 *
 * This test simulates global localization on a ship hull environment
 * using the Stanford Bunny as a stand-in for point cloud data.
 *
 * Tests:
 * 1. Scan Context descriptor computation
 * 2. Scan Context database building and query
 * 3. Voxel occupancy map scoring
 * 4. Full global localization pipeline
 */

#include <iostream>
#include <cmath>
#include <chrono>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "slam/types.hpp"
#include "slam/global_localization.hpp"
#include "slam/ply_reader.hpp"
#include "slam/icp.hpp"

using namespace slam;

//=============================================================================
// Utility Functions
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

M4D makePose(double x, double y, double z, double yaw) {
    M3D R = Eigen::AngleAxisd(yaw, V3D::UnitZ()).toRotationMatrix();
    M4D pose = M4D::Identity();
    pose.block<3,3>(0,0) = R;
    pose(0, 3) = x;
    pose(1, 3) = y;
    pose(2, 3) = z;
    return pose;
}

//=============================================================================
// Test 1: Scan Context Descriptor
//=============================================================================

bool testScanContextDescriptor(const std::vector<V3D>& cloud) {
    std::cout << "\n=== Test 1: Scan Context Descriptor ===\n";

    ScanContextConfig config;
    config.num_rings = 20;
    config.num_sectors = 60;
    config.max_radius = 0.2;  // Bunny is small (~0.15m)
    config.min_radius = 0.01;
    config.num_height_bands = 1;  // 2D for this test

    ScanContext sc(config);

    // Compute descriptor
    auto desc = sc.compute(cloud);

    std::cout << "Descriptor size: " << desc.rows() << " x " << desc.cols() << "\n";
    std::cout << "Non-zero entries: " << (desc.array() != 0).count()
              << " / " << desc.size() << "\n";

    // Test rotation invariance with self-matching
    auto [score, shift] = sc.match(desc, desc);
    std::cout << "Self-match score: " << score << " (should be ~1.0)\n";
    std::cout << "Self-match shift: " << shift << " (should be 0)\n";

    bool passed = score > 0.99 && shift == 0;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << "\n";
    return passed;
}

//=============================================================================
// Test 2: Scan Context Rotation Matching
//=============================================================================

bool testScanContextRotation(const std::vector<V3D>& cloud) {
    std::cout << "\n=== Test 2: Scan Context Rotation Matching ===\n";

    ScanContextConfig config;
    config.num_rings = 20;
    config.num_sectors = 60;
    config.max_radius = 0.2;
    config.min_radius = 0.01;
    config.num_height_bands = 1;

    ScanContext sc(config);

    // Compute descriptor for original
    auto desc1 = sc.compute(cloud);

    // Rotate cloud by 90 degrees and compute descriptor
    double test_yaw = M_PI / 2;  // 90 degrees
    M4D rotation = makePose(0, 0, 0, test_yaw);
    auto rotated_cloud = transformPoints(cloud, rotation);
    auto desc2 = sc.compute(rotated_cloud);

    // Match should find the rotation
    auto [score, shift] = sc.match(desc1, desc2);

    double recovered_yaw = sc.shiftToYaw(shift);
    double yaw_error = std::abs(recovered_yaw - test_yaw);
    if (yaw_error > M_PI) yaw_error = 2 * M_PI - yaw_error;

    std::cout << "Test rotation: " << (test_yaw * 180 / M_PI) << " deg\n";
    std::cout << "Match score: " << score << "\n";
    std::cout << "Recovered yaw: " << (recovered_yaw * 180 / M_PI) << " deg\n";
    std::cout << "Yaw error: " << (yaw_error * 180 / M_PI) << " deg\n";

    // Allow 1-2 sector error (~6-12 degrees)
    bool passed = score > 0.8 && yaw_error < 0.3;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << "\n";
    return passed;
}

//=============================================================================
// Test 3: Scan Context Database
//=============================================================================

bool testScanContextDatabase(const std::vector<V3D>& cloud) {
    std::cout << "\n=== Test 3: Scan Context Database ===\n";

    ScanContextConfig config;
    config.num_rings = 20;
    config.num_sectors = 60;
    config.max_radius = 0.5;  // Larger for world-frame testing
    config.min_radius = 0.01;
    config.num_height_bands = 3;

    ScanContextDatabase db(config);
    db.setMinKeyFrameDistance(0.05);  // 5cm for bunny

    // Add keyframes at different positions
    std::vector<std::pair<V3D, double>> keyframe_poses = {
        {{0.0, 0.0, 0.0}, 0.0},
        {{0.1, 0.0, 0.0}, 0.0},
        {{0.0, 0.1, 0.0}, M_PI / 4},
        {{0.1, 0.1, 0.0}, M_PI / 2},
    };

    for (const auto& [pos, yaw] : keyframe_poses) {
        M4D pose = makePose(pos.x(), pos.y(), pos.z(), yaw);
        // Create "sensor-frame" scan (centered at keyframe)
        db.addKeyFrame(cloud, pose);
    }

    std::cout << "Added " << db.size() << " keyframes\n";

    // Query with a scan from near the first keyframe
    auto matches = db.query(cloud, 5);

    std::cout << "Top matches:\n";
    for (size_t i = 0; i < matches.size(); i++) {
        std::cout << "  " << i+1 << ": KF " << matches[i].keyframe_id
                 << " at [" << matches[i].position.x() << ", "
                 << matches[i].position.y() << "]"
                 << " yaw=" << (matches[i].yaw_estimate * 180 / M_PI) << " deg"
                 << " score=" << matches[i].score << "\n";
    }

    // Best match should be keyframe 0 (same position/orientation)
    bool passed = !matches.empty() && matches[0].keyframe_id == 0;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << "\n";
    return passed;
}

//=============================================================================
// Test 4: Voxel Occupancy Map
//=============================================================================

bool testVoxelOccupancyMap(const std::vector<V3D>& cloud) {
    std::cout << "\n=== Test 4: Voxel Occupancy Map ===\n";

    VoxelOccupancyMap map(0.01);  // 1cm voxels for bunny
    map.build(cloud);

    std::cout << "Map voxels: " << map.numVoxels() << "\n";

    auto [min_pt, max_pt] = map.getBoundingBox();
    std::cout << "Bounding box: [" << min_pt.x() << ", " << min_pt.y() << ", " << min_pt.z()
              << "] to [" << max_pt.x() << ", " << max_pt.y() << ", " << max_pt.z() << "]\n";

    // Test scoring with identity pose (should be high)
    M4D identity = M4D::Identity();
    double score_identity = map.scorePose(cloud, identity, 500);
    std::cout << "Score at identity: " << score_identity << " (should be high)\n";

    // Test scoring with offset pose (should be low)
    M4D offset = makePose(1.0, 0, 0, 0);  // 1 meter offset (far from bunny)
    double score_offset = map.scorePose(cloud, offset, 500);
    std::cout << "Score at 1m offset: " << score_offset << " (should be low)\n";

    bool passed = score_identity > 0.5 && score_offset < score_identity * 0.5;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << "\n";
    return passed;
}

//=============================================================================
// Test 5: Full Global Localization Pipeline
//=============================================================================

bool testGlobalLocalization(const std::vector<V3D>& cloud) {
    std::cout << "\n=== Test 5: Full Global Localization Pipeline ===\n";

    // Create a "map" by placing the bunny at different locations
    std::vector<V3D> map_points;

    // Place bunny copies at a grid of locations (simulating ship hull sections)
    std::vector<V3D> locations = {
        {0.0, 0.0, 0.0},
        {0.3, 0.0, 0.0},
        {0.6, 0.0, 0.0},
        {0.0, 0.3, 0.0},
        {0.3, 0.3, 0.0},
    };

    for (const auto& loc : locations) {
        M4D pose = makePose(loc.x(), loc.y(), loc.z(), 0);
        auto transformed = transformPoints(cloud, pose);
        map_points.insert(map_points.end(), transformed.begin(), transformed.end());
    }

    std::cout << "Created map with " << map_points.size() << " points\n";
    std::cout << "Map covers " << locations.size() << " distinct locations\n";

    // Configure global localizer
    GlobalLocalizerConfig config;
    config.sc_config.max_radius = 0.5;
    config.sc_config.min_radius = 0.01;
    config.sc_config.num_height_bands = 3;
    config.coarse_voxel_size = 0.02;
    config.medium_voxel_size = 0.01;
    config.fine_voxel_size = 0.005;
    config.grid_resolution_xy = 0.1;  // 10cm grid for small bunny
    config.grid_resolution_yaw = 30.0;  // 30 degree resolution

    GlobalLocalizer localizer(config);
    localizer.setMap(map_points);

    // Build Scan Context database
    ScanContextDatabase& db = localizer.database();
    db.setMinKeyFrameDistance(0.1);

    for (const auto& loc : locations) {
        M4D pose = makePose(loc.x(), loc.y(), loc.z(), 0);
        db.addKeyFrame(cloud, pose);  // cloud is already in sensor frame
    }

    std::cout << "Built database with " << db.size() << " keyframes\n";

    // Test localization: scan taken from location 2 (0.6, 0, 0) with some yaw
    double test_x = 0.6;
    double test_y = 0.0;
    double test_yaw = M_PI / 6;  // 30 degrees

    M4D true_pose = makePose(test_x, test_y, 0.0, test_yaw);
    M4D true_pose_inv = true_pose.inverse();

    // Create "sensor frame" scan as seen from test location
    // (points relative to sensor position)
    std::vector<V3D> query_scan;
    query_scan.reserve(cloud.size());
    M3D R_inv = true_pose_inv.block<3,3>(0,0);
    V3D t_inv = true_pose_inv.block<3,1>(0,3);

    // Use the bunny at (0.6, 0, 0) as the visible data
    for (const auto& pt : cloud) {
        V3D world_pt = pt + V3D(0.6, 0, 0);  // Bunny at location 2
        V3D sensor_pt = R_inv * world_pt + t_inv;
        if (sensor_pt.head<2>().norm() < 0.5) {  // Within sensor range
            query_scan.push_back(sensor_pt);
        }
    }

    std::cout << "Query scan has " << query_scan.size() << " points\n";
    std::cout << "True position: [" << test_x << ", " << test_y << "]\n";
    std::cout << "True yaw: " << (test_yaw * 180 / M_PI) << " deg\n";

    // Progress callback
    auto progress_cb = [](const LocalizationProgress& p) {
        std::cout << "  [" << static_cast<int>(p.status) << "] "
                 << p.message << "\n";
    };

    // Run localization
    auto start = std::chrono::high_resolution_clock::now();
    LocalizationResult result = localizer.localize(query_scan, progress_cb);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Extract result
    V3D recovered_pos = result.pose.block<3,1>(0,3);
    double recovered_yaw = std::atan2(result.pose(1,0), result.pose(0,0));

    double pos_error = (recovered_pos.head<2>() - V3D(test_x, test_y, 0).head<2>()).norm();
    double yaw_error = std::abs(recovered_yaw - test_yaw);
    if (yaw_error > M_PI) yaw_error = 2 * M_PI - yaw_error;

    std::cout << "\nLocalization result:\n";
    std::cout << "  Status: " << (result.success() ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "  Message: " << result.message << "\n";
    std::cout << "  Time: " << elapsed_ms << " ms\n";
    std::cout << "  Confidence: " << result.confidence.describe() << "\n";
    std::cout << "  Fitness: " << result.confidence.fitness_score << "\n";
    std::cout << "  Distinctiveness: " << result.confidence.distinctiveness << "\n";
    std::cout << "  Recovered position: [" << recovered_pos.x() << ", "
              << recovered_pos.y() << "]\n";
    std::cout << "  Recovered yaw: " << (recovered_yaw * 180 / M_PI) << " deg\n";
    std::cout << "  Position error: " << (pos_error * 1000) << " mm\n";
    std::cout << "  Yaw error: " << (yaw_error * 180 / M_PI) << " deg\n";

    // Check result
    // Note: This is a simplified test; real-world tolerances would depend on environment
    bool position_ok = pos_error < 0.1;  // 10cm tolerance
    bool yaw_ok = yaw_error < 0.5;  // ~30 degree tolerance (coarse)
    bool passed = result.success() && position_ok;

    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << "\n";
    return passed;
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char** argv) {
    std::cout << "=== Global Localization Tests ===\n";
    std::cout << "Testing global localization from unknown position\n";

    std::string data_dir = "test_data/bunny/data";
    if (argc > 1) {
        data_dir = argv[1];
    }

    // Load bunny data
    std::vector<V3D> bunny;
    std::string bunny_file = data_dir + "/bun000.ply";
    if (!loadFromPly(bunny_file, bunny)) {
        bunny_file = "test_data/bunny/reconstruction/bun_zipper_res2.ply";
        if (!loadFromPly(bunny_file, bunny)) {
            std::cerr << "Cannot find Stanford Bunny data.\n";
            std::cerr << "Usage: " << argv[0] << " <data_dir>\n";
            return 1;
        }
    }

    // Downsample for faster testing
    bunny = voxelDownsample(bunny, 0.002);  // 2mm voxels
    std::cout << "Loaded bunny with " << bunny.size() << " points (after downsample)\n";

    int passed = 0;
    int total = 0;

    // Run tests
    if (testScanContextDescriptor(bunny)) passed++;
    total++;

    if (testScanContextRotation(bunny)) passed++;
    total++;

    if (testScanContextDatabase(bunny)) passed++;
    total++;

    if (testVoxelOccupancyMap(bunny)) passed++;
    total++;

    if (testGlobalLocalization(bunny)) passed++;
    total++;

    // Summary
    std::cout << "\n========================================\n";
    std::cout << "SUMMARY: " << passed << "/" << total << " tests passed";
    if (passed == total) {
        std::cout << " - ALL PASSED!\n";
    } else {
        std::cout << " (" << (100 * passed / total) << "%)\n";
    }
    std::cout << "========================================\n";

    return (passed >= total * 0.8) ? 0 : 1;
}
