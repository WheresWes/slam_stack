/**
 * @file benchmark_global_localization.cpp
 * @brief Quick performance benchmark for global localization components
 */

#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "slam/types.hpp"
#include "slam/global_localization.hpp"
#include "slam/ply_reader.hpp"
#include "slam/icp.hpp"
#include "slam/icp_accelerated.hpp"

using namespace slam;

int main(int argc, char** argv) {
    std::cout << "============================================\n";
    std::cout << "  Global Localization Performance Benchmark\n";
    std::cout << "============================================\n\n";

    // Load test data
    std::string bunny_file = "test_data/bunny/data/bun000.ply";
    if (argc > 1) bunny_file = argv[1];

    std::vector<V3D> cloud;
    if (!loadFromPly(bunny_file, cloud)) {
        bunny_file = "test_data/bunny/reconstruction/bun_zipper_res2.ply";
        if (!loadFromPly(bunny_file, cloud)) {
            std::cerr << "Cannot load test data\n";
            return 1;
        }
    }

    cloud = voxelDownsample(cloud, 0.003);
    std::cout << "Test cloud: " << cloud.size() << " points\n\n";

    //=========================================================================
    // Benchmark 1: Scan Context Descriptor Computation
    //=========================================================================
    std::cout << "=== 1. Scan Context Descriptor ===\n";
    {
        ScanContextConfig config;
        config.num_rings = 20;
        config.num_sectors = 60;
        config.max_radius = 0.2;
        config.min_radius = 0.01;
        config.num_height_bands = 3;

        ScanContext sc(config);

        auto start = std::chrono::high_resolution_clock::now();
        int iterations = 50;
        Eigen::MatrixXd desc;
        for (int i = 0; i < iterations; i++) {
            desc = sc.compute(cloud);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double time_us = std::chrono::duration<double, std::micro>(end - start).count() / iterations;

        std::cout << "  Compute time: " << std::fixed << std::setprecision(1)
                  << time_us << " us/descriptor\n";
        std::cout << "  Descriptor: " << desc.rows() << "x" << desc.cols()
                  << " = " << desc.size() << " elements\n";
    }

    //=========================================================================
    // Benchmark 2: Scan Context Matching
    //=========================================================================
    std::cout << "\n=== 2. Scan Context Matching ===\n";
    {
        ScanContextConfig config;
        config.num_rings = 20;
        config.num_sectors = 60;
        config.max_radius = 0.2;

        ScanContext sc(config);
        auto desc1 = sc.compute(cloud);
        auto desc2 = sc.compute(cloud);

        auto start = std::chrono::high_resolution_clock::now();
        int iterations = 100;
        for (int i = 0; i < iterations; i++) {
            auto [score, shift] = sc.match(desc1, desc2);
            (void)score; (void)shift;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double time_us = std::chrono::duration<double, std::micro>(end - start).count() / iterations;

        std::cout << "  Match time: " << std::fixed << std::setprecision(1)
                  << time_us << " us/match\n";
    }

    //=========================================================================
    // Benchmark 3: Voxel Occupancy Map
    //=========================================================================
    std::cout << "\n=== 3. Voxel Occupancy Map ===\n";
    {
        VoxelOccupancyMap map(0.01);

        auto start = std::chrono::high_resolution_clock::now();
        map.build(cloud);
        auto end = std::chrono::high_resolution_clock::now();
        double build_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "  Build time: " << std::fixed << std::setprecision(2)
                  << build_ms << " ms\n";
        std::cout << "  Voxels: " << map.numVoxels() << "\n";

        // Benchmark scoring
        M4D pose = M4D::Identity();
        start = std::chrono::high_resolution_clock::now();
        int iterations = 20;
        for (int i = 0; i < iterations; i++) {
            double score = map.scorePose(cloud, pose, 500);
            (void)score;
        }
        end = std::chrono::high_resolution_clock::now();
        double score_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        std::cout << "  Score time: " << score_ms << " ms/pose (500 samples)\n";
    }

    //=========================================================================
    // Benchmark 4: ICP Alignment
    //=========================================================================
    std::cout << "\n=== 4. ICP Alignment ===\n";
    {
        // Create target (original) and source (rotated)
        M3D R = Eigen::AngleAxisd(0.1, V3D::UnitZ()).toRotationMatrix();
        V3D t(0.01, 0.005, 0);

        std::vector<V3D> source;
        source.reserve(cloud.size());
        for (const auto& p : cloud) {
            source.push_back(R * p + t);
        }

        ICPConfig config;
        config.max_iterations = 30;
        config.convergence_threshold = 1e-6;
        config.max_correspondence_dist = 0.05;
        config.method = ICPMethod::POINT_TO_POINT;

        ICP icp(config);

        auto start = std::chrono::high_resolution_clock::now();
        int iterations = 5;
        ICPResult result;
        for (int i = 0; i < iterations; i++) {
            result = icp.align(source, cloud, M4D::Identity());
        }
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        std::cout << "  Point-to-Point ICP: " << std::fixed << std::setprecision(1)
                  << time_ms << " ms\n";
        std::cout << "  Iterations: " << result.num_iterations
                  << ", RMSE: " << std::setprecision(4) << result.rmse << "\n";

        // GICP
        config.method = ICPMethod::GICP;
        ICP icp_gicp(config);
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            result = icp_gicp.align(source, cloud, M4D::Identity());
        }
        end = std::chrono::high_resolution_clock::now();
        time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        std::cout << "  GICP: " << std::setprecision(1) << time_ms << " ms\n";
        std::cout << "  Iterations: " << result.num_iterations
                  << ", RMSE: " << std::setprecision(4) << result.rmse << "\n";

#ifdef HAS_NANOFLANN
        // Accelerated ICP with kd-tree
        ICPConfig accel_config;
        accel_config.max_iterations = 30;
        accel_config.convergence_threshold = 1e-6;
        accel_config.max_correspondence_dist = 0.05;

        ICPAccelerated icp_accel(accel_config);

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            result = icp_accel.align(source, cloud, M4D::Identity());
        }
        end = std::chrono::high_resolution_clock::now();
        time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        std::cout << "  Accelerated ICP (kd-tree): " << std::setprecision(1) << time_ms << " ms\n";
        std::cout << "  Iterations: " << result.num_iterations
                  << ", RMSE: " << std::setprecision(4) << result.rmse << "\n";

        // Multi-scale accelerated
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            result = MultiScaleICPAccelerated::align(source, cloud, M4D::Identity(),
                                                      {0.02, 0.01, 0.005});
        }
        end = std::chrono::high_resolution_clock::now();
        time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        std::cout << "  Multi-Scale Accelerated: " << std::setprecision(1) << time_ms << " ms\n";
        std::cout << "  Iterations: " << result.num_iterations
                  << ", RMSE: " << std::setprecision(4) << result.rmse << "\n";
#else
        std::cout << "  (nanoflann not available - accelerated ICP disabled)\n";
#endif
    }

    //=========================================================================
    // Benchmark 5: Database Query
    //=========================================================================
    std::cout << "\n=== 5. Scan Context Database Query ===\n";
    {
        ScanContextConfig config;
        config.num_rings = 20;
        config.num_sectors = 60;
        config.max_radius = 0.5;

        ScanContextDatabase db(config);
        db.setMinKeyFrameDistance(0.1);

        // Add keyframes
        for (int i = 0; i < 20; i++) {
            M4D pose = M4D::Identity();
            pose(0, 3) = i * 0.1;
            db.addKeyFrame(cloud, pose);
        }
        std::cout << "  Database: " << db.size() << " keyframes\n";

        auto start = std::chrono::high_resolution_clock::now();
        int iterations = 10;
        for (int i = 0; i < iterations; i++) {
            auto matches = db.query(cloud, 5);
            (void)matches;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        std::cout << "  Query time (top-5): " << std::fixed << std::setprecision(2)
                  << time_ms << " ms\n";
    }

    //=========================================================================
    // Benchmark 6: Voxel Downsampling
    //=========================================================================
    std::cout << "\n=== 6. Voxel Downsampling ===\n";
    {
        auto start = std::chrono::high_resolution_clock::now();
        int iterations = 20;
        std::vector<V3D> downsampled;
        for (int i = 0; i < iterations; i++) {
            downsampled = voxelDownsample(cloud, 0.005);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

        std::cout << "  Time: " << std::fixed << std::setprecision(2) << time_ms << " ms\n";
        std::cout << "  " << cloud.size() << " -> " << downsampled.size() << " points\n";
    }

    //=========================================================================
    // Benchmark 7: Scaling Test (simulate larger maps)
    //=========================================================================
    std::cout << "\n=== 7. ICP Scaling Test ===\n";
    {
        // Create larger point clouds by duplicating and offsetting
        std::vector<size_t> test_sizes = {1000, 5000, 10000, 25000};

        for (size_t target_size : test_sizes) {
            // Build a larger cloud
            std::vector<V3D> large_cloud;
            large_cloud.reserve(target_size);

            int copies = static_cast<int>((target_size + cloud.size() - 1) / cloud.size());
            for (int c = 0; c < copies && large_cloud.size() < target_size; c++) {
                V3D offset(c * 0.2, 0, 0);
                for (const auto& p : cloud) {
                    if (large_cloud.size() >= target_size) break;
                    large_cloud.push_back(p + offset);
                }
            }

            // Create transformed source
            M3D R = Eigen::AngleAxisd(0.05, V3D::UnitZ()).toRotationMatrix();
            V3D t(0.01, 0.005, 0);
            std::vector<V3D> source;
            source.reserve(large_cloud.size() / 2);
            for (size_t i = 0; i < large_cloud.size() / 2; i++) {
                source.push_back(R * large_cloud[i] + t);
            }

            // Time brute-force ICP
            ICPConfig config;
            config.max_iterations = 10;  // Limit iterations for fair comparison
            config.max_correspondence_dist = 0.1;
            config.method = ICPMethod::POINT_TO_POINT;
            ICP icp_basic(config);

            auto start = std::chrono::high_resolution_clock::now();
            auto result = icp_basic.align(source, large_cloud, M4D::Identity());
            auto end = std::chrono::high_resolution_clock::now();
            double basic_ms = std::chrono::duration<double, std::milli>(end - start).count();

#ifdef HAS_NANOFLANN
            // Time accelerated ICP
            ICPAccelerated icp_accel(config);

            start = std::chrono::high_resolution_clock::now();
            result = icp_accel.align(source, large_cloud, M4D::Identity());
            end = std::chrono::high_resolution_clock::now();
            double accel_ms = std::chrono::duration<double, std::milli>(end - start).count();

            double speedup = basic_ms / accel_ms;

            std::cout << "  " << source.size() << " vs " << large_cloud.size() << " points: "
                      << std::fixed << std::setprecision(1)
                      << "basic=" << basic_ms << "ms, accel=" << accel_ms << "ms, "
                      << "speedup=" << std::setprecision(1) << speedup << "x\n";
#else
            std::cout << "  " << source.size() << " vs " << large_cloud.size() << " points: "
                      << std::fixed << std::setprecision(1)
                      << "basic=" << basic_ms << "ms (no accel available)\n";
#endif
        }
    }

    //=========================================================================
    // Benchmark 8: OpenMP Thread Scaling
    //=========================================================================
#ifdef MP_EN
    std::cout << "\n=== 8. OpenMP Thread Scaling ===\n";
    {
        int max_threads = omp_get_max_threads();
        std::cout << "  Max threads available: " << max_threads << "\n";

        // Create a moderately large test case
        std::vector<V3D> large_target;
        large_target.reserve(10000);
        for (int c = 0; c < 3; c++) {
            V3D offset(c * 0.2, 0, 0);
            for (const auto& p : cloud) {
                if (large_target.size() >= 10000) break;
                large_target.push_back(p + offset);
            }
        }

        M3D R = Eigen::AngleAxisd(0.05, V3D::UnitZ()).toRotationMatrix();
        V3D t(0.01, 0.005, 0);
        std::vector<V3D> source;
        for (size_t i = 0; i < large_target.size() / 2; i++) {
            source.push_back(R * large_target[i] + t);
        }

#ifdef HAS_NANOFLANN
        ICPConfig config;
        config.max_iterations = 20;
        config.max_correspondence_dist = 0.1;
        ICPAccelerated icp(config);

        std::vector<int> thread_counts = {1, 2, 4};
        for (int t : thread_counts) {
            if (t > max_threads) break;
            omp_set_num_threads(t);

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 3; i++) {
                auto result = icp.align(source, large_target, M4D::Identity());
                (void)result;
            }
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count() / 3;

            std::cout << "  " << t << " thread(s): " << std::fixed << std::setprecision(1)
                      << ms << " ms\n";
        }

        // Restore max threads
        omp_set_num_threads(max_threads);
#else
        std::cout << "  (nanoflann required for thread scaling test)\n";
#endif
    }
#endif

    //=========================================================================
    // Summary
    //=========================================================================
    std::cout << "\n============================================\n";
    std::cout << "  BENCHMARK COMPLETE\n";
    std::cout << "============================================\n";

    return 0;
}
