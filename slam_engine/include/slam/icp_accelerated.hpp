/**
 * @file icp_accelerated.hpp
 * @brief Hardware-accelerated ICP using nanoflann kd-tree and OpenMP
 *
 * Provides O(n log m) nearest neighbor search instead of O(n*m) brute force.
 * Uses OpenMP for parallel correspondence finding and transform computation.
 */

#ifndef SLAM_ICP_ACCELERATED_HPP
#define SLAM_ICP_ACCELERATED_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "slam/types.hpp"
#include "slam/icp.hpp"  // For ICPConfig, ICPResult, ICPMethod

#ifdef HAS_NANOFLANN
#include <nanoflann.hpp>
#endif

#ifdef MP_EN
#include <omp.h>
#endif

namespace slam {

#ifdef HAS_NANOFLANN

//=============================================================================
// Point Cloud Adaptor for nanoflann
//=============================================================================

struct PointCloudAdaptor {
    const std::vector<V3D>& points;

    PointCloudAdaptor(const std::vector<V3D>& pts) : points(pts) {}

    inline size_t kdtree_get_point_count() const { return points.size(); }

    inline double kdtree_get_pt(size_t idx, size_t dim) const {
        return points[idx](dim);
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

// KD-tree type definition
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>,
    PointCloudAdaptor,
    3  // dimensions
>;

//=============================================================================
// Accelerated ICP Implementation
//=============================================================================

/**
 * @brief High-performance ICP with kd-tree and OpenMP acceleration
 */
class ICPAccelerated {
public:
    explicit ICPAccelerated(const ICPConfig& config = ICPConfig())
        : config_(config) {}

    /**
     * @brief Build kd-tree for target point cloud (do once, reuse for multiple alignments)
     */
    static std::unique_ptr<KDTree> buildKDTree(const std::vector<V3D>& points) {
        auto adaptor = std::make_shared<PointCloudAdaptor>(points);
        auto tree = std::make_unique<KDTree>(3, *adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree->buildIndex();
        return tree;
    }

    /**
     * @brief Align source to target using pre-built kd-tree
     *
     * @param source Source points to align
     * @param target Target points (reference)
     * @param target_tree Pre-built kd-tree for target
     * @param initial_guess Initial transformation estimate
     * @return ICP result with transformation and quality metrics
     */
    ICPResult align(const std::vector<V3D>& source,
                    const std::vector<V3D>& target,
                    const M4D& initial_guess = M4D::Identity()) {

        // Build kd-tree for target
        PointCloudAdaptor adaptor(target);
        KDTree tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree.buildIndex();

        return alignWithTree(source, target, tree, initial_guess);
    }

    /**
     * @brief Align using pre-built kd-tree (faster for multiple alignments to same target)
     */
    ICPResult alignWithTree(const std::vector<V3D>& source,
                            const std::vector<V3D>& target,
                            const KDTree& tree,
                            const M4D& initial_guess = M4D::Identity()) {

        ICPResult result;
        result.transformation = initial_guess;

        if (source.empty() || target.empty()) {
            return result;
        }

        M4D T_current = initial_guess;
        double prev_error = std::numeric_limits<double>::max();

        const double max_dist_sq = config_.max_correspondence_dist * config_.max_correspondence_dist;
        const int n_source = static_cast<int>(source.size());

        // Pre-allocate correspondence buffers
        std::vector<V3D> src_matched(n_source);
        std::vector<V3D> tgt_matched(n_source);
        std::vector<double> distances(n_source);
        std::vector<int> valid(n_source);

        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            result.num_iterations = iter + 1;

            M3D R = T_current.block<3, 3>(0, 0);
            V3D t = T_current.block<3, 1>(0, 3);

            int num_valid = 0;
            double sum_sq_dist = 0.0;

            // Parallel correspondence finding
            #ifdef MP_EN
            #pragma omp parallel for reduction(+:num_valid, sum_sq_dist) schedule(static)
            #endif
            for (int i = 0; i < n_source; ++i) {
                V3D pt_trans = R * source[i] + t;

                // KD-tree query
                double query[3] = {pt_trans.x(), pt_trans.y(), pt_trans.z()};
                size_t nn_idx;
                double nn_dist_sq;

                nanoflann::KNNResultSet<double> result_set(1);
                result_set.init(&nn_idx, &nn_dist_sq);
                tree.findNeighbors(result_set, query, nanoflann::SearchParameters(10));

                if (nn_dist_sq < max_dist_sq) {
                    src_matched[i] = pt_trans;
                    tgt_matched[i] = target[nn_idx];
                    distances[i] = std::sqrt(nn_dist_sq);
                    valid[i] = 1;
                    num_valid++;
                    sum_sq_dist += nn_dist_sq;
                } else {
                    valid[i] = 0;
                }
            }

            if (num_valid < 10) {
                break;
            }

            // Compact valid correspondences
            std::vector<V3D> src_valid, tgt_valid;
            src_valid.reserve(num_valid);
            tgt_valid.reserve(num_valid);

            for (int i = 0; i < n_source; ++i) {
                if (valid[i]) {
                    src_valid.push_back(src_matched[i]);
                    tgt_valid.push_back(tgt_matched[i]);
                }
            }

            // Compute metrics
            result.num_inliers = num_valid;
            result.fitness_score = static_cast<double>(num_valid) / n_source;
            result.rmse = std::sqrt(sum_sq_dist / num_valid);

            // Check convergence
            if (std::abs(prev_error - result.rmse) < config_.convergence_threshold) {
                result.converged = true;
                break;
            }
            prev_error = result.rmse;

            // Compute transformation update
            M4D delta_T = computeTransform(src_valid, tgt_valid);
            T_current = delta_T * T_current;
        }

        result.transformation = T_current;
        result.converged = result.converged || (result.fitness_score > config_.fitness_threshold);

        return result;
    }

    void setConfig(const ICPConfig& config) { config_ = config; }
    const ICPConfig& getConfig() const { return config_; }

private:
    ICPConfig config_;

    /**
     * @brief Compute rigid transformation using SVD (Point-to-Point)
     */
    M4D computeTransform(const std::vector<V3D>& source,
                         const std::vector<V3D>& target) {

        const int n = static_cast<int>(source.size());

        // Compute centroids (parallel reduction)
        V3D src_centroid = V3D::Zero();
        V3D tgt_centroid = V3D::Zero();

        #ifdef MP_EN
        #pragma omp parallel
        {
            V3D local_src = V3D::Zero();
            V3D local_tgt = V3D::Zero();

            #pragma omp for nowait
            for (int i = 0; i < n; ++i) {
                local_src += source[i];
                local_tgt += target[i];
            }

            #pragma omp critical
            {
                src_centroid += local_src;
                tgt_centroid += local_tgt;
            }
        }
        #else
        for (int i = 0; i < n; ++i) {
            src_centroid += source[i];
            tgt_centroid += target[i];
        }
        #endif

        src_centroid /= n;
        tgt_centroid /= n;

        // Build cross-covariance matrix (parallel)
        M3D H = M3D::Zero();

        #ifdef MP_EN
        #pragma omp parallel
        {
            M3D local_H = M3D::Zero();

            #pragma omp for nowait
            for (int i = 0; i < n; ++i) {
                V3D src_centered = source[i] - src_centroid;
                V3D tgt_centered = target[i] - tgt_centroid;
                local_H += src_centered * tgt_centered.transpose();
            }

            #pragma omp critical
            {
                H += local_H;
            }
        }
        #else
        for (int i = 0; i < n; ++i) {
            V3D src_centered = source[i] - src_centroid;
            V3D tgt_centered = target[i] - tgt_centroid;
            H += src_centered * tgt_centered.transpose();
        }
        #endif

        // SVD
        Eigen::JacobiSVD<M3D> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        M3D R = svd.matrixV() * svd.matrixU().transpose();

        // Handle reflection
        if (R.determinant() < 0) {
            M3D V = svd.matrixV();
            V.col(2) *= -1;
            R = V * svd.matrixU().transpose();
        }

        V3D t = tgt_centroid - R * src_centroid;

        M4D T = M4D::Identity();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = t;

        return T;
    }
};

//=============================================================================
// Multi-Scale Accelerated ICP
//=============================================================================

/**
 * @brief Multi-scale ICP with kd-tree acceleration
 */
class MultiScaleICPAccelerated {
public:
    /**
     * @brief Align with coarse-to-fine multi-scale approach
     */
    static ICPResult align(const std::vector<V3D>& source,
                           const std::vector<V3D>& target,
                           const M4D& initial_guess = M4D::Identity(),
                           const std::vector<double>& voxel_sizes = {0.1, 0.05, 0.02}) {

        ICPResult result;
        result.transformation = initial_guess;
        M4D current_transform = initial_guess;

        for (size_t level = 0; level < voxel_sizes.size(); ++level) {
            double voxel_size = voxel_sizes[level];

            // Downsample both clouds
            auto src_down = voxelDownsample(source, voxel_size);
            auto tgt_down = voxelDownsample(target, voxel_size);

            if (src_down.size() < 50 || tgt_down.size() < 50) {
                continue;
            }

            // Configure ICP for this scale
            ICPConfig config;
            config.max_correspondence_dist = voxel_size * 5.0;
            config.max_iterations = 30;
            config.convergence_threshold = voxel_size * 0.01;
            config.fitness_threshold = 0.3;

            ICPAccelerated icp(config);
            result = icp.align(src_down, tgt_down, current_transform);
            current_transform = result.transformation;

            // Early termination if not converging
            if (!result.converged && result.fitness_score < 0.1) {
                break;
            }
        }

        return result;
    }

    /**
     * @brief Align source to pre-built target kd-tree (for repeated queries)
     */
    static ICPResult alignToMap(const std::vector<V3D>& source,
                                 const std::vector<V3D>& target,
                                 const KDTree& target_tree,
                                 const M4D& initial_guess = M4D::Identity(),
                                 const std::vector<double>& voxel_sizes = {0.1, 0.05, 0.02}) {

        ICPResult result;
        result.transformation = initial_guess;
        M4D current_transform = initial_guess;

        for (size_t level = 0; level < voxel_sizes.size(); ++level) {
            double voxel_size = voxel_sizes[level];

            // Downsample source only (target tree is pre-built for full resolution)
            auto src_down = voxelDownsample(source, voxel_size);

            if (src_down.size() < 50) {
                continue;
            }

            ICPConfig config;
            config.max_correspondence_dist = voxel_size * 5.0;
            config.max_iterations = 30;
            config.convergence_threshold = voxel_size * 0.01;
            config.fitness_threshold = 0.3;

            ICPAccelerated icp(config);
            result = icp.alignWithTree(src_down, target, target_tree, current_transform);
            current_transform = result.transformation;

            if (!result.converged && result.fitness_score < 0.1) {
                break;
            }
        }

        return result;
    }
};

#else  // No nanoflann - fall back to basic ICP

class ICPAccelerated : public ICP {
public:
    explicit ICPAccelerated(const ICPConfig& config = ICPConfig()) : ICP(config) {}
};

class MultiScaleICPAccelerated : public MultiScaleICP {
};

#endif  // HAS_NANOFLANN

//=============================================================================
// Profiling Utilities
//=============================================================================

/**
 * @brief Simple timer for profiling
 */
class ScopedTimer {
public:
    ScopedTimer(const std::string& name, double& elapsed_ms)
        : name_(name), elapsed_ms_(elapsed_ms),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        elapsed_ms_ = std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::string name_;
    double& elapsed_ms_;
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief Performance metrics for global localization
 */
struct LocalizationProfile {
    double scan_context_ms = 0;
    double database_query_ms = 0;
    double candidate_scoring_ms = 0;
    double icp_coarse_ms = 0;
    double icp_medium_ms = 0;
    double icp_fine_ms = 0;
    double total_ms = 0;

    void print() const {
        std::cout << "=== Localization Profile ===\n";
        std::cout << "  Scan Context:      " << scan_context_ms << " ms\n";
        std::cout << "  Database Query:    " << database_query_ms << " ms\n";
        std::cout << "  Candidate Scoring: " << candidate_scoring_ms << " ms\n";
        std::cout << "  ICP Coarse:        " << icp_coarse_ms << " ms\n";
        std::cout << "  ICP Medium:        " << icp_medium_ms << " ms\n";
        std::cout << "  ICP Fine:          " << icp_fine_ms << " ms\n";
        std::cout << "  -------------------\n";
        std::cout << "  Total:             " << total_ms << " ms\n";
    }
};

} // namespace slam

#endif // SLAM_ICP_ACCELERATED_HPP
