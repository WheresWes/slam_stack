/**
 * @file icp.hpp
 * @brief Native ICP implementation for global re-localization
 *
 * Provides multiple ICP variants:
 * - Point-to-Point ICP (basic)
 * - Point-to-Plane ICP (faster convergence for planar environments)
 * - Generalized ICP (GICP - plane-to-plane, most robust)
 * - Multi-scale ICP (coarse-to-fine)
 *
 * Also includes utilities:
 * - FOV cropping for efficient scan-to-map matching
 * - Normal estimation for point clouds
 * - Voxel downsampling
 *
 * No external dependencies (Open3D, etc.)
 */

#ifndef SLAM_ICP_HPP
#define SLAM_ICP_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include <numeric>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "slam/types.hpp"
#ifdef HAS_PCL
#include "slam/ikd-Tree/ikd_Tree.h"
#endif

namespace slam {

//=============================================================================
// ICP Configuration
//=============================================================================

enum class ICPMethod {
    POINT_TO_POINT,    // Basic ICP - minimizes point-to-point distance
    POINT_TO_PLANE,    // Minimizes distance along target normal (faster convergence)
    GICP               // Generalized ICP - plane-to-plane (most robust)
};

struct ICPConfig {
    ICPMethod method = ICPMethod::POINT_TO_POINT;
    int max_iterations = 30;
    double convergence_threshold = 1e-6;   // Change in error for convergence
    double max_correspondence_dist = 2.0;  // Max distance for point pairs
    double fitness_threshold = 0.3;        // Min inlier ratio for success
    int normal_estimation_k = 10;          // K neighbors for normal estimation
    double normal_estimation_radius = 0.5; // Radius for normal estimation
    int num_threads = 4;
};

struct ICPResult {
    M4D transformation = M4D::Identity();
    double fitness_score = 0.0;            // Inlier ratio (0-1)
    double rmse = std::numeric_limits<double>::max();
    int num_inliers = 0;
    int num_iterations = 0;
    bool converged = false;
};

//=============================================================================
// FOV Cropping Utility
//=============================================================================

struct FOVConfig {
    double fov_horizontal = 2.0 * M_PI;  // Full 360 for spinning LiDAR
    double fov_vertical = M_PI;           // Full vertical
    double min_range = 0.5;               // Minimum range (m)
    double max_range = 100.0;             // Maximum range (m)
};

/**
 * @brief Crop point cloud to sensor field of view
 *
 * Filters points based on the sensor's FOV from a given pose.
 * This significantly reduces computation for scan-to-map matching.
 *
 * @param points Input point cloud (in map frame)
 * @param sensor_pose Sensor pose in map frame
 * @param config FOV configuration
 * @return Points visible from sensor pose
 */
inline std::vector<V3D> cropToFOV(const std::vector<V3D>& points,
                                   const M4D& sensor_pose,
                                   const FOVConfig& config = FOVConfig()) {
    std::vector<V3D> cropped;
    cropped.reserve(points.size() / 4);  // Rough estimate

    // Transform from map to sensor frame
    M3D R = sensor_pose.block<3, 3>(0, 0);
    V3D t = sensor_pose.block<3, 1>(0, 3);
    M3D R_inv = R.transpose();

    double half_fov_h = config.fov_horizontal / 2.0;
    double half_fov_v = config.fov_vertical / 2.0;
    double min_range_sq = config.min_range * config.min_range;
    double max_range_sq = config.max_range * config.max_range;

    for (const auto& pt_map : points) {
        // Transform to sensor frame
        V3D pt_sensor = R_inv * (pt_map - t);

        double range_sq = pt_sensor.squaredNorm();

        // Range check
        if (range_sq < min_range_sq || range_sq > max_range_sq) {
            continue;
        }

        // Horizontal angle (azimuth)
        double azimuth = std::atan2(pt_sensor.y(), pt_sensor.x());

        // Vertical angle (elevation)
        double range_xy = std::sqrt(pt_sensor.x() * pt_sensor.x() +
                                    pt_sensor.y() * pt_sensor.y());
        double elevation = std::atan2(pt_sensor.z(), range_xy);

        // FOV check
        if (std::abs(azimuth) <= half_fov_h && std::abs(elevation) <= half_fov_v) {
            cropped.push_back(pt_map);
        }
    }

    return cropped;
}

//=============================================================================
// Normal Estimation
//=============================================================================

/**
 * @brief Point with normal information
 */
struct PointWithNormal {
    V3D point;
    V3D normal;
    M3D covariance;  // For GICP

    PointWithNormal() : point(V3D::Zero()), normal(V3D::UnitZ()),
                        covariance(M3D::Identity()) {}
    PointWithNormal(const V3D& p, const V3D& n)
        : point(p), normal(n), covariance(M3D::Identity()) {}
};

/**
 * @brief Estimate normals for a point cloud using PCA
 *
 * @param points Input point cloud
 * @param k Number of nearest neighbors for normal estimation
 * @return Points with estimated normals
 */
inline std::vector<PointWithNormal> estimateNormals(
    const std::vector<V3D>& points,
    int k = 10) {

    std::vector<PointWithNormal> result;
    result.reserve(points.size());

    if (points.size() < static_cast<size_t>(k)) {
        // Not enough points for normal estimation
        for (const auto& pt : points) {
            result.emplace_back(pt, V3D::UnitZ());
        }
        return result;
    }

    // For each point, find k nearest neighbors and compute normal via PCA
    for (size_t i = 0; i < points.size(); ++i) {
        // Find k nearest neighbors (brute force for simplicity)
        std::vector<std::pair<double, size_t>> distances;
        distances.reserve(points.size());

        for (size_t j = 0; j < points.size(); ++j) {
            if (i != j) {
                double dist_sq = (points[i] - points[j]).squaredNorm();
                distances.emplace_back(dist_sq, j);
            }
        }

        // Partial sort to get k nearest
        size_t num_neighbors = std::min(static_cast<size_t>(k), distances.size());
        std::partial_sort(distances.begin(),
                          distances.begin() + num_neighbors,
                          distances.end());

        // Compute centroid of neighborhood
        V3D centroid = points[i];
        for (size_t n = 0; n < num_neighbors; ++n) {
            centroid += points[distances[n].second];
        }
        centroid /= static_cast<double>(num_neighbors + 1);

        // Compute covariance matrix
        M3D cov = M3D::Zero();
        V3D diff = points[i] - centroid;
        cov += diff * diff.transpose();
        for (size_t n = 0; n < num_neighbors; ++n) {
            diff = points[distances[n].second] - centroid;
            cov += diff * diff.transpose();
        }
        cov /= static_cast<double>(num_neighbors + 1);

        // Eigen decomposition - smallest eigenvector is normal
        Eigen::SelfAdjointEigenSolver<M3D> solver(cov);
        V3D normal = solver.eigenvectors().col(0);  // Smallest eigenvalue

        // Ensure consistent normal orientation (pointing "up" in Z)
        if (normal.z() < 0) {
            normal = -normal;
        }

        PointWithNormal pwn;
        pwn.point = points[i];
        pwn.normal = normal;
        pwn.covariance = cov;
        result.push_back(pwn);
    }

    return result;
}

//=============================================================================
// Voxel Downsampling
//=============================================================================

/**
 * @brief Downsample point cloud using voxel grid filter
 */
inline std::vector<V3D> voxelDownsample(const std::vector<V3D>& points,
                                         double voxel_size) {
    if (voxel_size <= 0 || points.empty()) return points;

    std::unordered_map<int64_t, V3D> voxel_sum;
    std::unordered_map<int64_t, int> voxel_count;

    auto hashVoxel = [voxel_size](const V3D& pt) -> int64_t {
        int64_t ix = static_cast<int64_t>(std::floor(pt.x() / voxel_size));
        int64_t iy = static_cast<int64_t>(std::floor(pt.y() / voxel_size));
        int64_t iz = static_cast<int64_t>(std::floor(pt.z() / voxel_size));
        return (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791);
    };

    for (const auto& pt : points) {
        int64_t hash = hashVoxel(pt);
        if (voxel_sum.find(hash) == voxel_sum.end()) {
            voxel_sum[hash] = pt;
            voxel_count[hash] = 1;
        } else {
            voxel_sum[hash] += pt;
            voxel_count[hash]++;
        }
    }

    std::vector<V3D> result;
    result.reserve(voxel_sum.size());
    for (auto& kv : voxel_sum) {
        result.push_back(kv.second / static_cast<double>(voxel_count[kv.first]));
    }

    return result;
}

//=============================================================================
// ICP Implementation
//=============================================================================

class ICP {
public:
    ICP() = default;
    explicit ICP(const ICPConfig& config) : config_(config) {}

    /**
     * @brief Align source point cloud to target using ICP
     * @param source Source points to transform
     * @param target Target points (reference)
     * @param initial_guess Initial transformation estimate
     * @return ICP result with transformation and quality metrics
     */
    ICPResult align(const std::vector<V3D>& source,
                    const std::vector<V3D>& target,
                    const M4D& initial_guess = M4D::Identity()) {

        // Dispatch to appropriate method
        switch (config_.method) {
            case ICPMethod::POINT_TO_PLANE:
                return alignPointToPlane(source, target, initial_guess);
            case ICPMethod::GICP:
                return alignGICP(source, target, initial_guess);
            case ICPMethod::POINT_TO_POINT:
            default:
                return alignPointToPoint(source, target, initial_guess);
        }
    }

#ifdef HAS_PCL
    /**
     * @brief Align source to target using ikd-tree for efficient NN search
     */
    template<typename PointType>
    ICPResult alignWithKdTree(const std::vector<V3D>& source,
                               KD_TREE<PointType>& kdtree,
                               const M4D& initial_guess = M4D::Identity()) {
        ICPResult result;
        result.transformation = initial_guess;

        if (source.empty() || kdtree.Root_Node == nullptr) {
            return result;
        }

        M4D T_current = initial_guess;
        double prev_error = std::numeric_limits<double>::max();

        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            result.num_iterations = iter + 1;

            // Transform source points and find correspondences via ikd-tree
            std::vector<V3D> src_transformed;
            std::vector<V3D> tgt_matched;
            std::vector<double> distances;

            src_transformed.reserve(source.size());
            tgt_matched.reserve(source.size());
            distances.reserve(source.size());

            M3D R = T_current.block<3, 3>(0, 0);
            V3D t = T_current.block<3, 1>(0, 3);

            for (const auto& pt : source) {
                V3D pt_trans = R * pt + t;

                // Query ikd-tree for nearest neighbor
                PointType query_pt;
                query_pt.x = static_cast<float>(pt_trans.x());
                query_pt.y = static_cast<float>(pt_trans.y());
                query_pt.z = static_cast<float>(pt_trans.z());

                std::vector<PointType, Eigen::aligned_allocator<PointType>> nearest;
                std::vector<float> sq_dists;
                kdtree.Nearest_Search(query_pt, 1, nearest, sq_dists,
                                      static_cast<float>(config_.max_correspondence_dist));

                if (!nearest.empty() && sq_dists[0] < config_.max_correspondence_dist * config_.max_correspondence_dist) {
                    src_transformed.push_back(pt_trans);
                    tgt_matched.emplace_back(nearest[0].x, nearest[0].y, nearest[0].z);
                    distances.push_back(std::sqrt(sq_dists[0]));
                }
            }

            if (src_transformed.size() < 10) {
                break;
            }

            // Compute metrics
            result.num_inliers = static_cast<int>(src_transformed.size());
            result.fitness_score = static_cast<double>(result.num_inliers) / source.size();

            double sum_sq = 0;
            for (double d : distances) sum_sq += d * d;
            result.rmse = std::sqrt(sum_sq / distances.size());

            // Check convergence
            if (std::abs(prev_error - result.rmse) < config_.convergence_threshold) {
                result.converged = true;
                break;
            }
            prev_error = result.rmse;

            // Compute transformation via SVD
            M4D delta_T = computeSVDTransform(src_transformed, tgt_matched);
            T_current = delta_T * T_current;
        }

        result.transformation = T_current;
        result.converged = result.converged || (result.fitness_score > config_.fitness_threshold);

        return result;
    }
#endif // HAS_PCL

    void setConfig(const ICPConfig& config) { config_ = config; }
    const ICPConfig& getConfig() const { return config_; }

    /**
     * @brief Run a single ICP iteration (for progress reporting)
     *
     * This maintains internal state between calls. The first call initializes
     * the iteration, subsequent calls continue from previous state.
     *
     * @param source Source points to transform
     * @param target Target points (reference)
     * @param current_transform Current transformation estimate
     * @return Result after this iteration (use converged flag to check completion)
     */
    ICPResult alignOneIteration(const std::vector<V3D>& source,
                                const std::vector<V3D>& target,
                                const M4D& current_transform) {
        ICPResult result;
        result.transformation = current_transform;

        if (source.empty() || target.empty()) {
            result.converged = true;
            return result;
        }

        // Transform source points
        std::vector<V3D> transformed_source;
        transformed_source.reserve(source.size());
        M3D R = current_transform.block<3, 3>(0, 0);
        V3D t = current_transform.block<3, 1>(0, 3);
        for (const auto& pt : source) {
            transformed_source.push_back(R * pt + t);
        }

        // Find correspondences
        std::vector<std::pair<int, int>> correspondences;
        std::vector<double> distances;
        findCorrespondences(transformed_source, target, correspondences, distances);

        if (correspondences.size() < 10) {
            result.converged = true;  // Can't continue, treat as converged
            result.fitness_score = 0.0;
            return result;
        }

        // Compute metrics
        result.num_inliers = static_cast<int>(correspondences.size());
        result.fitness_score = static_cast<double>(result.num_inliers) / source.size();

        double sum_sq_error = 0;
        for (double d : distances) {
            sum_sq_error += d * d;
        }
        result.rmse = std::sqrt(sum_sq_error / distances.size());

        // Check convergence against previous error
        if (std::abs(prev_error_ - result.rmse) < config_.convergence_threshold) {
            result.converged = true;
        }
        prev_error_ = result.rmse;

        // Compute transformation update
        M4D delta_T = computePointToPointTransform(transformed_source, target, correspondences);
        result.transformation = delta_T * current_transform;
        result.num_iterations = 1;

        return result;
    }

    /**
     * @brief Reset internal state for alignOneIteration
     */
    void resetIterationState() {
        prev_error_ = std::numeric_limits<double>::max();
    }

private:
    double prev_error_ = std::numeric_limits<double>::max();
    ICPConfig config_;

    //=========================================================================
    // Point-to-Point ICP
    //=========================================================================

    ICPResult alignPointToPoint(const std::vector<V3D>& source,
                                 const std::vector<V3D>& target,
                                 const M4D& initial_guess) {
        ICPResult result;
        result.transformation = initial_guess;

        if (source.empty() || target.empty()) {
            return result;
        }

        M4D T_current = initial_guess;
        double prev_error = std::numeric_limits<double>::max();

        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            result.num_iterations = iter + 1;

            // Transform source points
            std::vector<V3D> transformed_source;
            transformed_source.reserve(source.size());
            for (const auto& pt : source) {
                V3D pt_h = T_current.block<3, 3>(0, 0) * pt + T_current.block<3, 1>(0, 3);
                transformed_source.push_back(pt_h);
            }

            // Find correspondences
            std::vector<std::pair<int, int>> correspondences;
            std::vector<double> distances;
            findCorrespondences(transformed_source, target, correspondences, distances);

            if (correspondences.size() < 10) {
                break;
            }

            // Compute metrics
            result.num_inliers = static_cast<int>(correspondences.size());
            result.fitness_score = static_cast<double>(result.num_inliers) / source.size();

            double sum_sq_error = 0;
            for (double d : distances) {
                sum_sq_error += d * d;
            }
            result.rmse = std::sqrt(sum_sq_error / distances.size());

            // Check convergence
            if (std::abs(prev_error - result.rmse) < config_.convergence_threshold) {
                result.converged = true;
                break;
            }
            prev_error = result.rmse;

            // Compute transformation update
            M4D delta_T = computePointToPointTransform(transformed_source, target, correspondences);
            T_current = delta_T * T_current;
        }

        result.transformation = T_current;
        result.converged = result.converged || (result.fitness_score > config_.fitness_threshold);

        return result;
    }

    //=========================================================================
    // Point-to-Plane ICP
    //=========================================================================

    ICPResult alignPointToPlane(const std::vector<V3D>& source,
                                 const std::vector<V3D>& target,
                                 const M4D& initial_guess) {
        ICPResult result;
        result.transformation = initial_guess;

        if (source.empty() || target.empty()) {
            return result;
        }

        // Estimate normals for target
        auto target_with_normals = estimateNormals(target, config_.normal_estimation_k);

        M4D T_current = initial_guess;
        double prev_error = std::numeric_limits<double>::max();

        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            result.num_iterations = iter + 1;

            // Transform source points
            std::vector<V3D> transformed_source;
            transformed_source.reserve(source.size());
            M3D R = T_current.block<3, 3>(0, 0);
            V3D t = T_current.block<3, 1>(0, 3);
            for (const auto& pt : source) {
                transformed_source.push_back(R * pt + t);
            }

            // Find correspondences with normals
            std::vector<V3D> src_pts, tgt_pts, tgt_normals;
            std::vector<double> distances;
            double max_dist_sq = config_.max_correspondence_dist * config_.max_correspondence_dist;

            for (size_t i = 0; i < transformed_source.size(); ++i) {
                double min_dist_sq = max_dist_sq;
                int best_j = -1;

                for (size_t j = 0; j < target_with_normals.size(); ++j) {
                    double dist_sq = (transformed_source[i] - target_with_normals[j].point).squaredNorm();
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        best_j = static_cast<int>(j);
                    }
                }

                if (best_j >= 0) {
                    src_pts.push_back(transformed_source[i]);
                    tgt_pts.push_back(target_with_normals[best_j].point);
                    tgt_normals.push_back(target_with_normals[best_j].normal);
                    distances.push_back(std::sqrt(min_dist_sq));
                }
            }

            if (src_pts.size() < 10) {
                break;
            }

            // Compute metrics
            result.num_inliers = static_cast<int>(src_pts.size());
            result.fitness_score = static_cast<double>(result.num_inliers) / source.size();

            // Compute point-to-plane RMSE
            double sum_sq_error = 0;
            for (size_t i = 0; i < src_pts.size(); ++i) {
                V3D diff = src_pts[i] - tgt_pts[i];
                double plane_dist = diff.dot(tgt_normals[i]);
                sum_sq_error += plane_dist * plane_dist;
            }
            result.rmse = std::sqrt(sum_sq_error / src_pts.size());

            // Check convergence
            if (std::abs(prev_error - result.rmse) < config_.convergence_threshold) {
                result.converged = true;
                break;
            }
            prev_error = result.rmse;

            // Compute point-to-plane transformation (linearized)
            M4D delta_T = computePointToPlaneTransform(src_pts, tgt_pts, tgt_normals);
            T_current = delta_T * T_current;
        }

        result.transformation = T_current;
        result.converged = result.converged || (result.fitness_score > config_.fitness_threshold);

        return result;
    }

    //=========================================================================
    // Generalized ICP (GICP)
    //=========================================================================

    ICPResult alignGICP(const std::vector<V3D>& source,
                         const std::vector<V3D>& target,
                         const M4D& initial_guess) {
        ICPResult result;
        result.transformation = initial_guess;

        if (source.empty() || target.empty()) {
            return result;
        }

        // Estimate normals and covariances for both point clouds
        auto source_with_normals = estimateNormals(source, config_.normal_estimation_k);
        auto target_with_normals = estimateNormals(target, config_.normal_estimation_k);

        M4D T_current = initial_guess;
        double prev_error = std::numeric_limits<double>::max();

        for (int iter = 0; iter < config_.max_iterations; ++iter) {
            result.num_iterations = iter + 1;

            M3D R = T_current.block<3, 3>(0, 0);
            V3D t = T_current.block<3, 1>(0, 3);

            // Find correspondences
            std::vector<size_t> src_indices, tgt_indices;
            std::vector<double> distances;
            double max_dist_sq = config_.max_correspondence_dist * config_.max_correspondence_dist;

            for (size_t i = 0; i < source_with_normals.size(); ++i) {
                V3D pt_trans = R * source_with_normals[i].point + t;
                double min_dist_sq = max_dist_sq;
                int best_j = -1;

                for (size_t j = 0; j < target_with_normals.size(); ++j) {
                    double dist_sq = (pt_trans - target_with_normals[j].point).squaredNorm();
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        best_j = static_cast<int>(j);
                    }
                }

                if (best_j >= 0) {
                    src_indices.push_back(i);
                    tgt_indices.push_back(static_cast<size_t>(best_j));
                    distances.push_back(std::sqrt(min_dist_sq));
                }
            }

            if (src_indices.size() < 10) {
                break;
            }

            // Compute metrics
            result.num_inliers = static_cast<int>(src_indices.size());
            result.fitness_score = static_cast<double>(result.num_inliers) / source.size();

            double sum_sq = 0;
            for (double d : distances) sum_sq += d * d;
            result.rmse = std::sqrt(sum_sq / distances.size());

            // Check convergence
            if (std::abs(prev_error - result.rmse) < config_.convergence_threshold) {
                result.converged = true;
                break;
            }
            prev_error = result.rmse;

            // Compute GICP transformation
            M4D delta_T = computeGICPTransform(source_with_normals, target_with_normals,
                                                src_indices, tgt_indices, T_current);
            T_current = delta_T * T_current;
        }

        result.transformation = T_current;
        result.converged = result.converged || (result.fitness_score > config_.fitness_threshold);

        return result;
    }

    //=========================================================================
    // Helper Functions
    //=========================================================================

    void findCorrespondences(const std::vector<V3D>& source,
                             const std::vector<V3D>& target,
                             std::vector<std::pair<int, int>>& correspondences,
                             std::vector<double>& distances) {
        correspondences.clear();
        distances.clear();

        double max_dist_sq = config_.max_correspondence_dist * config_.max_correspondence_dist;

        for (size_t i = 0; i < source.size(); ++i) {
            double min_dist_sq = max_dist_sq;
            int best_j = -1;

            for (size_t j = 0; j < target.size(); ++j) {
                double dist_sq = (source[i] - target[j]).squaredNorm();
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    best_j = static_cast<int>(j);
                }
            }

            if (best_j >= 0) {
                correspondences.emplace_back(static_cast<int>(i), best_j);
                distances.push_back(std::sqrt(min_dist_sq));
            }
        }
    }

    M4D computePointToPointTransform(const std::vector<V3D>& source,
                                      const std::vector<V3D>& target,
                                      const std::vector<std::pair<int, int>>& correspondences) {
        std::vector<V3D> src_pts, tgt_pts;
        src_pts.reserve(correspondences.size());
        tgt_pts.reserve(correspondences.size());

        for (const auto& corr : correspondences) {
            src_pts.push_back(source[corr.first]);
            tgt_pts.push_back(target[corr.second]);
        }

        return computeSVDTransform(src_pts, tgt_pts);
    }

    M4D computeSVDTransform(const std::vector<V3D>& source,
                            const std::vector<V3D>& target) {
        // Compute centroids
        V3D src_centroid = V3D::Zero();
        V3D tgt_centroid = V3D::Zero();

        for (size_t i = 0; i < source.size(); ++i) {
            src_centroid += source[i];
            tgt_centroid += target[i];
        }
        src_centroid /= static_cast<double>(source.size());
        tgt_centroid /= static_cast<double>(target.size());

        // Compute covariance matrix H
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < source.size(); ++i) {
            V3D src_centered = source[i] - src_centroid;
            V3D tgt_centered = target[i] - tgt_centroid;
            H += src_centered * tgt_centered.transpose();
        }

        // SVD decomposition
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        // Compute rotation
        Eigen::Matrix3d R = V * U.transpose();

        // Handle reflection case
        if (R.determinant() < 0) {
            V.col(2) *= -1;
            R = V * U.transpose();
        }

        // Compute translation
        V3D t = tgt_centroid - R * src_centroid;

        // Build 4x4 transformation
        M4D T = M4D::Identity();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = t;

        return T;
    }

    /**
     * @brief Point-to-plane transformation using linearized least squares
     *
     * Minimizes sum of (n_i . (R*p_i + t - q_i))^2
     * Using small angle approximation for rotation
     */
    M4D computePointToPlaneTransform(const std::vector<V3D>& source,
                                      const std::vector<V3D>& target,
                                      const std::vector<V3D>& normals) {
        // Build linear system Ax = b
        // x = [rx, ry, rz, tx, ty, tz]^T (small rotation angles + translation)
        Eigen::MatrixXd A(source.size(), 6);
        Eigen::VectorXd b(source.size());

        for (size_t i = 0; i < source.size(); ++i) {
            const V3D& p = source[i];
            const V3D& q = target[i];
            const V3D& n = normals[i];

            // Cross product term for rotation: (p x n)
            V3D pxn = p.cross(n);

            A(i, 0) = pxn.x();
            A(i, 1) = pxn.y();
            A(i, 2) = pxn.z();
            A(i, 3) = n.x();
            A(i, 4) = n.y();
            A(i, 5) = n.z();

            b(i) = n.dot(q - p);
        }

        // Solve using SVD (robust to rank deficiency)
        Eigen::VectorXd x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

        // Build transformation from small angles
        double rx = x(0), ry = x(1), rz = x(2);
        double tx = x(3), ty = x(4), tz = x(5);

        M3D R;
        R << 1,   -rz,  ry,
             rz,   1,  -rx,
            -ry,  rx,   1;

        // Orthonormalize R using SVD
        Eigen::JacobiSVD<M3D> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();

        M4D T = M4D::Identity();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = V3D(tx, ty, tz);

        return T;
    }

    /**
     * @brief GICP transformation using plane-to-plane error metric
     */
    M4D computeGICPTransform(const std::vector<PointWithNormal>& source,
                              const std::vector<PointWithNormal>& target,
                              const std::vector<size_t>& src_indices,
                              const std::vector<size_t>& tgt_indices,
                              const M4D& current_transform) {
        // Simplified GICP - use point-to-plane with averaged normals
        // Full GICP would use covariance matrices for weighting

        std::vector<V3D> src_pts, tgt_pts, avg_normals;
        M3D R = current_transform.block<3, 3>(0, 0);
        V3D t = current_transform.block<3, 1>(0, 3);

        for (size_t i = 0; i < src_indices.size(); ++i) {
            V3D p_trans = R * source[src_indices[i]].point + t;
            src_pts.push_back(p_trans);
            tgt_pts.push_back(target[tgt_indices[i]].point);

            // Average normals (simplified - full GICP uses covariance)
            V3D n_src_rotated = R * source[src_indices[i]].normal;
            V3D n_tgt = target[tgt_indices[i]].normal;
            V3D avg_n = (n_src_rotated + n_tgt).normalized();
            avg_normals.push_back(avg_n);
        }

        return computePointToPlaneTransform(src_pts, tgt_pts, avg_normals);
    }
};

//=============================================================================
// Multi-Scale ICP (Coarse-to-Fine)
//=============================================================================

class MultiScaleICP {
public:
    MultiScaleICP() = default;

    /**
     * @brief Perform multi-scale ICP alignment
     */
    ICPResult align(const std::vector<V3D>& source,
                    const std::vector<V3D>& target,
                    const M4D& initial_guess = M4D::Identity(),
                    const std::vector<double>& scales = {2.0, 1.0, 0.5},
                    ICPMethod method = ICPMethod::POINT_TO_POINT) {

        ICPResult result;
        result.transformation = initial_guess;
        M4D current_transform = initial_guess;

        for (double scale : scales) {
            // Downsample both clouds
            std::vector<V3D> src_down = voxelDownsample(source, scale);
            std::vector<V3D> tgt_down = voxelDownsample(target, scale);

            // Configure ICP for this scale
            ICPConfig config;
            config.method = method;
            config.max_correspondence_dist = scale * 3.0;
            config.max_iterations = 20;
            config.convergence_threshold = scale * 0.001;

            ICP icp(config);
            result = icp.align(src_down, tgt_down, current_transform);
            current_transform = result.transformation;

            if (!result.converged && result.fitness_score < 0.1) {
                break;
            }
        }

        return result;
    }

#ifdef HAS_PCL
    /**
     * @brief Multi-scale ICP with ikd-tree
     */
    template<typename PointType>
    ICPResult align(const std::vector<V3D>& source,
                    KD_TREE<PointType>& target_kdtree,
                    const M4D& initial_guess = M4D::Identity(),
                    const std::vector<double>& scales = {2.0, 1.0, 0.5}) {

        ICPResult result;
        result.transformation = initial_guess;
        M4D current_transform = initial_guess;

        for (double scale : scales) {
            std::vector<V3D> downsampled = voxelDownsample(source, scale);

            ICPConfig config;
            config.max_correspondence_dist = scale * 3.0;
            config.max_iterations = 20;
            config.convergence_threshold = scale * 0.001;

            ICP icp(config);
            result = icp.alignWithKdTree(downsampled, target_kdtree, current_transform);
            current_transform = result.transformation;

            if (!result.converged && result.fitness_score < 0.1) {
                break;
            }
        }

        return result;
    }
#endif // HAS_PCL
};

} // namespace slam

#endif // SLAM_ICP_HPP
