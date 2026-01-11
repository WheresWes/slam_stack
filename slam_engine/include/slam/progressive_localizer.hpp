/**
 * @file progressive_localizer.hpp
 * @brief Coverage-based progressive global localization using FAST-LIO
 *
 * This module implements a robust global localization system that:
 * 1. Lets FAST-LIO build a local map during accumulation (accurate odometry)
 * 2. Monitors coverage (voxel count, rotation) to determine readiness
 * 3. Runs global localization to find pose in pre-built map
 * 4. Swaps to pre-built map for continued localization-mode tracking
 *
 * Key insight: FAST-LIO's local map IS the accumulated geometry.
 * No need for separate accumulation - just monitor what FAST-LIO builds.
 */

#ifndef PROGRESSIVE_LOCALIZER_HPP
#define PROGRESSIVE_LOCALIZER_HPP

#include <vector>
#include <unordered_set>
#include <cmath>
#include <string>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "slam/types.hpp"
#include "slam/icp.hpp"

namespace slam {

//=============================================================================
// Voxel Key for spatial hashing
//=============================================================================

struct VoxelKey {
    int32_t x, y, z;

    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const {
        size_t h = 14695981039346656037ULL;
        h ^= static_cast<size_t>(k.x + 1048576);
        h *= 1099511628211ULL;
        h ^= static_cast<size_t>(k.y + 1048576);
        h *= 1099511628211ULL;
        h ^= static_cast<size_t>(k.z + 1048576);
        h *= 1099511628211ULL;
        return h;
    }
};

//=============================================================================
// Localization Status
//=============================================================================

enum class LocalizationStatus {
    NOT_STARTED,          // Haven't begun accumulation
    WAITING_IMU_INIT,     // Waiting for IMU initialization
    ACCUMULATING,         // FAST-LIO building local map
    VIEW_SATURATED,       // Current view exhausted, need to move
    READY_TO_ATTEMPT,     // Have enough coverage, attempting localization
    ATTEMPTING,           // Currently running global ICP
    LOW_CONFIDENCE,       // Attempted but confidence too low
    SUCCESS,              // Localization succeeded, ready for tracking
    FAILED                // Gave up after max attempts
};

inline const char* toString(LocalizationStatus status) {
    switch (status) {
        case LocalizationStatus::NOT_STARTED: return "NOT_STARTED";
        case LocalizationStatus::WAITING_IMU_INIT: return "WAITING_IMU_INIT";
        case LocalizationStatus::ACCUMULATING: return "ACCUMULATING";
        case LocalizationStatus::VIEW_SATURATED: return "VIEW_SATURATED";
        case LocalizationStatus::READY_TO_ATTEMPT: return "READY_TO_ATTEMPT";
        case LocalizationStatus::ATTEMPTING: return "ATTEMPTING";
        case LocalizationStatus::LOW_CONFIDENCE: return "LOW_CONFIDENCE";
        case LocalizationStatus::SUCCESS: return "SUCCESS";
        case LocalizationStatus::FAILED: return "FAILED";
        default: return "UNKNOWN";
    }
}

struct LocalizationResult {
    LocalizationStatus status = LocalizationStatus::NOT_STARTED;
    M4D pose = M4D::Identity();           // Final pose in pre-built map frame
    M4D transform = M4D::Identity();      // Transform from local to pre-built map
    double confidence = 0.0;
    std::string message;

    // Coverage stats (for UI feedback)
    int local_map_voxels = 0;             // Voxels in FAST-LIO's local map
    double rotation_deg = 0.0;            // Total rotation from trajectory
    double distance_m = 0.0;              // Total distance traveled
    int attempt_number = 0;
    int local_map_points = 0;
};

//=============================================================================
// Coverage Monitor - Monitors FAST-LIO's map building progress
//=============================================================================

/**
 * @brief Monitors FAST-LIO's local map building to determine localization readiness
 *
 * Instead of accumulating points ourselves, we monitor what FAST-LIO builds.
 * This gives us accurate, drift-free accumulated geometry.
 */
class CoverageMonitor {
public:
    // Coverage thresholds
    static constexpr double VOXEL_SIZE = 0.5;           // 50cm voxels
    static constexpr int MIN_VOXELS_FOR_ATTEMPT = 400;  // Minimum to attempt
    static constexpr int GOOD_VOXELS = 800;             // Good coverage
    static constexpr int IDEAL_VOXELS = 1200;           // Excellent coverage
    static constexpr double MIN_ROTATION_RAD = M_PI / 4;  // 45 degrees
    static constexpr double GOOD_ROTATION_RAD = M_PI / 2; // 90 degrees
    static constexpr int SATURATION_CHECKS = 5;         // Checks without new voxels
    static constexpr int MIN_NEW_VOXELS = 10;           // Threshold for "discovering new"

    CoverageMonitor() { reset(); }

    void reset() {
        current_voxels_.clear();
        last_voxel_count_ = 0;
        checks_since_new_geometry_ = 0;
        total_rotation_ = 0.0;
        total_distance_ = 0.0;
        last_position_ = V3D::Zero();
        last_rotation_ = M3D::Identity();
        first_update_ = true;
        check_count_ = 0;
    }

    /**
     * @brief Update coverage metrics from FAST-LIO's current state
     * @param map_points Current points in FAST-LIO's ikd-tree
     * @param trajectory Current trajectory (all poses)
     */
    void update(const std::vector<WorldPoint>& map_points,
                const std::vector<M4D>& trajectory) {
        check_count_++;

        // Count voxels in current map
        std::unordered_set<VoxelKey, VoxelKeyHash> new_voxels;
        for (const auto& pt : map_points) {
            new_voxels.insert(toVoxelKey(V3D(pt.x, pt.y, pt.z)));
        }

        // Check for new voxel discovery
        int new_voxel_count = static_cast<int>(new_voxels.size());
        int newly_discovered = new_voxel_count - last_voxel_count_;

        if (newly_discovered < MIN_NEW_VOXELS) {
            checks_since_new_geometry_++;
        } else {
            checks_since_new_geometry_ = 0;
        }

        last_voxel_count_ = new_voxel_count;
        current_voxels_ = std::move(new_voxels);

        // Compute total rotation and distance from trajectory
        if (!trajectory.empty()) {
            // Get current pose
            const M4D& current_pose = trajectory.back();
            V3D current_pos = current_pose.block<3, 1>(0, 3);
            M3D current_rot = current_pose.block<3, 3>(0, 0);

            if (!first_update_) {
                // Accumulate distance
                total_distance_ += (current_pos - last_position_).norm();

                // Accumulate rotation (angle of relative rotation)
                M3D delta_rot = last_rotation_.transpose() * current_rot;
                Eigen::AngleAxisd aa(delta_rot);
                total_rotation_ += std::abs(aa.angle());
            }

            last_position_ = current_pos;
            last_rotation_ = current_rot;
            first_update_ = false;
        }
    }

    /**
     * @brief Check if we have enough coverage to attempt localization
     */
    bool isReadyForAttempt() const {
        int voxels = static_cast<int>(current_voxels_.size());

        // Strong voxel coverage alone
        if (voxels >= IDEAL_VOXELS) return true;

        // Good voxels + good rotation
        if (voxels >= GOOD_VOXELS && total_rotation_ >= MIN_ROTATION_RAD) return true;

        // Minimum voxels + excellent rotation
        if (voxels >= MIN_VOXELS_FOR_ATTEMPT && total_rotation_ >= GOOD_ROTATION_RAD) return true;

        // Lots of rotation compensates for fewer voxels
        if (total_rotation_ >= M_PI && voxels >= 300) return true;

        return false;
    }

    /**
     * @brief Check if current view is saturated (no new geometry being discovered)
     */
    bool isViewSaturated() const {
        return checks_since_new_geometry_ >= SATURATION_CHECKS;
    }

    int getVoxelCount() const { return static_cast<int>(current_voxels_.size()); }
    double getRotationRad() const { return total_rotation_; }
    double getRotationDeg() const { return total_rotation_ * 180.0 / M_PI; }
    double getDistanceM() const { return total_distance_; }
    int getChecksSinceNewGeometry() const { return checks_since_new_geometry_; }

    std::string getStatusMessage() const {
        int voxels = static_cast<int>(current_voxels_.size());
        double rot_deg = total_rotation_ * 180.0 / M_PI;

        if (voxels < 50) {
            return "Starting - move LiDAR to begin scanning...";
        }

        if (isViewSaturated() && !isReadyForAttempt()) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "View saturated (%d voxels, %.0f deg) - rotate or move LiDAR",
                     voxels, rot_deg);
            return std::string(buf);
        }

        if (isReadyForAttempt()) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "Ready: %d voxels, %.0f deg - attempting localization...",
                     voxels, rot_deg);
            return std::string(buf);
        }

        // Progress indicator
        int target = (total_rotation_ < MIN_ROTATION_RAD) ? GOOD_VOXELS : IDEAL_VOXELS;
        int pct = std::min(100, voxels * 100 / target);

        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Accumulating: %d voxels, %.0f deg rotation [%d%%]",
                 voxels, rot_deg, pct);
        return std::string(buf);
    }

private:
    VoxelKey toVoxelKey(const V3D& pt) const {
        return VoxelKey{
            static_cast<int32_t>(std::floor(pt.x() / VOXEL_SIZE)),
            static_cast<int32_t>(std::floor(pt.y() / VOXEL_SIZE)),
            static_cast<int32_t>(std::floor(pt.z() / VOXEL_SIZE))
        };
    }

    std::unordered_set<VoxelKey, VoxelKeyHash> current_voxels_;
    int last_voxel_count_ = 0;
    int checks_since_new_geometry_ = 0;
    double total_rotation_ = 0.0;
    double total_distance_ = 0.0;
    V3D last_position_;
    M3D last_rotation_;
    bool first_update_ = true;
    int check_count_ = 0;
};

//=============================================================================
// Voxel Occupancy Map for Fast Hypothesis Scoring
//=============================================================================

class VoxelOccupancyMap {
public:
    static constexpr double VOXEL_SIZE = 0.5;

    void build(const std::vector<V3D>& points) {
        occupied_.clear();
        for (const auto& pt : points) {
            occupied_.insert(toKey(pt));
        }
    }

    void build(const std::vector<WorldPoint>& points) {
        occupied_.clear();
        for (const auto& pt : points) {
            occupied_.insert(toKey(V3D(pt.x, pt.y, pt.z)));
        }
    }

    double scoreHypothesis(const std::vector<V3D>& points, const M4D& pose) const {
        if (points.empty() || occupied_.empty()) return 0.0;

        M3D R = pose.block<3, 3>(0, 0);
        V3D t = pose.block<3, 1>(0, 3);

        int hits = 0;
        for (const auto& pt : points) {
            V3D pt_world = R * pt + t;
            if (occupied_.count(toKey(pt_world)) > 0) {
                hits++;
            }
        }

        return static_cast<double>(hits) / static_cast<double>(points.size());
    }

    size_t size() const { return occupied_.size(); }

private:
    VoxelKey toKey(const V3D& pt) const {
        return VoxelKey{
            static_cast<int32_t>(std::floor(pt.x() / VOXEL_SIZE)),
            static_cast<int32_t>(std::floor(pt.y() / VOXEL_SIZE)),
            static_cast<int32_t>(std::floor(pt.z() / VOXEL_SIZE))
        };
    }

    std::unordered_set<VoxelKey, VoxelKeyHash> occupied_;
};

//=============================================================================
// Progressive Global Localizer Configuration
//=============================================================================

struct ProgressiveLocalizerConfig {
    // Confidence thresholds
    double min_confidence = 0.45;      // Minimum to accept
    double high_confidence = 0.65;     // High confidence - definitely accept
    int max_attempts = 10;             // Max localization attempts

    // Grid search parameters (for global localization)
    double grid_step = 2.0;            // meters
    double yaw_step_deg = 30.0;        // degrees
    bool search_z = false;             // Search in Z (usually false for ground robots)
    double z_value = 0.0;              // Fixed Z value if not searching

    // ICP parameters
    double coarse_voxel = 0.5;
    double medium_voxel = 0.2;
    double fine_voxel = 0.1;
};

//=============================================================================
// Progressive Global Localizer
//=============================================================================

/**
 * @brief Progressive global localizer that uses FAST-LIO for accumulation
 *
 * Workflow:
 * 1. Store pre-built map in memory
 * 2. Let FAST-LIO run in SLAM mode, building a local map
 * 3. Monitor coverage (voxels, rotation) via CoverageMonitor
 * 4. When ready, run global localization (local map → pre-built map)
 * 5. On success, return transform to apply
 *
 * The caller (SlamEngine) is responsible for:
 * - Swapping the ikd-tree to use pre-built map
 * - Applying the transform to current state
 * - Enabling localization mode
 */
class ProgressiveGlobalLocalizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ProgressiveGlobalLocalizer() = default;

    explicit ProgressiveGlobalLocalizer(const ProgressiveLocalizerConfig& config)
        : config_(config) {}

    /**
     * @brief Set the pre-built map for localization
     */
    void setPrebuiltMap(const std::vector<WorldPoint>& map_points) {
        prebuilt_map_points_.clear();
        prebuilt_map_points_.reserve(map_points.size());
        for (const auto& pt : map_points) {
            prebuilt_map_points_.emplace_back(pt.x, pt.y, pt.z);
        }

        prebuilt_voxels_.build(prebuilt_map_points_);

        // Compute bounds
        prebuilt_min_ = V3D(1e9, 1e9, 1e9);
        prebuilt_max_ = V3D(-1e9, -1e9, -1e9);
        for (const auto& pt : prebuilt_map_points_) {
            prebuilt_min_ = prebuilt_min_.cwiseMin(pt);
            prebuilt_max_ = prebuilt_max_.cwiseMax(pt);
        }

        std::cout << "[ProgressiveLocalizer] Pre-built map set: "
                  << prebuilt_map_points_.size() << " points" << std::endl;
        std::cout << "  Bounds: [" << prebuilt_min_.transpose() << "] to ["
                  << prebuilt_max_.transpose() << "]" << std::endl;
    }

    /**
     * @brief Reset for a new localization attempt
     */
    void reset() {
        coverage_monitor_.reset();
        status_ = LocalizationStatus::NOT_STARTED;
        attempt_count_ = 0;
        best_transform_ = M4D::Identity();
        best_confidence_ = 0.0;
    }

    /**
     * @brief Check coverage and attempt localization if ready
     *
     * Call this periodically (e.g., after each SLAM cycle) with FAST-LIO's
     * current local map and trajectory.
     *
     * @param local_map_points Points from FAST-LIO's ikd-tree
     * @param trajectory FAST-LIO's trajectory (all poses)
     * @param current_pose Current FAST-LIO pose
     * @return Localization result with status and transform if successful
     */
    LocalizationResult checkAndLocalize(
        const std::vector<WorldPoint>& local_map_points,
        const std::vector<M4D>& trajectory,
        const M4D& current_pose) {

        LocalizationResult result;

        if (status_ == LocalizationStatus::SUCCESS) {
            result.status = LocalizationStatus::SUCCESS;
            result.transform = best_transform_;
            result.pose = best_transform_ * current_pose;
            result.confidence = best_confidence_;
            result.message = "Already localized";
            return result;
        }

        // Update coverage metrics
        coverage_monitor_.update(local_map_points, trajectory);

        result.local_map_voxels = coverage_monitor_.getVoxelCount();
        result.rotation_deg = coverage_monitor_.getRotationDeg();
        result.distance_m = coverage_monitor_.getDistanceM();
        result.local_map_points = static_cast<int>(local_map_points.size());
        result.attempt_number = attempt_count_;

        // Check for view saturation
        if (coverage_monitor_.isViewSaturated() && !coverage_monitor_.isReadyForAttempt()) {
            status_ = LocalizationStatus::VIEW_SATURATED;
            result.status = status_;
            result.message = coverage_monitor_.getStatusMessage();
            return result;
        }

        // Check if ready to attempt
        if (coverage_monitor_.isReadyForAttempt()) {
            status_ = LocalizationStatus::ATTEMPTING;
            attempt_count_++;
            result.attempt_number = attempt_count_;

            std::cout << "\n[ProgressiveLocalizer] Attempting global localization..."
                      << std::endl;
            std::cout << "  Local map: " << local_map_points.size() << " points, "
                      << result.local_map_voxels << " voxels" << std::endl;
            std::cout << "  Coverage: " << result.rotation_deg << " deg rotation, "
                      << result.distance_m << " m traveled" << std::endl;

            // Convert local map to V3D
            std::vector<V3D> local_points;
            local_points.reserve(local_map_points.size());
            for (const auto& pt : local_map_points) {
                local_points.emplace_back(pt.x, pt.y, pt.z);
            }

            // Run global localization
            auto [transform, confidence] = attemptGlobalLocalization(local_points);

            result.transform = transform;
            result.confidence = confidence;
            result.pose = transform * current_pose;

            std::cout << "  Result: " << (confidence * 100) << "% confidence"
                      << std::endl;

            if (confidence >= config_.high_confidence) {
                status_ = LocalizationStatus::SUCCESS;
                best_transform_ = transform;
                best_confidence_ = confidence;
                result.status = status_;
                result.message = "Localized with high confidence";
                std::cout << "  >>> SUCCESS (high confidence) <<<" << std::endl;
                return result;
            }

            if (confidence >= config_.min_confidence) {
                // Accept with moderate confidence if we have good coverage
                if (result.local_map_voxels >= CoverageMonitor::GOOD_VOXELS ||
                    result.rotation_deg >= 90.0) {
                    status_ = LocalizationStatus::SUCCESS;
                    best_transform_ = transform;
                    best_confidence_ = confidence;
                    result.status = status_;
                    result.message = "Localized (moderate confidence)";
                    std::cout << "  >>> SUCCESS (moderate confidence) <<<" << std::endl;
                    return result;
                }
            }

            // Not confident enough
            if (attempt_count_ >= config_.max_attempts) {
                status_ = LocalizationStatus::FAILED;
                result.status = status_;
                result.message = "Max attempts reached - localization failed";
                std::cout << "  >>> FAILED (max attempts) <<<" << std::endl;
                return result;
            }

            // Keep best attempt
            if (confidence > best_confidence_) {
                best_transform_ = transform;
                best_confidence_ = confidence;
            }

            status_ = LocalizationStatus::LOW_CONFIDENCE;
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "Attempt %d: %.0f%% confidence - continue scanning...",
                     attempt_count_, confidence * 100);
            result.status = status_;
            result.message = std::string(buf);
            return result;
        }

        // Still accumulating
        status_ = LocalizationStatus::ACCUMULATING;
        result.status = status_;
        result.message = coverage_monitor_.getStatusMessage();
        return result;
    }

    LocalizationStatus getStatus() const { return status_; }
    const CoverageMonitor& getCoverageMonitor() const { return coverage_monitor_; }
    M4D getBestTransform() const { return best_transform_; }
    double getBestConfidence() const { return best_confidence_; }
    int getAttemptCount() const { return attempt_count_; }

private:
    /**
     * @brief Run global localization: find transform from local map to pre-built map
     */
    std::pair<M4D, double> attemptGlobalLocalization(const std::vector<V3D>& local_map) {
        if (local_map.empty() || prebuilt_map_points_.empty()) {
            return {M4D::Identity(), 0.0};
        }

        // Downsample for hypothesis generation
        auto local_coarse = voxelDownsample(local_map, config_.coarse_voxel);

        std::cout << "  Generating hypotheses..." << std::flush;

        // Generate hypotheses via grid search
        std::vector<std::pair<M4D, double>> hypotheses;

        double yaw_step = config_.yaw_step_deg * M_PI / 180.0;
        int num_yaws = static_cast<int>(std::ceil(2 * M_PI / yaw_step));

        // Determine Z value
        double z_val = config_.search_z ?
            (prebuilt_min_.z() + prebuilt_max_.z()) * 0.5 :
            config_.z_value;

        // Grid search over pre-built map bounds
        for (double x = prebuilt_min_.x(); x <= prebuilt_max_.x(); x += config_.grid_step) {
            for (double y = prebuilt_min_.y(); y <= prebuilt_max_.y(); y += config_.grid_step) {
                for (int yi = 0; yi < num_yaws; yi++) {
                    double yaw = yi * yaw_step;

                    M4D pose = M4D::Identity();
                    pose(0, 0) = std::cos(yaw);
                    pose(0, 1) = -std::sin(yaw);
                    pose(1, 0) = std::sin(yaw);
                    pose(1, 1) = std::cos(yaw);
                    pose(0, 3) = x;
                    pose(1, 3) = y;
                    pose(2, 3) = z_val;

                    double score = prebuilt_voxels_.scoreHypothesis(local_coarse, pose);
                    if (score > 0.1) {  // Only keep plausible hypotheses
                        hypotheses.emplace_back(pose, score);
                    }
                }
            }
        }

        std::cout << " " << hypotheses.size() << " candidates" << std::endl;

        if (hypotheses.empty()) {
            std::cout << "  No plausible hypotheses found!" << std::endl;
            return {M4D::Identity(), 0.0};
        }

        // Sort by score descending
        std::sort(hypotheses.begin(), hypotheses.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        std::cout << "  Top hypothesis score: " << (hypotheses[0].second * 100) << "%"
                  << std::endl;

        // Refine top hypothesis with coarse-to-fine ICP
        M4D best_pose = hypotheses[0].first;

        // Coarse ICP
        std::cout << "  Coarse ICP..." << std::flush;
        {
            auto scan_down = voxelDownsample(local_map, config_.coarse_voxel);
            auto map_down = voxelDownsample(prebuilt_map_points_, config_.coarse_voxel);

            ICPConfig cfg;
            cfg.max_iterations = 30;
            cfg.max_correspondence_dist = 3.0;
            cfg.convergence_threshold = 1e-4;

            ICP icp(cfg);
            auto result = icp.align(scan_down, map_down, best_pose);
            best_pose = result.transformation;
            std::cout << " fitness=" << (result.fitness_score * 100) << "%" << std::endl;
        }

        // Medium ICP
        std::cout << "  Medium ICP..." << std::flush;
        {
            auto scan_down = voxelDownsample(local_map, config_.medium_voxel);
            auto map_down = voxelDownsample(prebuilt_map_points_, config_.medium_voxel);

            ICPConfig cfg;
            cfg.max_iterations = 40;
            cfg.max_correspondence_dist = 1.5;
            cfg.convergence_threshold = 1e-5;

            ICP icp(cfg);
            auto result = icp.align(scan_down, map_down, best_pose);
            best_pose = result.transformation;
            std::cout << " fitness=" << (result.fitness_score * 100) << "%" << std::endl;
        }

        // Fine ICP
        std::cout << "  Fine ICP..." << std::flush;
        double final_confidence;
        {
            auto scan_down = voxelDownsample(local_map, config_.fine_voxel);
            auto map_down = voxelDownsample(prebuilt_map_points_, config_.fine_voxel);

            ICPConfig cfg;
            cfg.max_iterations = 50;
            cfg.max_correspondence_dist = 0.5;
            cfg.convergence_threshold = 1e-6;

            ICP icp(cfg);
            auto result = icp.align(scan_down, map_down, best_pose);
            best_pose = result.transformation;
            final_confidence = result.fitness_score;
            std::cout << " fitness=" << (result.fitness_score * 100) << "%" << std::endl;
        }

        return {best_pose, final_confidence};
    }

    ProgressiveLocalizerConfig config_;
    CoverageMonitor coverage_monitor_;
    VoxelOccupancyMap prebuilt_voxels_;
    std::vector<V3D> prebuilt_map_points_;
    V3D prebuilt_min_, prebuilt_max_;

    LocalizationStatus status_ = LocalizationStatus::NOT_STARTED;
    int attempt_count_ = 0;
    M4D best_transform_ = M4D::Identity();
    double best_confidence_ = 0.0;
};

} // namespace slam

#endif // PROGRESSIVE_LOCALIZER_HPP
