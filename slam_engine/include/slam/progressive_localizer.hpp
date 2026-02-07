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
#include <functional>
#include <atomic>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "slam/types.hpp"
#include "slam/icp.hpp"

// Debug log file for SC diagnostics
inline std::ofstream& getSCLogFile() {
    static std::ofstream log_file("sc_debug.log", std::ios::out | std::ios::trunc);
    return log_file;
}

#define SC_LOG(msg) do { \
    std::cout << msg << std::endl; \
    getSCLogFile() << msg << std::endl; \
    getSCLogFile().flush(); \
} while(0)

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
    READY_FOR_LOCALIZATION, // Have enough coverage, ready to run ICP (robot should stop)
    AWAITING_ROBOT_STOP,  // Coverage met, waiting for robot to stop before ICP
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
        case LocalizationStatus::READY_FOR_LOCALIZATION: return "READY_FOR_LOCALIZATION";
        case LocalizationStatus::AWAITING_ROBOT_STOP: return "AWAITING_ROBOT_STOP";
        case LocalizationStatus::ATTEMPTING: return "ATTEMPTING";
        case LocalizationStatus::LOW_CONFIDENCE: return "LOW_CONFIDENCE";
        case LocalizationStatus::SUCCESS: return "SUCCESS";
        case LocalizationStatus::FAILED: return "FAILED";
        default: return "UNKNOWN";
    }
}

//=============================================================================
// Localization Progress - For UI feedback during ICP
//=============================================================================

enum class LocalizationStage {
    IDLE,
    GENERATING_HYPOTHESES,
    COARSE_ICP,
    MEDIUM_ICP,
    FINE_ICP,
    EVALUATING,
    COMPLETE
};

inline const char* toString(LocalizationStage stage) {
    switch (stage) {
        case LocalizationStage::IDLE: return "Idle";
        case LocalizationStage::GENERATING_HYPOTHESES: return "Generating hypotheses...";
        case LocalizationStage::COARSE_ICP: return "Coarse alignment...";
        case LocalizationStage::MEDIUM_ICP: return "Medium refinement...";
        case LocalizationStage::FINE_ICP: return "Fine alignment...";
        case LocalizationStage::EVALUATING: return "Evaluating confidence...";
        case LocalizationStage::COMPLETE: return "Complete";
        default: return "Unknown";
    }
}

struct LocalizationProgress {
    LocalizationStage stage = LocalizationStage::IDLE;
    float progress = 0.0f;       // 0.0 to 1.0 overall progress
    float stage_progress = 0.0f; // 0.0 to 1.0 within current stage
    std::string message;

    // ICP metrics (updated during refinement)
    int hypotheses_count = 0;
    int hypotheses_kept = 0;
    double best_score = 0.0;
    double current_fitness = 0.0;
    int current_iteration = 0;
    int max_iterations = 0;
};

// Progress callback type
using LocalizationProgressCallback = std::function<void(const LocalizationProgress&)>;

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
// Scan Context Configuration
//=============================================================================

/**
 * @brief Configuration for Scan Context descriptor
 *
 * Scan Context encodes a point cloud as a 2D polar representation for
 * fast place recognition. Optimized for Mid-360 LiDAR characteristics.
 */
struct ScanContextConfig {
    int num_rings = 20;              // Radial divisions
    int num_sectors = 60;            // Angular divisions (6 degree resolution)
    double max_radius = 40.0;        // Maximum range (Mid-360: 40m)
    double min_radius = 1.0;         // Minimum range (ignore close points)
    int num_height_bands = 5;        // Vertical divisions for 3D descriptor
    double min_height = -2.0;        // Minimum height (below sensor)
    double max_height = 15.0;        // Maximum height (above sensor)
};

//=============================================================================
// Scan Context Descriptor
//=============================================================================

/**
 * @brief Scan Context descriptor for fast place recognition
 *
 * Encodes a point cloud as a 2D polar representation with optional
 * height bands for 3D structure. Provides rotation-invariant matching
 * by searching over column shifts (corresponding to yaw rotations).
 *
 * Key insight: Two scans from similar locations will have similar
 * polar occupancy patterns, differing mainly by a column shift (rotation).
 */
class ScanContext {
public:
    using Descriptor = Eigen::MatrixXd;  // (num_rings × num_sectors × num_bands)
    using RingKey = Eigen::VectorXd;     // 1D summary for fast filtering

    explicit ScanContext(const ScanContextConfig& config = ScanContextConfig())
        : config_(config) {
        // Pre-compute ring boundaries (logarithmic spacing for better near-field resolution)
        ring_boundaries_.resize(config_.num_rings + 1);
        double log_min = std::log(config_.min_radius);
        double log_max = std::log(config_.max_radius);
        for (int i = 0; i <= config_.num_rings; i++) {
            double t = static_cast<double>(i) / config_.num_rings;
            ring_boundaries_[i] = std::exp(log_min + t * (log_max - log_min));
        }
    }

    /**
     * @brief Compute descriptor from point cloud
     * @param cloud Input point cloud in sensor frame
     * @return Scan Context descriptor matrix
     */
    Descriptor compute(const std::vector<V3D>& cloud) const {
        if (config_.num_height_bands > 1) {
            return compute3D(cloud);
        } else {
            return compute2D(cloud);
        }
    }

    /**
     * @brief Compute ring-key for fast candidate filtering
     * @param desc Full descriptor
     * @return 1D ring key (mean of each ring)
     */
    RingKey computeRingKey(const Descriptor& desc) const {
        return desc.rowwise().mean();
    }

    /**
     * @brief Match two descriptors with rotation search
     * @param query Query descriptor
     * @param candidate Candidate descriptor from database
     * @return pair<similarity_score, column_shift>
     */
    std::pair<double, int> match(const Descriptor& query,
                                  const Descriptor& candidate) const {
        double best_score = -1.0;
        int best_shift = 0;

        int num_cols = static_cast<int>(query.cols());
        int sector_cols = config_.num_sectors;

        for (int shift = 0; shift < sector_cols; shift++) {
            // Circular shift the candidate
            Descriptor shifted(candidate.rows(), candidate.cols());

            if (config_.num_height_bands > 1) {
                // 3D descriptor: shift within each height band
                for (int band = 0; band < config_.num_height_bands; band++) {
                    for (int s = 0; s < sector_cols; s++) {
                        int src_col = band * sector_cols + ((s + shift) % sector_cols);
                        int dst_col = band * sector_cols + s;
                        shifted.col(dst_col) = candidate.col(src_col);
                    }
                }
            } else {
                // 2D descriptor: simple column shift
                for (int c = 0; c < num_cols; c++) {
                    shifted.col(c) = candidate.col((c + shift) % num_cols);
                }
            }

            double score = cosineSimilarity(query, shifted);
            if (score > best_score) {
                best_score = score;
                best_shift = shift;
            }
        }

        return {best_score, best_shift};
    }

    /**
     * @brief Convert column shift to yaw angle
     */
    double shiftToYaw(int shift) const {
        return shift * (2.0 * M_PI / config_.num_sectors);
    }

    const ScanContextConfig& config() const { return config_; }

private:
    ScanContextConfig config_;
    std::vector<double> ring_boundaries_;

    /**
     * @brief Standard 2D Scan Context (max height per cell)
     */
    Descriptor compute2D(const std::vector<V3D>& cloud) const {
        Descriptor desc = Descriptor::Zero(config_.num_rings, config_.num_sectors);

        for (const auto& pt : cloud) {
            double range = std::sqrt(pt.x() * pt.x() + pt.y() * pt.y());

            if (range < config_.min_radius || range > config_.max_radius) continue;

            int ring = findRing(range);
            if (ring < 0 || ring >= config_.num_rings) continue;

            double angle = std::atan2(pt.y(), pt.x());  // -π to π
            int sector = static_cast<int>((angle + M_PI) / (2.0 * M_PI) * config_.num_sectors);
            sector = std::clamp(sector, 0, config_.num_sectors - 1);

            desc(ring, sector) = std::max(desc(ring, sector), pt.z());
        }

        // Debug: log descriptor stats
        int nonzero = (desc.array() > 0).count();
        double norm = Eigen::Map<const Eigen::VectorXd>(desc.data(), desc.size()).norm();
        SC_LOG("    [SC compute2D] cloud=" << cloud.size() << " pts, desc "
               << desc.rows() << "x" << desc.cols() << " nonzero=" << nonzero
               << " norm=" << std::fixed << std::setprecision(2) << norm);

        return desc;
    }

    /**
     * @brief 3D Scan Context with height bands (better for outdoor/hull environments)
     */
    Descriptor compute3D(const std::vector<V3D>& cloud) const {
        int total_cols = config_.num_sectors * config_.num_height_bands;
        Descriptor desc = Descriptor::Zero(config_.num_rings, total_cols);

        Eigen::MatrixXi counts = Eigen::MatrixXi::Zero(config_.num_rings, total_cols);

        double height_range = config_.max_height - config_.min_height;
        double band_height = height_range / config_.num_height_bands;

        for (const auto& pt : cloud) {
            double range = std::sqrt(pt.x() * pt.x() + pt.y() * pt.y());

            if (range < config_.min_radius || range > config_.max_radius) continue;
            if (pt.z() < config_.min_height || pt.z() > config_.max_height) continue;

            int ring = findRing(range);
            if (ring < 0 || ring >= config_.num_rings) continue;

            double angle = std::atan2(pt.y(), pt.x());
            int sector = static_cast<int>((angle + M_PI) / (2.0 * M_PI) * config_.num_sectors);
            sector = std::clamp(sector, 0, config_.num_sectors - 1);

            int band = static_cast<int>((pt.z() - config_.min_height) / band_height);
            band = std::clamp(band, 0, config_.num_height_bands - 1);

            int col = band * config_.num_sectors + sector;
            counts(ring, col)++;
        }

        // Normalize counts to 0-1 range
        double max_count = counts.maxCoeff();
        if (max_count > 0) {
            desc = counts.cast<double>() / max_count;
        }

        // Debug: log descriptor stats
        int nonzero = (desc.array() > 0).count();
        double norm = Eigen::Map<const Eigen::VectorXd>(desc.data(), desc.size()).norm();
        SC_LOG("    [SC compute3D] cloud=" << cloud.size() << " pts, desc "
               << desc.rows() << "x" << desc.cols() << " nonzero=" << nonzero
               << " max=" << max_count << " norm=" << std::fixed << std::setprecision(2) << norm);

        return desc;
    }

    int findRing(double range) const {
        auto it = std::lower_bound(ring_boundaries_.begin(), ring_boundaries_.end(), range);
        int idx = static_cast<int>(it - ring_boundaries_.begin()) - 1;
        return std::clamp(idx, 0, config_.num_rings - 1);
    }

    double cosineSimilarity(const Descriptor& a, const Descriptor& b) const {
        Eigen::VectorXd va = Eigen::Map<const Eigen::VectorXd>(a.data(), a.size());
        Eigen::VectorXd vb = Eigen::Map<const Eigen::VectorXd>(b.data(), b.size());

        double dot = va.dot(vb);
        double norm_a = va.norm();
        double norm_b = vb.norm();

        if (norm_a < 1e-10 || norm_b < 1e-10) return 0.0;

        return dot / (norm_a * norm_b);
    }
};

//=============================================================================
// Scan Context Database
//=============================================================================

/**
 * @brief Database of Scan Context descriptors for place recognition
 *
 * Stores keyframe descriptors with spatial indexing. Supports fast
 * querying using ring-key pre-filtering followed by full descriptor matching.
 */
class ScanContextDatabase {
public:
    struct KeyFrame {
        int id;
        V3D position;
        double yaw;
        ScanContext::Descriptor descriptor;
        ScanContext::RingKey ring_key;
    };

    struct Match {
        int keyframe_id;
        V3D position;
        double yaw_estimate;
        double score;
    };

    explicit ScanContextDatabase(const ScanContextConfig& config = ScanContextConfig())
        : sc_(config), config_(config), min_keyframe_distance_(5.0) {}

    /**
     * @brief Add a keyframe to the database
     * @param scan Point cloud in LOCAL/SENSOR frame (relative to keyframe position)
     * @param pose Pose of this keyframe in world frame
     * @return true if added (false if too close to existing keyframe)
     */
    bool addKeyFrame(const std::vector<V3D>& scan, const M4D& pose) {
        V3D position(pose(0, 3), pose(1, 3), pose(2, 3));

        // Check if too close to existing keyframe
        for (const auto& kf : keyframes_) {
            if ((position - kf.position).head<2>().norm() < min_keyframe_distance_) {
                return false;
            }
        }

        if (scan.size() < 100) {
            return false;  // Not enough points
        }

        // Compute descriptor from sensor-frame scan
        KeyFrame kf;
        kf.id = static_cast<int>(keyframes_.size());
        kf.position = position;
        kf.yaw = std::atan2(pose(1, 0), pose(0, 0));
        kf.descriptor = sc_.compute(scan);
        kf.ring_key = sc_.computeRingKey(kf.descriptor);

        keyframes_.push_back(kf);
        return true;
    }

    /**
     * @brief Query database for matching locations
     * @param scan Query scan in sensor frame
     * @param top_k Number of candidates to return
     * @return Vector of matches sorted by score (descending)
     */
    std::vector<Match> query(const std::vector<V3D>& scan, int top_k = 20) const {
        SC_LOG("  [SC query] scan=" << scan.size() << " pts, keyframes=" << keyframes_.size());
        if (keyframes_.empty() || scan.size() < 100) {
            SC_LOG("  [SC query] EARLY EXIT: empty keyframes or scan too small");
            return {};
        }

        // Compute query descriptor
        SC_LOG("  [SC query] Computing query descriptor...");
        auto query_desc = sc_.compute(scan);
        auto query_ring_key = sc_.computeRingKey(query_desc);

        // Log query descriptor stats
        int q_nonzero = (query_desc.array() > 0).count();
        double q_norm = Eigen::Map<const Eigen::VectorXd>(query_desc.data(), query_desc.size()).norm();
        SC_LOG("  [SC query] Query desc: " << query_desc.rows() << "x" << query_desc.cols()
               << " nonzero=" << q_nonzero << " norm=" << std::fixed << std::setprecision(2) << q_norm);

        // Pre-filter with ring key distance (fast)
        std::vector<std::pair<int, double>> ring_distances;
        ring_distances.reserve(keyframes_.size());

        for (const auto& kf : keyframes_) {
            double dist = (query_ring_key - kf.ring_key).norm();
            ring_distances.push_back({kf.id, dist});
        }

        // Sort by ring distance (ascending)
        std::sort(ring_distances.begin(), ring_distances.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });

        // Full matching on top candidates (3x top_k for filtering margin)
        int num_to_match = std::min(static_cast<int>(ring_distances.size()), top_k * 3);
        std::vector<Match> matches;
        matches.reserve(num_to_match);

        for (int i = 0; i < num_to_match; i++) {
            const auto& kf = keyframes_[ring_distances[i].first];

            auto [score, shift] = sc_.match(query_desc, kf.descriptor);

            // Log first few matches for debugging
            if (i < 5) {
                int kf_nonzero = (kf.descriptor.array() > 0).count();
                double kf_norm = Eigen::Map<const Eigen::VectorXd>(kf.descriptor.data(), kf.descriptor.size()).norm();
                SC_LOG("  [SC match " << i << "] kf=" << kf.id << " kf_nonzero=" << kf_nonzero
                       << " kf_norm=" << std::fixed << std::setprecision(2) << kf_norm
                       << " => score=" << std::setprecision(4) << score << " shift=" << shift);
            }

            double yaw_offset = sc_.shiftToYaw(shift);
            double estimated_yaw = kf.yaw + yaw_offset;

            // Normalize to [-π, π]
            while (estimated_yaw > M_PI) estimated_yaw -= 2.0 * M_PI;
            while (estimated_yaw < -M_PI) estimated_yaw += 2.0 * M_PI;

            matches.push_back({kf.id, kf.position, estimated_yaw, score});
        }

        // Sort by score (descending)
        std::sort(matches.begin(), matches.end(),
                 [](const Match& a, const Match& b) { return a.score > b.score; });

        // Return top_k
        if (matches.size() > static_cast<size_t>(top_k)) {
            matches.resize(top_k);
        }

        return matches;
    }

    size_t size() const { return keyframes_.size(); }
    bool empty() const { return keyframes_.empty(); }
    void clear() { keyframes_.clear(); }

    void setMinKeyFrameDistance(double dist) { min_keyframe_distance_ = dist; }
    double getMinKeyFrameDistance() const { return min_keyframe_distance_; }

    const KeyFrame& getKeyFrame(int id) const { return keyframes_[id]; }

private:
    ScanContext sc_;
    ScanContextConfig config_;
    std::vector<KeyFrame> keyframes_;
    double min_keyframe_distance_;
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
    double grid_step = 2.0;            // meters (1.0m when using hint for finer search)
    double yaw_step_deg = 30.0;        // degrees (15° when using hint for finer search)
    bool search_z = false;             // Search in Z (usually false for ground robots)
    double z_value = 0.0;              // Fixed Z value if not searching

    // Position hint for bounded search (CRITICAL for reliable localization)
    bool use_hint = false;             // If true, search only within hint bounds
    double hint_x = 0.0;               // Hint position X (map frame)
    double hint_y = 0.0;               // Hint position Y (map frame)
    double hint_radius = 10.0;         // Search radius around hint (meters)
    double hint_heading = 0.0;         // Heading direction (radians, 0 = +X)
    bool hint_heading_known = false;   // If true, search only ±60° around heading
    double hint_heading_range = M_PI / 3.0;  // ±60° if heading known

    // ICP parameters
    double coarse_voxel = 0.5;
    double medium_voxel = 0.2;
    double fine_voxel = 0.1;

    // Scan Context parameters (fast place recognition before ICP)
    // NOTE: SC disabled - not suitable for outdoor dry dock scenario with partial views
    // Grid search + multi-hypothesis ICP works better with surrounding dock structure
    bool use_scan_context = false;         // Disabled - use grid search instead
    ScanContextConfig sc_config;           // Descriptor configuration
    int sc_top_candidates = 50;            // Max candidates from SC query
    double sc_min_score = 0.05;            // Minimum SC match score (lowered for testing)
    double sc_confident_score = 0.3;       // High-confidence SC match
    double keyframe_spacing = 5.0;         // Meters between keyframes in database
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
     * @brief Set configuration
     */
    void setConfig(const ProgressiveLocalizerConfig& config) {
        config_ = config;
    }

    /**
     * @brief Cancel any in-progress localization attempt
     */
    void cancelLocalization() {
        cancel_requested_.store(true);
    }

    /**
     * @brief Check if cancellation was requested
     */
    bool isCancellationRequested() const {
        return cancel_requested_.load();
    }

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

        // Build Scan Context keyframe database if enabled
        if (config_.use_scan_context) {
            buildScanContextDatabase();
        }
    }

    /**
     * @brief Build Scan Context keyframe database from pre-built map
     *
     * Samples keyframe positions across the map and computes descriptors
     * for each. This enables fast place recognition before ICP.
     */
    void buildScanContextDatabase() {
        sc_database_ = ScanContextDatabase(config_.sc_config);
        sc_database_.setMinKeyFrameDistance(config_.keyframe_spacing);

        double max_radius = config_.sc_config.max_radius;
        double spacing = config_.keyframe_spacing;

        SC_LOG("[ScanContext] Building keyframe database...");
        SC_LOG("  Keyframe spacing: " << spacing << "m");
        SC_LOG("  Descriptor radius: " << max_radius << "m");
        SC_LOG("  Map bounds: X[" << prebuilt_min_.x() << "," << prebuilt_max_.x()
                  << "] Y[" << prebuilt_min_.y() << "," << prebuilt_max_.y() << "]");

        double map_width = prebuilt_max_.x() - prebuilt_min_.x();
        double map_height = prebuilt_max_.y() - prebuilt_min_.y();
        SC_LOG("  Map size: " << map_width << "m x " << map_height << "m");

        auto t_start = std::chrono::high_resolution_clock::now();

        // Sample keyframe positions on a grid
        int keyframes_added = 0;
        int positions_tested = 0;

        // Use slightly smaller step than spacing to ensure good coverage
        double step = spacing * 0.9;

        // Use a smaller margin (10% of radius) to allow keyframes closer to edges
        // The descriptor will just have fewer points on the edge sides
        double margin = std::min(max_radius * 0.1, 5.0);  // Max 5m margin

        double x_start = prebuilt_min_.x() + margin;
        double x_end = prebuilt_max_.x() - margin;
        double y_start = prebuilt_min_.y() + margin;
        double y_end = prebuilt_max_.y() - margin;

        SC_LOG("  Keyframe grid: X[" << x_start << "," << x_end
                  << "] Y[" << y_start << "," << y_end << "] step=" << step << "m");

        // Check if map is too small
        if (x_end <= x_start || y_end <= y_start) {
            SC_LOG("  WARNING: Map too small for keyframes!");
            return;
        }

        for (double kf_x = x_start; kf_x <= x_end; kf_x += step) {
            for (double kf_y = y_start; kf_y <= y_end; kf_y += step) {
                positions_tested++;

                // Extract points within max_radius of this keyframe position
                std::vector<V3D> local_scan;
                local_scan.reserve(10000);

                double kf_z = 0.0;  // Sensor height (assume ground-level)
                int z_count = 0;

                for (const auto& pt : prebuilt_map_points_) {
                    double dx = pt.x() - kf_x;
                    double dy = pt.y() - kf_y;
                    double range_2d = std::sqrt(dx * dx + dy * dy);

                    if (range_2d <= max_radius) {
                        // Transform to sensor frame (relative to keyframe position)
                        local_scan.emplace_back(dx, dy, pt.z() - kf_z);

                        // Estimate average Z for this region
                        if (pt.z() < 0.5) {  // Ground points
                            kf_z += pt.z();
                            z_count++;
                        }
                    }
                }

                // Use average ground height as sensor Z (sensor is ~0.5m above ground)
                if (z_count > 0) {
                    double ground_z = kf_z / z_count;
                    kf_z = ground_z + 0.5;  // Sensor is 0.5m above ground
                }

                // Only add keyframe if we have enough points (lowered threshold for indoor)
                if (local_scan.size() < 100) continue;

                // Create pose for this keyframe (yaw=0 since we don't know orientation)
                M4D kf_pose = M4D::Identity();
                kf_pose(0, 3) = kf_x;
                kf_pose(1, 3) = kf_y;
                kf_pose(2, 3) = kf_z;

                if (sc_database_.addKeyFrame(local_scan, kf_pose)) {
                    keyframes_added++;
                }
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double build_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        SC_LOG("  Positions tested: " << positions_tested);
        SC_LOG("  Keyframes added: " << keyframes_added);
        SC_LOG("  Build time: " << std::fixed << std::setprecision(0)
                  << build_time_ms << "ms");
        SC_LOG("  SC database size: " << sc_database_.size());
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
     * @brief Check accumulation progress WITHOUT running ICP
     *
     * Call this periodically (e.g., after each SLAM cycle) with FAST-LIO's
     * current local map and trajectory. This only monitors coverage and
     * does NOT trigger global localization.
     *
     * @param local_map_points Points from FAST-LIO's ikd-tree
     * @param trajectory FAST-LIO's trajectory (all poses)
     * @return Localization result with coverage stats and readiness status
     */
    LocalizationResult checkAccumulation(
        const std::vector<WorldPoint>& local_map_points,
        const std::vector<M4D>& trajectory) {

        LocalizationResult result;

        if (status_ == LocalizationStatus::SUCCESS) {
            result.status = LocalizationStatus::SUCCESS;
            result.transform = best_transform_;
            result.confidence = best_confidence_;
            result.message = "Already localized";
            return result;
        }

        if (status_ == LocalizationStatus::ATTEMPTING) {
            result.status = LocalizationStatus::ATTEMPTING;
            result.message = "Localization in progress...";
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

        // Check if ready to attempt (but DON'T run ICP here)
        if (coverage_monitor_.isReadyForAttempt()) {
            status_ = LocalizationStatus::READY_FOR_LOCALIZATION;
            result.status = status_;
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "Ready to localize: %d voxels, %.0f deg - STOP robot and click Localize",
                     result.local_map_voxels, result.rotation_deg);
            result.message = std::string(buf);
            return result;
        }

        // Still accumulating
        status_ = LocalizationStatus::ACCUMULATING;
        result.status = status_;
        result.message = coverage_monitor_.getStatusMessage();
        return result;
    }

    /**
     * @brief Check if ready to run global localization
     */
    bool isReadyForLocalization() const {
        return status_ == LocalizationStatus::READY_FOR_LOCALIZATION ||
               status_ == LocalizationStatus::LOW_CONFIDENCE;
    }

    /**
     * @brief Run global localization (should be called with robot STOPPED)
     *
     * This runs the full ICP pipeline with progress reporting.
     * IMPORTANT: Robot should be stationary when calling this!
     *
     * @param local_map_points Points from FAST-LIO's ikd-tree
     * @param current_pose Current FAST-LIO pose
     * @param progress_callback Callback for progress updates
     * @return Localization result with transform if successful
     */
    LocalizationResult runGlobalLocalization(
        const std::vector<WorldPoint>& local_map_points,
        const M4D& current_pose,
        LocalizationProgressCallback progress_callback = nullptr) {

        LocalizationResult result;
        cancel_requested_.store(false);

        result.local_map_voxels = coverage_monitor_.getVoxelCount();
        result.rotation_deg = coverage_monitor_.getRotationDeg();
        result.distance_m = coverage_monitor_.getDistanceM();
        result.local_map_points = static_cast<int>(local_map_points.size());

        status_ = LocalizationStatus::ATTEMPTING;
        attempt_count_++;
        result.attempt_number = attempt_count_;

        std::cout << "\n[ProgressiveLocalizer] Running global localization..."
                  << std::endl;
        std::cout << "  Local map: " << local_map_points.size() << " points, "
                  << result.local_map_voxels << " voxels" << std::endl;
        std::cout << "  Coverage: " << result.rotation_deg << " deg rotation, "
                  << result.distance_m << " m traveled" << std::endl;

        // CRITICAL: Re-center local map around robot's CURRENT position
        // This allows the hint to represent "where is the robot NOW" (user expectation)
        // rather than "where did the robot START" (SLAM frame origin)
        V3D robot_position(current_pose(0, 3), current_pose(1, 3), current_pose(2, 3));
        std::cout << "  Robot current local pos: (" << robot_position.x() << ", "
                  << robot_position.y() << ", " << robot_position.z() << ")" << std::endl;

        // Convert local map to V3D, re-centered around robot's current position
        std::vector<V3D> local_points;
        local_points.reserve(local_map_points.size());
        for (const auto& pt : local_map_points) {
            // Shift points so robot's current position becomes origin
            local_points.emplace_back(
                pt.x - robot_position.x(),
                pt.y - robot_position.y(),
                pt.z - robot_position.z()
            );
        }

        // Log re-centering effect
        V3D new_min(1e9, 1e9, 1e9), new_max(-1e9, -1e9, -1e9);
        for (const auto& pt : local_points) {
            new_min = new_min.cwiseMin(pt);
            new_max = new_max.cwiseMax(pt);
        }
        std::cout << "  Re-centered local map: [" << new_min.x() << "," << new_max.x()
                  << "] x [" << new_min.y() << "," << new_max.y() << "]" << std::endl;

        // Run global localization with progress
        auto [transform_recentered, confidence] = attemptGlobalLocalizationWithProgress(
            local_points, progress_callback);

        // Check for cancellation
        if (cancel_requested_.load()) {
            status_ = LocalizationStatus::READY_FOR_LOCALIZATION;
            result.status = status_;
            result.message = "Localization cancelled";
            return result;
        }

        // CRITICAL: Adjust transform to account for re-centering
        // ICP found transform_recentered: maps recentered local points -> map frame
        // We need transform that maps odom points -> map frame
        // Chain: odom -> recentered -> map
        //   recentered_point = odom_point - robot_position
        //   map_point = transform_recentered * recentered_point
        // So: T_odom_to_map = transform_recentered * T_odom_to_recentered
        // where T_odom_to_recentered = [I | -robot_position]
        M4D T_odom_to_recentered = M4D::Identity();
        T_odom_to_recentered(0, 3) = -robot_position.x();
        T_odom_to_recentered(1, 3) = -robot_position.y();
        T_odom_to_recentered(2, 3) = -robot_position.z();

        M4D transform = transform_recentered * T_odom_to_recentered;
        std::cout << "  Transform adjusted for robot movement" << std::endl;

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
                 "Attempt %d: %.0f%% confidence - try again or move robot",
                 attempt_count_, confidence * 100);
        result.status = status_;
        result.message = std::string(buf);
        return result;
    }

    /**
     * @brief Legacy method - same as checkAccumulation
     * @deprecated Use checkAccumulation() instead
     */
    LocalizationResult checkAndLocalize(
        const std::vector<WorldPoint>& local_map_points,
        const std::vector<M4D>& trajectory,
        const M4D& current_pose) {
        // Just check accumulation - don't auto-run ICP
        auto result = checkAccumulation(local_map_points, trajectory);
        if (result.status == LocalizationStatus::SUCCESS) {
            result.pose = best_transform_ * current_pose;
        }
        return result;
    }

    LocalizationStatus getStatus() const { return status_; }
    const CoverageMonitor& getCoverageMonitor() const { return coverage_monitor_; }
    M4D getBestTransform() const { return best_transform_; }
    double getBestConfidence() const { return best_confidence_; }
    int getAttemptCount() const { return attempt_count_; }

private:
    /**
     * @brief Helper to report progress
     */
    void reportProgress(LocalizationProgressCallback callback,
                       LocalizationStage stage, float progress,
                       float stage_progress, const std::string& message,
                       int hypotheses = 0, int kept = 0,
                       double best_score = 0.0, double fitness = 0.0,
                       int iter = 0, int max_iter = 0) {
        if (!callback) return;

        LocalizationProgress p;
        p.stage = stage;
        p.progress = progress;
        p.stage_progress = stage_progress;
        p.message = message;
        p.hypotheses_count = hypotheses;
        p.hypotheses_kept = kept;
        p.best_score = best_score;
        p.current_fitness = fitness;
        p.current_iteration = iter;
        p.max_iterations = max_iter;
        callback(p);
    }

    /**
     * @brief Run global localization with progress reporting
     */
    std::pair<M4D, double> attemptGlobalLocalizationWithProgress(
        const std::vector<V3D>& local_map,
        LocalizationProgressCallback callback) {

        if (local_map.empty() || prebuilt_map_points_.empty()) {
            return {M4D::Identity(), 0.0};
        }

        // Progress breakdown:
        // Hypothesis generation: 0-20%
        // Coarse ICP: 20-45%
        // Medium ICP: 45-70%
        // Fine ICP: 70-95%
        // Evaluation: 95-100%

        // === Stage 1: Generate hypotheses (0-20%) ===
        reportProgress(callback, LocalizationStage::GENERATING_HYPOTHESES,
                      0.0f, 0.0f, "Generating hypotheses...");

        // Downsample for hypothesis generation
        auto local_coarse = voxelDownsample(local_map, config_.coarse_voxel);

        std::cout << "  Generating hypotheses..." << std::flush;

        // Generate hypotheses
        std::vector<std::pair<M4D, double>> hypotheses;
        int total_hypotheses = 0;  // For progress reporting

        // === Try Scan Context first (fast place recognition) ===
        bool sc_found_candidates = false;
        SC_LOG("  [ScanContext] use_scan_context=" << config_.use_scan_context
               << " database_size=" << sc_database_.size());
        if (config_.use_scan_context && !sc_database_.empty()) {
            SC_LOG("\n  [ScanContext] Querying database (" << sc_database_.size()
                      << " keyframes)...");

            auto sc_matches = sc_database_.query(local_map, config_.sc_top_candidates);

            SC_LOG("  [ScanContext] Found " << sc_matches.size() << " matches");

            // Log all match scores for debugging
            SC_LOG("  [ScanContext] Match scores (threshold=" << config_.sc_min_score << "):");
            for (size_t i = 0; i < sc_matches.size() && i < 10; i++) {
                SC_LOG("    Match[" << i << "]: score=" << std::fixed << std::setprecision(3)
                       << sc_matches[i].score << " kf=" << sc_matches[i].keyframe_id
                       << " pos=(" << sc_matches[i].position.x() << ","
                       << sc_matches[i].position.y() << ")");
            }

            int good_matches = 0;
            int rejected_bounds = 0;
            int rejected_hint = 0;
            int rejected_score = 0;

            for (const auto& match : sc_matches) {
                // Physical plausibility check 1: Score threshold
                if (match.score < config_.sc_min_score) {
                    rejected_score++;
                    continue;
                }

                // Physical plausibility check 2: Within map bounds (no margin - keyframes already have margin)
                if (match.position.x() < prebuilt_min_.x() ||
                    match.position.x() > prebuilt_max_.x() ||
                    match.position.y() < prebuilt_min_.y() ||
                    match.position.y() > prebuilt_max_.y()) {
                    rejected_bounds++;
                    continue;
                }

                // Physical plausibility check 3: If using hint, check distance
                if (config_.use_hint) {
                    double dx = match.position.x() - config_.hint_x;
                    double dy = match.position.y() - config_.hint_y;
                    double hint_dist = std::sqrt(dx * dx + dy * dy);

                    // Allow 2x hint radius for SC (it's approximate)
                    if (hint_dist > config_.hint_radius * 2.0) {
                        rejected_hint++;
                        continue;
                    }
                }

                good_matches++;

                // Convert SC match to hypothesis pose
                // SC provides: position and estimated yaw from descriptor matching
                M4D pose = M4D::Identity();
                pose(0, 0) = std::cos(match.yaw_estimate);
                pose(0, 1) = -std::sin(match.yaw_estimate);
                pose(1, 0) = std::sin(match.yaw_estimate);
                pose(1, 1) = std::cos(match.yaw_estimate);
                pose(0, 3) = match.position.x();
                pose(1, 3) = match.position.y();
                pose(2, 3) = match.position.z();

                // Score using voxel occupancy (same as grid search)
                double voxel_score = prebuilt_voxels_.scoreHypothesis(local_coarse, pose);

                // Boost score based on SC confidence
                double combined_score = voxel_score * (0.5 + 0.5 * match.score);

                hypotheses.emplace_back(pose, combined_score);

                SC_LOG("    SC[" << match.keyframe_id << "]: pos=("
                          << std::fixed << std::setprecision(1)
                          << match.position.x() << "," << match.position.y()
                          << ") yaw=" << (match.yaw_estimate * 180.0 / M_PI)
                          << "deg SC=" << std::setprecision(2) << (match.score * 100)
                          << "% vox=" << (voxel_score * 100) << "%");
            }

            // Log rejection statistics
            if (rejected_score + rejected_bounds + rejected_hint > 0) {
                SC_LOG("  [ScanContext] Rejected: " << rejected_score << " low-score, "
                          << rejected_bounds << " out-of-bounds, "
                          << rejected_hint << " far-from-hint");
            }

            if (good_matches >= 3) {
                sc_found_candidates = true;
                total_hypotheses = good_matches;  // For progress reporting
                SC_LOG("  [ScanContext] Using " << good_matches
                          << " SC candidates (skipping grid search)");
            } else {
                SC_LOG("  [ScanContext] Only " << good_matches
                          << " good matches");
                // TEMP: Don't fall back - force SC-only mode for testing
                if (good_matches > 0) {
                    sc_found_candidates = true;
                    total_hypotheses = good_matches;
                    SC_LOG("  [ScanContext] FORCING SC-only mode (grid search disabled for testing)");
                } else {
                    SC_LOG("  [ScanContext] ERROR: No SC matches and grid search disabled!");
                    std::cout << "  [ScanContext] Check: Is the SC database empty? Did map loading work?" << std::endl;
                    reportProgress(callback, LocalizationStage::COMPLETE,
                                  1.0f, 1.0f, "SC failed - no candidates!");
                    return {M4D::Identity(), 0.0};
                }
            }
        }

        // === Fall back to grid search if SC didn't find enough ===
        // Grid search: primary method for dry dock scenario (SC disabled)
        if (!sc_found_candidates) {
            double yaw_step = config_.yaw_step_deg * M_PI / 180.0;

        // Determine search bounds (hint-based or full map)
        double x_min, x_max, y_min, y_max;
        double yaw_min, yaw_max;

        if (config_.use_hint) {
            // Bounded search around hint position
            x_min = config_.hint_x - config_.hint_radius;
            x_max = config_.hint_x + config_.hint_radius;
            y_min = config_.hint_y - config_.hint_radius;
            y_max = config_.hint_y + config_.hint_radius;

            // Clamp to map bounds
            x_min = std::max(x_min, prebuilt_min_.x());
            x_max = std::min(x_max, prebuilt_max_.x());
            y_min = std::max(y_min, prebuilt_min_.y());
            y_max = std::min(y_max, prebuilt_max_.y());

            if (config_.hint_heading_known) {
                // Search ±heading_range around hint heading
                yaw_min = config_.hint_heading - config_.hint_heading_range;
                yaw_max = config_.hint_heading + config_.hint_heading_range;
            } else {
                // Search all headings
                yaw_min = 0.0;
                yaw_max = 2.0 * M_PI;
            }

            std::cout << "  Hint-bounded search: X[" << x_min << "," << x_max
                      << "] Y[" << y_min << "," << y_max << "]"
                      << " Yaw[" << (yaw_min * 180.0 / M_PI) << "," << (yaw_max * 180.0 / M_PI) << "]deg"
                      << std::endl;
        } else {
            // Full map search
            x_min = prebuilt_min_.x();
            x_max = prebuilt_max_.x();
            y_min = prebuilt_min_.y();
            y_max = prebuilt_max_.y();
            yaw_min = 0.0;
            yaw_max = 2.0 * M_PI;
            std::cout << "  Full map search (no hint)" << std::endl;
        }

        int num_yaws = static_cast<int>(std::ceil((yaw_max - yaw_min) / yaw_step));
        if (num_yaws < 1) num_yaws = 1;

        // Determine Z value
        double z_val = config_.search_z ?
            (prebuilt_min_.z() + prebuilt_max_.z()) * 0.5 :
            config_.z_value;

        // Count total hypotheses for progress
        double x_range = x_max - x_min;
        double y_range = y_max - y_min;
        int x_steps = static_cast<int>(std::ceil(x_range / config_.grid_step)) + 1;
        int y_steps = static_cast<int>(std::ceil(y_range / config_.grid_step)) + 1;
        total_hypotheses = x_steps * y_steps * num_yaws;  // Update outer variable
        int processed_hypotheses = 0;

        std::cout << "  Grid: " << x_steps << "x" << y_steps << "x" << num_yaws
                  << " = " << total_hypotheses << " hypotheses" << std::endl;

        // Grid search over bounds
        for (double x = x_min; x <= x_max; x += config_.grid_step) {
            // Check for cancellation
            if (cancel_requested_.load()) {
                return {M4D::Identity(), 0.0};
            }

            for (double y = y_min; y <= y_max; y += config_.grid_step) {
                // Skip candidates outside circular hint region (if using hint)
                if (config_.use_hint) {
                    double dx = x - config_.hint_x;
                    double dy = y - config_.hint_y;
                    if (dx * dx + dy * dy > config_.hint_radius * config_.hint_radius) {
                        // Outside circle, but still count for progress
                        processed_hypotheses += num_yaws;
                        continue;
                    }
                }

                for (int yi = 0; yi < num_yaws; yi++) {
                    double yaw = yaw_min + yi * yaw_step;

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

                    processed_hypotheses++;
                }
            }

            // Update progress
            float stage_prog = static_cast<float>(processed_hypotheses) / total_hypotheses;
            float overall_prog = stage_prog * 0.20f;
            char buf[128];
            snprintf(buf, sizeof(buf), "Generating hypotheses: %d/%d tested, %zu kept",
                    processed_hypotheses, total_hypotheses, hypotheses.size());
            reportProgress(callback, LocalizationStage::GENERATING_HYPOTHESES,
                          overall_prog, stage_prog, buf,
                          total_hypotheses, static_cast<int>(hypotheses.size()));
        }
        } // end if (!sc_found_candidates)

        std::cout << " " << hypotheses.size() << " candidates" << std::endl;

        if (hypotheses.empty()) {
            std::cout << "  No plausible hypotheses found!" << std::endl;
            reportProgress(callback, LocalizationStage::COMPLETE,
                          1.0f, 1.0f, "No plausible hypotheses found!");
            return {M4D::Identity(), 0.0};
        }

        // Sort by voxel score descending
        std::sort(hypotheses.begin(), hypotheses.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        // Log top hypotheses for debugging
        std::cout << "\n=== LOCALIZATION LOG ===" << std::endl;
        std::cout << "  Total hypotheses generated: " << hypotheses.size() << std::endl;
        std::cout << "  Top 10 by voxel score:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), hypotheses.size()); i++) {
            const auto& hyp = hypotheses[i];
            double x = hyp.first(0, 3);
            double y = hyp.first(1, 3);
            double yaw = std::atan2(hyp.first(1, 0), hyp.first(0, 0)) * 180.0 / M_PI;
            std::cout << "    [" << i << "] pos=(" << std::fixed << std::setprecision(2)
                      << x << "," << y << ") yaw=" << std::setprecision(1) << yaw
                      << "deg score=" << std::setprecision(1) << (hyp.second * 100) << "%" << std::endl;
        }

        // === Stage 2: Multi-hypothesis coarse ICP (20-50%) ===
        // Refine top N hypotheses (not just top 1) for reliability
        const int NUM_TO_REFINE = std::min(20, static_cast<int>(hypotheses.size()));
        std::cout << "\n  Refining top " << NUM_TO_REFINE << " hypotheses with coarse ICP..." << std::endl;

        // Struct to hold refined candidates
        struct RefinedCandidate {
            M4D pose;
            double voxel_score;
            double coarse_fitness;
            double final_fitness;
            int original_rank;
        };
        std::vector<RefinedCandidate> refined_candidates;
        refined_candidates.reserve(NUM_TO_REFINE);

        // Downsample once for coarse ICP
        auto scan_coarse = voxelDownsample(local_map, config_.coarse_voxel);
        auto map_coarse = voxelDownsample(prebuilt_map_points_, config_.coarse_voxel);

        for (int h = 0; h < NUM_TO_REFINE; h++) {
            if (cancel_requested_.load()) return {M4D::Identity(), 0.0};

            M4D pose = hypotheses[h].first;
            double voxel_score = hypotheses[h].second;

            // Quick coarse ICP (fewer iterations since we're testing many)
            ICPConfig cfg;
            cfg.max_iterations = 15;  // Reduced from 30 - just enough to converge
            cfg.max_correspondence_dist = 3.0;
            cfg.convergence_threshold = 1e-4;

            ICP icp(cfg);
            double coarse_fitness = 0.0;
            for (int i = 0; i < cfg.max_iterations; i++) {
                auto result = icp.alignOneIteration(scan_coarse, map_coarse, pose);
                pose = result.transformation;
                coarse_fitness = result.fitness_score;
                if (result.converged) break;
            }

            // Log this candidate
            double x = pose(0, 3);
            double y = pose(1, 3);
            double yaw = std::atan2(pose(1, 0), pose(0, 0)) * 180.0 / M_PI;
            std::cout << "    [" << h << "] after coarse ICP: pos=(" << std::fixed << std::setprecision(2)
                      << x << "," << y << ") yaw=" << std::setprecision(1) << yaw
                      << "deg voxel=" << (voxel_score * 100) << "% fit=" << (coarse_fitness * 100) << "%" << std::endl;

            refined_candidates.push_back({pose, voxel_score, coarse_fitness, 0.0, h});

            // Update progress
            float stage_prog = static_cast<float>(h + 1) / NUM_TO_REFINE;
            float overall_prog = 0.20f + stage_prog * 0.30f;
            char buf[128];
            snprintf(buf, sizeof(buf), "Coarse ICP: %d/%d hypotheses (best: %.1f%%)",
                    h + 1, NUM_TO_REFINE, coarse_fitness * 100);
            reportProgress(callback, LocalizationStage::COARSE_ICP,
                          overall_prog, stage_prog, buf,
                          total_hypotheses, NUM_TO_REFINE);
        }

        // Sort by coarse fitness
        std::sort(refined_candidates.begin(), refined_candidates.end(),
                  [](const auto& a, const auto& b) { return a.coarse_fitness > b.coarse_fitness; });

        std::cout << "\n  Top 5 after coarse ICP:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), refined_candidates.size()); i++) {
            const auto& c = refined_candidates[i];
            double x = c.pose(0, 3);
            double y = c.pose(1, 3);
            double yaw = std::atan2(c.pose(1, 0), c.pose(0, 0)) * 180.0 / M_PI;
            std::cout << "    [" << i << "] pos=(" << std::fixed << std::setprecision(2)
                      << x << "," << y << ") yaw=" << std::setprecision(1) << yaw
                      << "deg coarse_fit=" << (c.coarse_fitness * 100) << "% (orig rank " << c.original_rank << ")" << std::endl;
        }

        // === Stage 3: Medium ICP on top 5 (50-75%) ===
        // Use MEDIUM voxels (0.2m) for quick comparison - much faster than fine!
        const int NUM_MEDIUM = std::min(5, static_cast<int>(refined_candidates.size()));
        std::cout << "\n  Running medium ICP on top " << NUM_MEDIUM << " candidates..." << std::endl;

        auto t_medium_start = std::chrono::high_resolution_clock::now();
        auto scan_medium = voxelDownsample(local_map, config_.medium_voxel);  // 0.2m voxels

        // CRITICAL OPTIMIZATION: Crop prebuilt map based on COARSE ICP results
        // The coarse stage has already narrowed down positions - use that info!
        std::vector<V3D> map_cropped_medium;
        {
            // Compute bounding box of top candidates from coarse ICP
            double crop_x_min = 1e9, crop_x_max = -1e9;
            double crop_y_min = 1e9, crop_y_max = -1e9;
            for (const auto& c : refined_candidates) {
                crop_x_min = std::min(crop_x_min, c.pose(0, 3));
                crop_x_max = std::max(crop_x_max, c.pose(0, 3));
                crop_y_min = std::min(crop_y_min, c.pose(1, 3));
                crop_y_max = std::max(crop_y_max, c.pose(1, 3));
            }
            // Add margin for local map extent and ICP convergence
            double margin = 20.0;  // Covers local map size + ICP drift
            crop_x_min -= margin; crop_x_max += margin;
            crop_y_min -= margin; crop_y_max += margin;

            std::cout << "    Medium ICP crop region: [" << crop_x_min << "," << crop_x_max
                      << "] x [" << crop_y_min << "," << crop_y_max << "]" << std::endl;

            map_cropped_medium.reserve(prebuilt_map_points_.size() / 4);
            for (const auto& pt : prebuilt_map_points_) {
                if (pt.x() >= crop_x_min && pt.x() <= crop_x_max &&
                    pt.y() >= crop_y_min && pt.y() <= crop_y_max) {
                    map_cropped_medium.push_back(pt);
                }
            }
            std::cout << "    Cropped prebuilt map from " << prebuilt_map_points_.size()
                      << " to " << map_cropped_medium.size() << " points" << std::endl;
        }
        auto map_medium = voxelDownsample(map_cropped_medium, config_.medium_voxel);
        std::cout << "    Downsampled: scan=" << scan_medium.size() << " map=" << map_medium.size() << " points" << std::endl;

        for (int h = 0; h < NUM_MEDIUM; h++) {
            if (cancel_requested_.load()) return {M4D::Identity(), 0.0};

            auto& candidate = refined_candidates[h];

            ICPConfig cfg;
            cfg.max_iterations = 20;  // Reduced from 40
            cfg.max_correspondence_dist = 0.6;
            cfg.convergence_threshold = 1e-5;

            ICP icp(cfg);
            auto result = icp.align(scan_medium, map_medium, candidate.pose);
            candidate.pose = result.transformation;
            candidate.final_fitness = result.fitness_score;

            // Log result
            double x = candidate.pose(0, 3);
            double y = candidate.pose(1, 3);
            double yaw = std::atan2(candidate.pose(1, 0), candidate.pose(0, 0)) * 180.0 / M_PI;
            std::cout << "    [" << h << "] medium: pos=(" << std::fixed << std::setprecision(2)
                      << x << "," << y << ") yaw=" << std::setprecision(1) << yaw
                      << "deg fit=" << (candidate.final_fitness * 100) << "%" << std::endl;

            // Update progress
            float stage_prog = static_cast<float>(h + 1) / NUM_MEDIUM;
            float overall_prog = 0.50f + stage_prog * 0.25f;
            char buf[128];
            snprintf(buf, sizeof(buf), "Medium ICP: %d/%d (best: %.1f%%)",
                    h + 1, NUM_MEDIUM, candidate.final_fitness * 100);
            reportProgress(callback, LocalizationStage::MEDIUM_ICP,
                          overall_prog, stage_prog, buf,
                          total_hypotheses, NUM_MEDIUM);
        }

        auto t_medium_end = std::chrono::high_resolution_clock::now();
        double medium_time_ms = std::chrono::duration<double, std::milli>(t_medium_end - t_medium_start).count();
        std::cout << "    Medium ICP took " << std::fixed << std::setprecision(0) << medium_time_ms << "ms" << std::endl;

        // Sort by medium fitness
        std::sort(refined_candidates.begin(), refined_candidates.begin() + NUM_MEDIUM,
                  [](const auto& a, const auto& b) { return a.final_fitness > b.final_fitness; });

        // === Stage 4: Fine ICP on ONLY the best candidate (75-90%) ===
        std::cout << "\n  Running fine ICP on best candidate only..." << std::endl;
        reportProgress(callback, LocalizationStage::FINE_ICP,
                      0.75f, 0.0f, "Fine alignment on best candidate...",
                      total_hypotheses, 1);

        auto t_fine_start = std::chrono::high_resolution_clock::now();
        auto scan_fine = voxelDownsample(local_map, config_.fine_voxel);  // 0.1m voxels

        // CRITICAL OPTIMIZATION: Crop tightly around best candidate for fine ICP
        // At this point we know approximate position, so we can use a tight crop
        std::vector<V3D> map_cropped_fine;
        {
            double best_x = refined_candidates[0].pose(0, 3);
            double best_y = refined_candidates[0].pose(1, 3);
            double fine_margin = 20.0;  // Tight crop around best candidate

            map_cropped_fine.reserve(map_cropped_medium.size());  // Usually similar size
            for (const auto& pt : prebuilt_map_points_) {
                if (pt.x() >= best_x - fine_margin && pt.x() <= best_x + fine_margin &&
                    pt.y() >= best_y - fine_margin && pt.y() <= best_y + fine_margin) {
                    map_cropped_fine.push_back(pt);
                }
            }
            std::cout << "    Cropped for fine ICP: " << map_cropped_fine.size()
                      << " points (around " << best_x << ", " << best_y << ")" << std::endl;
        }
        auto map_fine = voxelDownsample(map_cropped_fine, config_.fine_voxel);
        std::cout << "    Downsampled: scan=" << scan_fine.size() << " map=" << map_fine.size() << " points" << std::endl;

        {
            auto& best = refined_candidates[0];

            ICPConfig cfg;
            cfg.max_iterations = 30;
            cfg.max_correspondence_dist = 0.4;
            cfg.convergence_threshold = 1e-6;

            ICP icp(cfg);
            auto result = icp.align(scan_fine, map_fine, best.pose);
            best.pose = result.transformation;
            best.final_fitness = result.fitness_score;

            double x = best.pose(0, 3);
            double y = best.pose(1, 3);
            double yaw = std::atan2(best.pose(1, 0), best.pose(0, 0)) * 180.0 / M_PI;
            std::cout << "    Best final: pos=(" << std::fixed << std::setprecision(2)
                      << x << "," << y << ") yaw=" << std::setprecision(1) << yaw
                      << "deg fit=" << (best.final_fitness * 100) << "%" << std::endl;
        }

        auto t_fine_end = std::chrono::high_resolution_clock::now();
        double fine_time_ms = std::chrono::duration<double, std::milli>(t_fine_end - t_fine_start).count();
        std::cout << "    Fine ICP took " << std::fixed << std::setprecision(0) << fine_time_ms << "ms" << std::endl;

        reportProgress(callback, LocalizationStage::FINE_ICP,
                      0.90f, 1.0f, "Fine alignment complete",
                      total_hypotheses, 1);

        // === Stage 5: Distinctiveness check (90-95%) ===
        // Use medium fitness for distinctiveness (we only ran fine on the best)
        reportProgress(callback, LocalizationStage::EVALUATING,
                      0.90f, 0.0f, "Evaluating distinctiveness...",
                      total_hypotheses, NUM_MEDIUM);

        double best_fitness = refined_candidates[0].final_fitness;
        double second_fitness = NUM_MEDIUM > 1 ? refined_candidates[1].final_fitness : 0.0;
        double distinctiveness = (best_fitness - second_fitness) / std::max(0.01, best_fitness);

        std::cout << "\n=== DISTINCTIVENESS CHECK ===" << std::endl;
        std::cout << "  Best fitness (fine):   " << std::fixed << std::setprecision(1) << (best_fitness * 100) << "%" << std::endl;
        std::cout << "  Second fitness (med):  " << (second_fitness * 100) << "%" << std::endl;
        std::cout << "  Distinctiveness: " << (distinctiveness * 100) << "%" << std::endl;

        // Position comparison between top 2
        if (NUM_MEDIUM > 1) {
            double dx = refined_candidates[0].pose(0, 3) - refined_candidates[1].pose(0, 3);
            double dy = refined_candidates[0].pose(1, 3) - refined_candidates[1].pose(1, 3);
            double dist = std::sqrt(dx * dx + dy * dy);
            double yaw1 = std::atan2(refined_candidates[0].pose(1, 0), refined_candidates[0].pose(0, 0)) * 180.0 / M_PI;
            double yaw2 = std::atan2(refined_candidates[1].pose(1, 0), refined_candidates[1].pose(0, 0)) * 180.0 / M_PI;
            std::cout << "  Distance between top 2: " << std::setprecision(2) << dist << "m" << std::endl;
            std::cout << "  Yaw difference: " << std::setprecision(1) << std::abs(yaw1 - yaw2) << "deg" << std::endl;
        }

        // Determine result quality
        std::string quality;
        if (best_fitness >= 0.60 && distinctiveness >= 0.15) {
            quality = "HIGH CONFIDENCE";
        } else if (best_fitness >= 0.50 && distinctiveness >= 0.10) {
            quality = "MEDIUM CONFIDENCE";
        } else if (distinctiveness < 0.10) {
            quality = "AMBIGUOUS (low distinctiveness)";
        } else {
            quality = "LOW CONFIDENCE";
        }
        std::cout << "  Result: " << quality << std::endl;
        std::cout << "========================\n" << std::endl;

        M4D best_pose = refined_candidates[0].pose;
        double final_confidence = best_fitness;

        // === Stage 5: Complete ===
        char buf[128];
        snprintf(buf, sizeof(buf), "Complete: %.1f%% (distinct: %.0f%%)",
                 final_confidence * 100, distinctiveness * 100);
        reportProgress(callback, LocalizationStage::COMPLETE,
                      1.0f, 1.0f, buf,
                      total_hypotheses, static_cast<int>(hypotheses.size()));

        return {best_pose, final_confidence};
    }

    /**
     * @brief Run global localization (legacy, no progress)
     */
    std::pair<M4D, double> attemptGlobalLocalization(const std::vector<V3D>& local_map) {
        return attemptGlobalLocalizationWithProgress(local_map, nullptr);
    }

    ProgressiveLocalizerConfig config_;
    CoverageMonitor coverage_monitor_;
    VoxelOccupancyMap prebuilt_voxels_;
    ScanContextDatabase sc_database_;      // Keyframe database for place recognition
    std::vector<V3D> prebuilt_map_points_;
    V3D prebuilt_min_, prebuilt_max_;

    LocalizationStatus status_ = LocalizationStatus::NOT_STARTED;
    int attempt_count_ = 0;
    M4D best_transform_ = M4D::Identity();
    double best_confidence_ = 0.0;
    std::atomic<bool> cancel_requested_{false};
};

} // namespace slam

#endif // PROGRESSIVE_LOCALIZER_HPP
