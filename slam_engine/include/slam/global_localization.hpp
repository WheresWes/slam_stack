/**
 * @file global_localization.hpp
 * @brief Global localization from unknown initial position
 *
 * Designed for challenging outdoor environments like ship hulls in dry dock.
 * Uses multi-descriptor place recognition with hierarchical verification.
 *
 * Key features:
 * - Scan Context descriptor for place recognition
 * - Height-band descriptors for Mid-360's vertical coverage
 * - Curvature estimation for curved surfaces (ship hull)
 * - Voxel occupancy map for fast pose scoring
 * - Multi-level verification pyramid
 * - Robust disambiguation for symmetric environments
 */

#ifndef SLAM_GLOBAL_LOCALIZATION_HPP
#define SLAM_GLOBAL_LOCALIZATION_HPP

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <random>

#include "slam/types.hpp"
#include "slam/icp.hpp"

namespace slam {

//=============================================================================
// Forward Declarations
//=============================================================================

class ScanContext;
class ScanContextDatabase;
class VoxelOccupancyMap;
class GlobalLocalizer;

//=============================================================================
// Configuration Structures
//=============================================================================

/**
 * @brief Configuration for Scan Context descriptor
 */
struct ScanContextConfig {
    int num_rings = 20;              // Radial divisions
    int num_sectors = 60;            // Angular divisions (6 degree resolution)
    double max_radius = 40.0;        // Maximum range (Mid-360: 40m)
    double min_radius = 1.0;         // Minimum range (ignore close points)
    int num_height_bands = 5;        // Vertical divisions for 3D descriptor
    double min_height = -2.0;        // Minimum height (below sensor)
    double max_height = 15.0;        // Maximum height (ship hull/deck)

    // For ship hull environment
    bool use_curvature = true;       // Include curvature in descriptor
    bool use_intensity = false;      // Include intensity (often unreliable outdoors)
};

/**
 * @brief Configuration for global localizer
 */
struct GlobalLocalizerConfig {
    // Timing
    double scan_accumulation_time = 1.5;    // seconds
    double max_total_time = 15.0;           // seconds

    // Reliability thresholds
    double min_fitness_for_acceptance = 0.45;    // 45% inliers (lower for outdoor)
    double high_confidence_fitness = 0.65;       // Can skip further checks
    double distinctiveness_threshold = 0.12;     // 12% better than runner-up

    // Candidate counts at each level
    int scan_context_candidates = 50;
    int grid_search_candidates = 500;
    int coarse_verification_candidates = 100;
    int medium_verification_candidates = 20;
    int fine_verification_candidates = 5;

    // Grid search parameters (fallback)
    double grid_resolution_xy = 3.0;        // meters (larger for ship scale)
    double grid_resolution_yaw = 10.0;      // degrees

    // Physical constraints
    double max_floor_deviation = 1.0;       // meters (dock floor variation)
    double min_obstacle_clearance = 0.3;    // meters

    // ICP parameters
    double coarse_voxel_size = 1.0;         // Large for speed
    double medium_voxel_size = 0.3;         // Medium detail
    double fine_voxel_size = 0.1;           // Full detail

    // Scan Context
    ScanContextConfig sc_config;
};

//=============================================================================
// Scan Context Descriptor
//=============================================================================

/**
 * @brief Scan Context descriptor for place recognition
 *
 * Encodes a point cloud as a 2D polar representation with optional
 * height bands for 3D structure. Optimized for LiDAR-based localization.
 *
 * For ship hull environment:
 * - Uses multiple height bands to capture hull curvature
 * - Larger radial bins for large-scale outdoor environment
 * - Optional curvature encoding for curved surfaces
 */
class ScanContext {
public:
    using Descriptor = Eigen::MatrixXd;  // (num_rings × num_sectors) or (num_rings × num_sectors × num_bands)
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
        if (config_.num_height_bands > 1) {
            // For 3D descriptor, reshape and average
            int total_cols = config_.num_sectors * config_.num_height_bands;
            Eigen::MatrixXd reshaped = desc;
            return reshaped.rowwise().mean();
        } else {
            return desc.rowwise().mean();
        }
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

        int num_cols = query.cols();

        for (int shift = 0; shift < config_.num_sectors; shift++) {
            // Circular shift the candidate
            Descriptor shifted(candidate.rows(), candidate.cols());
            for (int c = 0; c < num_cols; c++) {
                int src_col = (c + shift) % num_cols;
                // Handle 3D descriptor with multiple bands
                if (config_.num_height_bands > 1) {
                    int sector = c % config_.num_sectors;
                    int band = c / config_.num_sectors;
                    int src_sector = (sector + shift) % config_.num_sectors;
                    shifted.col(c) = candidate.col(band * config_.num_sectors + src_sector);
                } else {
                    shifted.col(c) = candidate.col(src_col);
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

    /**
     * @brief Get configuration
     */
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

            // Skip points outside range
            if (range < config_.min_radius || range > config_.max_radius) continue;

            // Find ring (binary search on log-spaced boundaries)
            int ring = findRing(range);
            if (ring < 0 || ring >= config_.num_rings) continue;

            // Find sector
            double angle = std::atan2(pt.y(), pt.x());  // -π to π
            int sector = static_cast<int>((angle + M_PI) / (2.0 * M_PI) * config_.num_sectors);
            sector = std::clamp(sector, 0, config_.num_sectors - 1);

            // Store max height
            desc(ring, sector) = std::max(desc(ring, sector), pt.z());
        }

        return desc;
    }

    /**
     * @brief 3D Scan Context with height bands
     *
     * For ship hull: captures structure at different heights
     * - Ground level (dock floor)
     * - Lower hull
     * - Upper hull
     * - Deck/superstructure
     * - Sky (should be mostly empty)
     */
    Descriptor compute3D(const std::vector<V3D>& cloud) const {
        int total_cols = config_.num_sectors * config_.num_height_bands;
        Descriptor desc = Descriptor::Zero(config_.num_rings, total_cols);

        // Count points per cell (for density encoding)
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

            // Use point density as feature (more robust than max height for outdoor)
            counts(ring, col)++;
        }

        // Normalize counts to 0-1 range
        double max_count = counts.maxCoeff();
        if (max_count > 0) {
            desc = counts.cast<double>() / max_count;
        }

        return desc;
    }

    /**
     * @brief Find ring index using log-spaced boundaries
     */
    int findRing(double range) const {
        // Binary search
        auto it = std::lower_bound(ring_boundaries_.begin(), ring_boundaries_.end(), range);
        int idx = static_cast<int>(it - ring_boundaries_.begin()) - 1;
        return std::clamp(idx, 0, config_.num_rings - 1);
    }

    /**
     * @brief Compute cosine similarity between two matrices
     */
    double cosineSimilarity(const Descriptor& a, const Descriptor& b) const {
        // Flatten and compute cosine similarity
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
 * Stores keyframe descriptors with spatial indexing for fast retrieval.
 */
class ScanContextDatabase {
public:
    struct KeyFrame {
        int id;
        V3D position;
        M3D orientation;
        double yaw;  // For quick access
        ScanContext::Descriptor descriptor;
        ScanContext::RingKey ring_key;
        double timestamp;
    };

    struct Match {
        int keyframe_id;
        V3D position;
        double yaw_estimate;
        double score;
    };

    explicit ScanContextDatabase(const ScanContextConfig& config = ScanContextConfig())
        : sc_(config), config_(config) {}

    /**
     * @brief Add a keyframe to the database
     * @param scan Point cloud in world frame
     * @param pose Pose of this keyframe
     * @param timestamp Time of capture
     * @return true if added (false if too close to existing keyframe)
     */
    bool addKeyFrame(const std::vector<V3D>& scan,
                      const M4D& pose,
                      double timestamp = 0.0) {
        V3D position = pose.block<3,1>(0,3);

        // Check if too close to existing keyframe
        for (const auto& kf : keyframes_) {
            if ((position - kf.position).norm() < min_keyframe_distance_) {
                return false;
            }
        }

        // Transform scan to local frame (sensor-centric)
        M4D pose_inv = pose.inverse();
        std::vector<V3D> local_scan;
        local_scan.reserve(scan.size());
        M3D R_inv = pose_inv.block<3,3>(0,0);
        V3D t_inv = pose_inv.block<3,1>(0,3);
        for (const auto& pt : scan) {
            local_scan.push_back(R_inv * pt + t_inv);
        }

        // Compute descriptor
        KeyFrame kf;
        kf.id = static_cast<int>(keyframes_.size());
        kf.position = position;
        kf.orientation = pose.block<3,3>(0,0);
        kf.yaw = std::atan2(pose(1,0), pose(0,0));
        kf.descriptor = sc_.compute(local_scan);
        kf.ring_key = sc_.computeRingKey(kf.descriptor);
        kf.timestamp = timestamp;

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
        if (keyframes_.empty()) {
            return {};
        }

        // Compute query descriptor
        auto query_desc = sc_.compute(scan);
        auto query_ring_key = sc_.computeRingKey(query_desc);

        // Pre-filter with ring key distance (fast)
        std::vector<std::pair<int, double>> ring_distances;
        ring_distances.reserve(keyframes_.size());

        for (const auto& kf : keyframes_) {
            double dist = (query_ring_key - kf.ring_key).norm();
            ring_distances.push_back({kf.id, dist});
        }

        // Sort by ring distance
        std::sort(ring_distances.begin(), ring_distances.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });

        // Full matching on top candidates (3x top_k for filtering)
        int num_to_match = std::min(static_cast<int>(ring_distances.size()), top_k * 3);
        std::vector<Match> matches;
        matches.reserve(num_to_match);

        for (int i = 0; i < num_to_match; i++) {
            const auto& kf = keyframes_[ring_distances[i].first];

            auto [score, shift] = sc_.match(query_desc, kf.descriptor);

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

    /**
     * @brief Get number of keyframes
     */
    size_t size() const { return keyframes_.size(); }

    /**
     * @brief Check if database is empty
     */
    bool empty() const { return keyframes_.empty(); }

    /**
     * @brief Clear database
     */
    void clear() { keyframes_.clear(); }

    /**
     * @brief Get keyframe by ID
     */
    const KeyFrame& getKeyFrame(int id) const { return keyframes_[id]; }

    /**
     * @brief Set minimum distance between keyframes
     */
    void setMinKeyFrameDistance(double dist) { min_keyframe_distance_ = dist; }

    /**
     * @brief Get bounding box of all keyframes
     */
    std::pair<V3D, V3D> getBoundingBox() const {
        if (keyframes_.empty()) {
            return {V3D::Zero(), V3D::Zero()};
        }

        V3D min_pt = keyframes_[0].position;
        V3D max_pt = keyframes_[0].position;

        for (const auto& kf : keyframes_) {
            min_pt = min_pt.cwiseMin(kf.position);
            max_pt = max_pt.cwiseMax(kf.position);
        }

        // Add margin
        V3D margin(5.0, 5.0, 2.0);
        return {min_pt - margin, max_pt + margin};
    }

private:
    ScanContext sc_;
    ScanContextConfig config_;
    std::vector<KeyFrame> keyframes_;
    double min_keyframe_distance_ = 5.0;  // meters (larger for ship scale)
};

//=============================================================================
// Voxel Occupancy Map
//=============================================================================

/**
 * @brief Hash function for 3D voxel keys
 */
struct VoxelKeyHash {
    std::size_t operator()(const Eigen::Vector3i& v) const {
        // Use a simple hash combining
        std::size_t h = 0;
        h ^= std::hash<int>()(v.x()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(v.y()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(v.z()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

/**
 * @brief Fast voxel-based occupancy map for pose scoring
 *
 * Uses hash-based storage for O(1) occupancy lookup.
 * Supports multi-resolution for coarse-to-fine verification.
 */
class VoxelOccupancyMap {
public:
    explicit VoxelOccupancyMap(double voxel_size = 0.5)
        : voxel_size_(voxel_size), inv_voxel_size_(1.0 / voxel_size) {}

    /**
     * @brief Build map from point cloud
     */
    void build(const std::vector<V3D>& points) {
        occupied_voxels_.clear();
        occupied_voxels_.reserve(points.size() / 10);  // Rough estimate

        for (const auto& pt : points) {
            Eigen::Vector3i key = toVoxelKey(pt);
            occupied_voxels_.insert(key);
        }

        // Compute bounding box
        if (!points.empty()) {
            min_bound_ = points[0];
            max_bound_ = points[0];
            for (const auto& pt : points) {
                min_bound_ = min_bound_.cwiseMin(pt);
                max_bound_ = max_bound_.cwiseMax(pt);
            }
        }
    }

    /**
     * @brief Add points to existing map
     */
    void addPoints(const std::vector<V3D>& points) {
        for (const auto& pt : points) {
            Eigen::Vector3i key = toVoxelKey(pt);
            occupied_voxels_.insert(key);

            min_bound_ = min_bound_.cwiseMin(pt);
            max_bound_ = max_bound_.cwiseMax(pt);
        }
    }

    /**
     * @brief Check if a point falls in an occupied voxel
     */
    bool isOccupied(const V3D& pt) const {
        return occupied_voxels_.count(toVoxelKey(pt)) > 0;
    }

    /**
     * @brief Score a transformed scan against the map
     * @param scan Scan points in sensor frame
     * @param pose Pose to evaluate
     * @param num_samples Number of points to sample (0 = all)
     * @return Occupancy score (0-1, fraction of points in occupied voxels)
     */
    double scorePose(const std::vector<V3D>& scan,
                      const M4D& pose,
                      int num_samples = 200) const {
        if (scan.empty() || occupied_voxels_.empty()) return 0.0;

        M3D R = pose.block<3,3>(0,0);
        V3D t = pose.block<3,1>(0,3);

        int hits = 0;
        int step = 1;
        int count = static_cast<int>(scan.size());

        if (num_samples > 0 && num_samples < count) {
            step = count / num_samples;
            count = num_samples;
        }

        for (int i = 0; i < count; i++) {
            const V3D& pt = scan[i * step];
            V3D world_pt = R * pt + t;
            if (isOccupied(world_pt)) {
                hits++;
            }
        }

        return static_cast<double>(hits) / count;
    }

    /**
     * @brief Estimate floor Z coordinate at a given XY position
     */
    double estimateFloorZ(double x, double y) const {
        // Find lowest occupied voxel in this column
        double min_z = max_bound_.z();

        Eigen::Vector3i base_key = toVoxelKey(V3D(x, y, 0));

        for (int dz = static_cast<int>(min_bound_.z() / voxel_size_);
             dz <= static_cast<int>(max_bound_.z() / voxel_size_); dz++) {
            Eigen::Vector3i key(base_key.x(), base_key.y(), dz);
            if (occupied_voxels_.count(key) > 0) {
                double z = dz * voxel_size_;
                if (z < min_z) min_z = z;
                break;  // Found lowest
            }
        }

        return min_z;
    }

    /**
     * @brief Get feature density at a location (for search prioritization)
     */
    double getFeatureDensity(double x, double y, double radius = 5.0) const {
        int count = 0;
        int cells = static_cast<int>(radius / voxel_size_);
        int total_cells = 0;

        Eigen::Vector3i center = toVoxelKey(V3D(x, y, 0));

        for (int dx = -cells; dx <= cells; dx++) {
            for (int dy = -cells; dy <= cells; dy++) {
                for (int dz = static_cast<int>(min_bound_.z() / voxel_size_);
                     dz <= static_cast<int>(max_bound_.z() / voxel_size_); dz++) {
                    Eigen::Vector3i key(center.x() + dx, center.y() + dy, dz);
                    if (occupied_voxels_.count(key) > 0) {
                        count++;
                    }
                    total_cells++;
                }
            }
        }

        return static_cast<double>(count) / std::max(1, total_cells);
    }

    /**
     * @brief Get bounding box
     */
    std::pair<V3D, V3D> getBoundingBox() const {
        return {min_bound_, max_bound_};
    }

    /**
     * @brief Get voxel size
     */
    double voxelSize() const { return voxel_size_; }

    /**
     * @brief Get number of occupied voxels
     */
    size_t numVoxels() const { return occupied_voxels_.size(); }

    /**
     * @brief Clear map
     */
    void clear() {
        occupied_voxels_.clear();
        min_bound_ = V3D::Zero();
        max_bound_ = V3D::Zero();
    }

private:
    double voxel_size_;
    double inv_voxel_size_;
    std::unordered_set<Eigen::Vector3i, VoxelKeyHash> occupied_voxels_;
    V3D min_bound_ = V3D::Zero();
    V3D max_bound_ = V3D::Zero();

    Eigen::Vector3i toVoxelKey(const V3D& pt) const {
        return Eigen::Vector3i(
            static_cast<int>(std::floor(pt.x() * inv_voxel_size_)),
            static_cast<int>(std::floor(pt.y() * inv_voxel_size_)),
            static_cast<int>(std::floor(pt.z() * inv_voxel_size_))
        );
    }
};

//=============================================================================
// Curvature Estimator
//=============================================================================

/**
 * @brief Estimate local surface curvature for curved surface matching
 *
 * Particularly useful for ship hull localization where position
 * along the hull correlates with curvature (bow is sharp, midship is gentle).
 */
class CurvatureEstimator {
public:
    /**
     * @brief Estimate mean curvature at each point using PCA
     * @param points Input point cloud
     * @param k Number of neighbors for local PCA
     * @return Vector of curvature values (0 = flat, higher = more curved)
     */
    static std::vector<double> estimate(const std::vector<V3D>& points, int k = 20) {
        std::vector<double> curvatures(points.size(), 0.0);

        if (points.size() < static_cast<size_t>(k)) return curvatures;

        // Simple brute-force neighbor search (for small point clouds)
        // For large clouds, would use KD-tree
        for (size_t i = 0; i < points.size(); i++) {
            // Find k nearest neighbors
            std::vector<std::pair<double, size_t>> distances;
            distances.reserve(points.size());

            for (size_t j = 0; j < points.size(); j++) {
                if (i == j) continue;
                double dist = (points[i] - points[j]).squaredNorm();
                distances.push_back({dist, j});
            }

            std::partial_sort(distances.begin(),
                            distances.begin() + std::min(k, static_cast<int>(distances.size())),
                            distances.end(),
                            [](const auto& a, const auto& b) { return a.first < b.first; });

            // Compute covariance of neighbors
            V3D mean = V3D::Zero();
            int num_neighbors = std::min(k, static_cast<int>(distances.size()));

            for (int n = 0; n < num_neighbors; n++) {
                mean += points[distances[n].second];
            }
            mean /= num_neighbors;

            M3D cov = M3D::Zero();
            for (int n = 0; n < num_neighbors; n++) {
                V3D d = points[distances[n].second] - mean;
                cov += d * d.transpose();
            }
            cov /= num_neighbors;

            // Curvature from eigenvalues
            Eigen::SelfAdjointEigenSolver<M3D> solver(cov);
            V3D eigenvalues = solver.eigenvalues();

            // Curvature = smallest eigenvalue / sum (0 = flat plane, 0.33 = sphere)
            double sum = eigenvalues.sum();
            if (sum > 1e-10) {
                curvatures[i] = eigenvalues.minCoeff() / sum;
            }
        }

        return curvatures;
    }

    /**
     * @brief Compute curvature histogram for a scan
     * @param points Input point cloud
     * @param num_bins Number of histogram bins
     * @return Normalized histogram of curvature values
     */
    static Eigen::VectorXd curvatureHistogram(const std::vector<V3D>& points, int num_bins = 20) {
        auto curvatures = estimate(points, 20);

        Eigen::VectorXd hist = Eigen::VectorXd::Zero(num_bins);

        for (double c : curvatures) {
            // Curvature typically 0-0.5, map to bins
            int bin = static_cast<int>(c * num_bins * 2);
            bin = std::clamp(bin, 0, num_bins - 1);
            hist(bin)++;
        }

        // Normalize
        double sum = hist.sum();
        if (sum > 0) hist /= sum;

        return hist;
    }
};

//=============================================================================
// Global Localizer Result Structures
//=============================================================================

/**
 * @brief Confidence metrics for localization result
 */
struct LocalizationConfidence {
    double fitness_score = 0.0;        // Inlier ratio (0-1)
    double distinctiveness = 0.0;       // Gap to second-best (0-1)
    double geometric_quality = 0.0;     // Hessian condition (0-1)
    double physical_plausibility = 0.0; // Passes sanity checks (0-1)
    int verification_level = 0;         // How far through pipeline (1-4)

    double overall() const {
        return fitness_score * 0.35 +
               distinctiveness * 0.25 +
               geometric_quality * 0.20 +
               physical_plausibility * 0.20;
    }

    bool isAcceptable() const {
        return fitness_score >= 0.45 &&
               distinctiveness >= 0.08 &&
               overall() >= 0.50;
    }

    bool isHighlyReliable() const {
        return fitness_score >= 0.65 &&
               distinctiveness >= 0.15 &&
               geometric_quality >= 0.40 &&
               overall() >= 0.70;
    }

    std::string describe() const {
        if (isHighlyReliable()) return "HIGH - position confirmed";
        if (isAcceptable()) return "MEDIUM - position likely correct";
        if (fitness_score >= 0.30) return "LOW - position uncertain";
        return "FAILED - could not localize";
    }
};

/**
 * @brief Localization status enum
 */
enum class LocalizationStatus {
    ACCUMULATING_SCAN,
    COMPUTING_DESCRIPTORS,
    SEARCHING_CANDIDATES,
    COARSE_VERIFICATION,
    MEDIUM_VERIFICATION,
    FINE_VERIFICATION,
    DISAMBIGUATION,
    SUCCESS,
    AMBIGUOUS,
    FAILED
};

/**
 * @brief Progress information for UI feedback
 */
struct LocalizationProgress {
    LocalizationStatus status = LocalizationStatus::ACCUMULATING_SCAN;
    double elapsed_time = 0.0;
    double estimated_remaining = 0.0;
    int candidates_evaluated = 0;
    int candidates_remaining = 0;
    double current_best_fitness = 0.0;
    std::string message;
};

/**
 * @brief Final localization result
 */
struct LocalizationResult {
    M4D pose = M4D::Identity();
    LocalizationConfidence confidence;
    LocalizationStatus status = LocalizationStatus::FAILED;
    double elapsed_time = 0.0;
    std::string message;

    // For debugging/visualization
    std::vector<std::pair<M4D, double>> top_hypotheses;

    bool success() const {
        return status == LocalizationStatus::SUCCESS ||
               status == LocalizationStatus::AMBIGUOUS;
    }
};

//=============================================================================
// Global Localizer
//=============================================================================

/**
 * @brief Main global localization class
 *
 * Orchestrates the complete localization pipeline:
 * 1. Scan Context place recognition
 * 2. Grid search fallback (if needed)
 * 3. Multi-level geometric verification
 * 4. Disambiguation for symmetric environments
 */
class GlobalLocalizer {
public:
    using ProgressCallback = std::function<void(const LocalizationProgress&)>;

    explicit GlobalLocalizer(const GlobalLocalizerConfig& config = GlobalLocalizerConfig())
        : config_(config),
          sc_database_(config.sc_config),
          coarse_map_(config.coarse_voxel_size),
          medium_map_(config.medium_voxel_size),
          fine_map_(config.fine_voxel_size) {}

    /**
     * @brief Set map data for localization
     * @param map_points Full map point cloud
     */
    void setMap(const std::vector<V3D>& map_points) {
        map_points_ = map_points;

        // Build occupancy maps at multiple resolutions
        coarse_map_.build(map_points);
        medium_map_.build(map_points);
        fine_map_.build(map_points);

        // Compute bounding box
        auto [min_pt, max_pt] = coarse_map_.getBoundingBox();
        map_min_ = min_pt;
        map_max_ = max_pt;

        has_map_ = true;
    }

    /**
     * @brief Set Scan Context database (built during mapping)
     */
    void setScanContextDatabase(const ScanContextDatabase& db) {
        sc_database_ = db;
    }

    /**
     * @brief Get mutable reference to database for building
     */
    ScanContextDatabase& database() { return sc_database_; }

    /**
     * @brief Set gravity alignment from IMU
     * @param gravity_R Rotation that aligns sensor frame to world (Z-up)
     */
    void setGravityAlignment(const M3D& gravity_R) {
        gravity_alignment_ = gravity_R;
        has_gravity_ = true;
    }

    /**
     * @brief Main localization entry point
     * @param scan Accumulated scan in sensor frame
     * @param progress_cb Optional callback for progress updates
     * @return Localization result with pose and confidence
     */
    LocalizationResult localize(const std::vector<V3D>& scan,
                                 ProgressCallback progress_cb = nullptr) {
        auto start_time = std::chrono::steady_clock::now();
        LocalizationResult result;

        auto elapsed = [&start_time]() {
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time).count();
        };

        auto reportProgress = [&](LocalizationStatus status, const std::string& msg) {
            if (progress_cb) {
                LocalizationProgress p;
                p.status = status;
                p.elapsed_time = elapsed();
                p.message = msg;
                progress_cb(p);
            }
        };

        // Validate inputs
        if (!has_map_) {
            result.status = LocalizationStatus::FAILED;
            result.message = "No map loaded";
            return result;
        }

        if (scan.size() < 1000) {
            result.status = LocalizationStatus::FAILED;
            result.message = "Insufficient scan data";
            return result;
        }

        //=====================================================================
        // Phase 1: Place recognition
        //=====================================================================
        reportProgress(LocalizationStatus::COMPUTING_DESCRIPTORS, "Analyzing scan...");

        std::vector<Candidate> candidates;

        if (!sc_database_.empty()) {
            auto sc_matches = sc_database_.query(scan, config_.scan_context_candidates);

            for (const auto& m : sc_matches) {
                Candidate c;
                c.position = m.position;
                c.yaw = m.yaw_estimate;
                c.score = m.score;
                c.source = "ScanContext";
                candidates.push_back(c);
            }
        }

        // Check if place recognition gave good candidates
        bool sc_confident = !candidates.empty() && candidates[0].score > 0.5;

        //=====================================================================
        // Phase 2: Grid search fallback
        //=====================================================================
        if (!sc_confident) {
            reportProgress(LocalizationStatus::SEARCHING_CANDIDATES,
                          "Searching all locations...");

            auto grid_candidates = gridSearch(scan, 5.0);  // 5 second budget

            for (const auto& gc : grid_candidates) {
                candidates.push_back(gc);
            }
        }

        if (candidates.empty()) {
            result.status = LocalizationStatus::FAILED;
            result.message = "No candidate locations found";
            result.elapsed_time = elapsed();
            return result;
        }

        //=====================================================================
        // Phase 3: Coarse verification
        //=====================================================================
        reportProgress(LocalizationStatus::COARSE_VERIFICATION,
                      "Verifying candidates...");

        // Score all candidates with voxel occupancy
        for (auto& c : candidates) {
            M4D pose = constructPose(c.position, c.yaw);
            c.score = coarse_map_.scorePose(scan, pose, 200);
        }

        // Sort and keep top candidates
        std::sort(candidates.begin(), candidates.end(),
                 [](const Candidate& a, const Candidate& b) {
                     return a.score > b.score;
                 });

        int num_coarse = std::min(config_.coarse_verification_candidates,
                                   static_cast<int>(candidates.size()));
        candidates.resize(num_coarse);

        // Coarse ICP on top candidates
        std::vector<VerifiedCandidate> verified;
        verified.reserve(num_coarse);

        ICPConfig coarse_icp_config;
        coarse_icp_config.max_iterations = 10;
        coarse_icp_config.max_correspondence_dist = 3.0;
        coarse_icp_config.convergence_threshold = 1e-4;
        coarse_icp_config.method = ICPMethod::POINT_TO_POINT;

        auto scan_down = voxelDownsample(scan, config_.coarse_voxel_size);
        auto map_down = voxelDownsample(map_points_, config_.coarse_voxel_size);

        ICP coarse_icp(coarse_icp_config);

        for (const auto& c : candidates) {
            M4D initial_pose = constructPose(c.position, c.yaw);
            auto icp_result = coarse_icp.align(scan_down, map_down, initial_pose);

            VerifiedCandidate vc;
            vc.pose = icp_result.transformation;
            vc.fitness = icp_result.fitness_score;
            vc.converged = icp_result.converged;
            vc.source = c.source;
            verified.push_back(vc);
        }

        // Sort by fitness
        std::sort(verified.begin(), verified.end(),
                 [](const VerifiedCandidate& a, const VerifiedCandidate& b) {
                     return a.fitness > b.fitness;
                 });

        //=====================================================================
        // Phase 4: Medium verification
        //=====================================================================
        reportProgress(LocalizationStatus::MEDIUM_VERIFICATION,
                      "Refining positions...");

        int num_medium = std::min(config_.medium_verification_candidates,
                                   static_cast<int>(verified.size()));
        verified.resize(num_medium);

        ICPConfig medium_icp_config;
        medium_icp_config.max_iterations = 30;
        medium_icp_config.max_correspondence_dist = 1.0;
        medium_icp_config.convergence_threshold = 1e-6;
        medium_icp_config.method = ICPMethod::POINT_TO_PLANE;

        auto scan_medium = voxelDownsample(scan, config_.medium_voxel_size);
        auto map_medium = voxelDownsample(map_points_, config_.medium_voxel_size);

        ICP medium_icp(medium_icp_config);

        for (auto& vc : verified) {
            auto icp_result = medium_icp.align(scan_medium, map_medium, vc.pose);
            vc.pose = icp_result.transformation;
            vc.fitness = icp_result.fitness_score;
            vc.converged = icp_result.converged;
        }

        std::sort(verified.begin(), verified.end(),
                 [](const VerifiedCandidate& a, const VerifiedCandidate& b) {
                     return a.fitness > b.fitness;
                 });

        //=====================================================================
        // Phase 5: Fine verification
        //=====================================================================
        reportProgress(LocalizationStatus::FINE_VERIFICATION,
                      "Final refinement...");

        int num_fine = std::min(config_.fine_verification_candidates,
                                 static_cast<int>(verified.size()));
        verified.resize(num_fine);

        ICPConfig fine_icp_config;
        fine_icp_config.max_iterations = 50;
        fine_icp_config.max_correspondence_dist = 0.5;
        fine_icp_config.convergence_threshold = 1e-7;
        fine_icp_config.method = ICPMethod::POINT_TO_PLANE;

        auto scan_fine = voxelDownsample(scan, config_.fine_voxel_size);
        auto map_fine = voxelDownsample(map_points_, config_.fine_voxel_size);

        ICP fine_icp(fine_icp_config);

        for (auto& vc : verified) {
            auto icp_result = fine_icp.align(scan_fine, map_fine, vc.pose);
            vc.pose = icp_result.transformation;
            vc.fitness = icp_result.fitness_score;
            vc.rmse = icp_result.rmse;
            vc.converged = icp_result.converged;

            // Physical plausibility check
            vc.is_plausible = checkPhysicalPlausibility(vc.pose);

            // Geometric quality (simplified)
            vc.geometric_quality = vc.converged ? 0.8 : 0.3;
        }

        std::sort(verified.begin(), verified.end(),
                 [](const VerifiedCandidate& a, const VerifiedCandidate& b) {
                     return a.fitness > b.fitness;
                 });

        //=====================================================================
        // Phase 6: Disambiguation
        //=====================================================================
        reportProgress(LocalizationStatus::DISAMBIGUATION,
                      "Confirming location...");

        // Calculate distinctiveness
        double distinctiveness = 0.0;
        if (verified.size() >= 2) {
            distinctiveness = (verified[0].fitness - verified[1].fitness) /
                             std::max(0.01, verified[0].fitness);
        } else if (verified.size() == 1) {
            distinctiveness = 1.0;  // Only one candidate
        }

        // Build result
        if (!verified.empty()) {
            const auto& best = verified[0];

            result.pose = best.pose;
            result.confidence.fitness_score = best.fitness;
            result.confidence.distinctiveness = distinctiveness;
            result.confidence.geometric_quality = best.geometric_quality;
            result.confidence.physical_plausibility = best.is_plausible ? 1.0 : 0.5;
            result.confidence.verification_level = 4;

            // Store top hypotheses
            for (const auto& vc : verified) {
                result.top_hypotheses.push_back({vc.pose, vc.fitness});
            }

            // Determine status
            if (result.confidence.isHighlyReliable()) {
                result.status = LocalizationStatus::SUCCESS;
                result.message = "Localization successful. " +
                                result.confidence.describe();
            } else if (result.confidence.isAcceptable()) {
                if (distinctiveness < 0.10) {
                    result.status = LocalizationStatus::AMBIGUOUS;
                    result.message = "Position found but multiple similar locations exist. "
                                    "Moving slightly may help confirm.";
                } else {
                    result.status = LocalizationStatus::SUCCESS;
                    result.message = "Localization successful. " +
                                    result.confidence.describe();
                }
            } else {
                result.status = LocalizationStatus::FAILED;
                result.message = "Could not reliably determine position. "
                                "Try a location with more distinctive features.";
            }
        } else {
            result.status = LocalizationStatus::FAILED;
            result.message = "No valid candidates after verification.";
        }

        result.elapsed_time = elapsed();
        return result;
    }

    /**
     * @brief Check if map is loaded
     */
    bool hasMap() const { return has_map_; }

    /**
     * @brief Get configuration
     */
    const GlobalLocalizerConfig& config() const { return config_; }

private:
    struct Candidate {
        V3D position;
        double yaw;
        double score;
        std::string source;
    };

    struct VerifiedCandidate {
        M4D pose;
        double fitness;
        double rmse;
        bool converged;
        bool is_plausible;
        double geometric_quality;
        std::string source;
    };

    GlobalLocalizerConfig config_;
    ScanContextDatabase sc_database_;
    VoxelOccupancyMap coarse_map_;
    VoxelOccupancyMap medium_map_;
    VoxelOccupancyMap fine_map_;
    std::vector<V3D> map_points_;

    V3D map_min_ = V3D::Zero();
    V3D map_max_ = V3D::Zero();
    M3D gravity_alignment_ = M3D::Identity();

    bool has_map_ = false;
    bool has_gravity_ = false;

    /**
     * @brief Construct 4x4 pose from position and yaw
     */
    M4D constructPose(const V3D& position, double yaw) const {
        M3D yaw_R = Eigen::AngleAxisd(yaw, V3D::UnitZ()).toRotationMatrix();
        M3D full_R = yaw_R;

        if (has_gravity_) {
            full_R = yaw_R * gravity_alignment_;
        }

        M4D pose = M4D::Identity();
        pose.block<3,3>(0,0) = full_R;
        pose.block<3,1>(0,3) = position;
        return pose;
    }

    /**
     * @brief Grid search fallback when place recognition fails
     */
    std::vector<Candidate> gridSearch(const std::vector<V3D>& scan,
                                       double time_budget_seconds) {
        auto start_time = std::chrono::steady_clock::now();
        std::vector<Candidate> candidates;

        double step_xy = config_.grid_resolution_xy;
        double step_yaw = config_.grid_resolution_yaw * M_PI / 180.0;

        // Subsample scan for fast scoring
        auto scan_sample = voxelDownsample(scan, 1.0);
        if (scan_sample.size() > 300) {
            std::vector<V3D> subsampled;
            int step = scan_sample.size() / 300;
            for (size_t i = 0; i < scan_sample.size(); i += step) {
                subsampled.push_back(scan_sample[i]);
            }
            scan_sample = subsampled;
        }

        // Build search order (prioritize areas with features)
        std::vector<std::pair<V3D, double>> search_positions;

        for (double x = map_min_.x(); x <= map_max_.x(); x += step_xy) {
            for (double y = map_min_.y(); y <= map_max_.y(); y += step_xy) {
                double density = coarse_map_.getFeatureDensity(x, y);
                double z = coarse_map_.estimateFloorZ(x, y);
                search_positions.push_back({V3D(x, y, z), density});
            }
        }

        // Sort by density (search feature-rich areas first)
        std::sort(search_positions.begin(), search_positions.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });

        for (const auto& [pos, density] : search_positions) {
            // Check time budget
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (std::chrono::duration<double>(elapsed).count() > time_budget_seconds) {
                break;
            }

            for (double yaw = 0; yaw < 2 * M_PI; yaw += step_yaw) {
                M4D pose = constructPose(pos, yaw);
                double score = coarse_map_.scorePose(scan_sample, pose, 200);

                if (score > 0.20) {  // Promising candidate
                    Candidate c;
                    c.position = pos;
                    c.yaw = yaw;
                    c.score = score;
                    c.source = "GridSearch";
                    candidates.push_back(c);
                }
            }
        }

        // Sort by score
        std::sort(candidates.begin(), candidates.end(),
                 [](const Candidate& a, const Candidate& b) {
                     return a.score > b.score;
                 });

        // Limit number of candidates
        if (candidates.size() > static_cast<size_t>(config_.grid_search_candidates)) {
            candidates.resize(config_.grid_search_candidates);
        }

        return candidates;
    }

    /**
     * @brief Check if pose is physically plausible
     */
    bool checkPhysicalPlausibility(const M4D& pose) const {
        V3D position = pose.block<3,1>(0,3);

        // Check within map bounds (with margin)
        V3D margin(5.0, 5.0, 3.0);
        if ((position.array() < (map_min_ - margin).array()).any() ||
            (position.array() > (map_max_ + margin).array()).any()) {
            return false;
        }

        // Check Z is near expected floor
        double expected_z = coarse_map_.estimateFloorZ(position.x(), position.y());
        if (std::abs(position.z() - expected_z) > config_.max_floor_deviation) {
            return false;
        }

        return true;
    }
};

} // namespace slam

#endif // SLAM_GLOBAL_LOCALIZATION_HPP
