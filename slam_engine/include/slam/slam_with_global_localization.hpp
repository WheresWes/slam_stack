/**
 * @file slam_with_global_localization.hpp
 * @brief SLAM Engine extension with global localization capabilities
 *
 * This wrapper adds global localization support to the SLAM engine,
 * enabling localization from completely unknown initial positions.
 *
 * Designed for challenging environments like ship hulls in dry dock.
 */

#ifndef SLAM_WITH_GLOBAL_LOCALIZATION_HPP
#define SLAM_WITH_GLOBAL_LOCALIZATION_HPP

#include "slam/slam_engine.hpp"
#include "slam/global_localization.hpp"

namespace slam {

/**
 * @brief SLAM Engine with Global Localization Support
 *
 * Extends basic SLAM functionality with:
 * - Scan Context keyframe database building during mapping
 * - Global localization from completely unknown positions
 * - Progress callbacks for UI feedback
 */
class SlamWithGlobalLocalization {
public:
    using ProgressCallback = GlobalLocalizer::ProgressCallback;

    SlamWithGlobalLocalization() = default;

    /**
     * @brief Initialize with configuration
     */
    bool init(const SlamConfig& slam_config,
              const GlobalLocalizerConfig& global_config = GlobalLocalizerConfig()) {
        slam_config_ = slam_config;
        global_config_ = global_config;

        if (!engine_.init(slam_config)) {
            return false;
        }

        globalLocalizer_ = std::make_unique<GlobalLocalizer>(global_config);
        return true;
    }

    /**
     * @brief Get underlying SLAM engine (for direct access)
     */
    SlamEngine& engine() { return engine_; }
    const SlamEngine& engine() const { return engine_; }

    //=========================================================================
    // Mapping Phase: Building Scan Context Database
    //=========================================================================

    /**
     * @brief Process a scan and optionally add to Scan Context database
     *
     * Call this after engine_.processScan() to build the localization database.
     * Should be called at regular intervals (e.g., every 5 meters of motion).
     */
    void addKeyFrameIfNeeded() {
        if (!engine_.isInitialized()) return;

        // Get current pose
        M4D pose = engine_.getPose();
        V3D position = pose.block<3,1>(0,3);

        // Check if far enough from last keyframe
        if (keyframe_count_ > 0) {
            double dist = (position - last_keyframe_position_).norm();
            if (dist < keyframe_distance_threshold_) {
                return;  // Too close to last keyframe
            }
        }

        // Get current world-frame point cloud
        auto world_points = engine_.getMapPoints();
        if (world_points.empty()) return;

        // Convert to sensor-frame points for Scan Context
        M4D pose_inv = pose.inverse();
        M3D R_inv = pose_inv.block<3,3>(0,0);
        V3D t_inv = pose_inv.block<3,1>(0,3);

        std::vector<V3D> sensor_points;
        sensor_points.reserve(world_points.size());

        // Only use recent points (within sensor range)
        for (const auto& wp : world_points) {
            V3D world_pt(wp.x, wp.y, wp.z);
            V3D sensor_pt = R_inv * world_pt + t_inv;
            double range = sensor_pt.head<2>().norm();

            if (range < global_config_.sc_config.max_radius &&
                range > global_config_.sc_config.min_radius) {
                sensor_points.push_back(sensor_pt);
            }
        }

        if (sensor_points.size() < 500) {
            return;  // Not enough points for reliable descriptor
        }

        // Add to database
        double timestamp = 0.0;  // Could get from engine if needed
        if (globalLocalizer_->database().addKeyFrame(sensor_points, pose, timestamp)) {
            last_keyframe_position_ = position;
            keyframe_count_++;
            std::cout << "[GlobalLoc] Added keyframe " << keyframe_count_
                     << " at position [" << position.x() << ", "
                     << position.y() << ", " << position.z() << "]" << std::endl;
        }
    }

    /**
     * @brief Finalize map building and prepare for localization
     *
     * Call this after mapping is complete to prepare the global localizer.
     */
    void finalizeMapping() {
        // Get all map points
        auto world_points = engine_.getMapPoints();
        if (world_points.empty()) {
            std::cerr << "[GlobalLoc] Warning: No map points for localization" << std::endl;
            return;
        }

        // Convert to V3D vector
        std::vector<V3D> map_points;
        map_points.reserve(world_points.size());
        for (const auto& wp : world_points) {
            map_points.emplace_back(wp.x, wp.y, wp.z);
        }

        // Set map in global localizer
        globalLocalizer_->setMap(map_points);

        // Set gravity alignment if available from IMU processor
        // (This would need access to the IMU processor's gravity alignment)

        std::cout << "[GlobalLoc] Mapping finalized:" << std::endl;
        std::cout << "  - Map points: " << map_points.size() << std::endl;
        std::cout << "  - Keyframes: " << globalLocalizer_->database().size() << std::endl;

        auto [min_pt, max_pt] = globalLocalizer_->database().getBoundingBox();
        std::cout << "  - Bounds: [" << min_pt.x() << ", " << min_pt.y() << "] to ["
                 << max_pt.x() << ", " << max_pt.y() << "]" << std::endl;

        mapping_finalized_ = true;
    }

    //=========================================================================
    // Localization Phase
    //=========================================================================

    /**
     * @brief Perform global localization from completely unknown position
     *
     * @param scan Accumulated scan in sensor frame
     * @param progress_cb Optional callback for progress updates
     * @return Localization result with pose and confidence
     *
     * Usage:
     * 1. Accumulate scan data while stationary (1-2 seconds)
     * 2. Call this function with the accumulated scan
     * 3. Check result.success() and result.confidence
     * 4. If successful, use result.pose to initialize SLAM tracking
     */
    LocalizationResult globalLocalize(const std::vector<V3D>& scan,
                                       ProgressCallback progress_cb = nullptr) {
        if (!mapping_finalized_) {
            LocalizationResult result;
            result.status = LocalizationStatus::FAILED;
            result.message = "Mapping not finalized. Call finalizeMapping() first.";
            return result;
        }

        return globalLocalizer_->localize(scan, progress_cb);
    }

    /**
     * @brief Convenience overload: accumulate from current scan buffer
     *
     * Uses the most recent accumulated scan from the SLAM engine.
     */
    LocalizationResult globalLocalize(ProgressCallback progress_cb = nullptr) {
        // Get current scan from engine
        auto current_scan = engine_.getCurrentScan();

        if (current_scan.empty()) {
            LocalizationResult result;
            result.status = LocalizationStatus::FAILED;
            result.message = "No scan data available.";
            return result;
        }

        // Convert to V3D
        std::vector<V3D> scan_points;
        scan_points.reserve(current_scan.size());
        for (const auto& pt : current_scan) {
            scan_points.emplace_back(pt.x, pt.y, pt.z);
        }

        // Transform to sensor frame
        M4D pose = engine_.getPose();
        M4D pose_inv = pose.inverse();
        M3D R_inv = pose_inv.block<3,3>(0,0);
        V3D t_inv = pose_inv.block<3,1>(0,3);

        std::vector<V3D> sensor_points;
        sensor_points.reserve(scan_points.size());
        for (const auto& pt : scan_points) {
            sensor_points.push_back(R_inv * pt + t_inv);
        }

        return globalLocalizer_->localize(sensor_points, progress_cb);
    }

    /**
     * @brief Apply localization result to SLAM engine
     *
     * Call this after successful global localization to initialize tracking.
     */
    bool applyLocalizationResult(const LocalizationResult& result) {
        if (!result.success()) {
            return false;
        }

        // PRESERVE BIASES - IMU has been running during global localization
        engine_.setInitialPose(result.pose, true);
        std::cout << "[GlobalLoc] Applied localization result. Confidence: "
                 << result.confidence.describe() << std::endl;
        return true;
    }

    //=========================================================================
    // Map Save/Load with Scan Context Database
    //=========================================================================

    /**
     * @brief Save map and Scan Context database
     */
    bool saveMapWithDatabase(const std::string& map_filename,
                              const std::string& database_filename) {
        // Save map
        if (!engine_.saveMap(map_filename)) {
            return false;
        }

        // TODO: Serialize Scan Context database to file
        // For now, just save map
        std::cout << "[GlobalLoc] Map saved. Database serialization not yet implemented."
                 << std::endl;

        return true;
    }

    /**
     * @brief Load map and Scan Context database
     */
    bool loadMapWithDatabase(const std::string& map_filename,
                              const std::string& database_filename) {
        // Load map into SLAM engine
        if (!engine_.loadMap(map_filename)) {
            return false;
        }

        // Get map points from engine
        auto world_points = engine_.getMapPoints();
        std::vector<V3D> map_points;
        map_points.reserve(world_points.size());
        for (const auto& wp : world_points) {
            map_points.emplace_back(wp.x, wp.y, wp.z);
        }

        // Set in global localizer
        globalLocalizer_->setMap(map_points);

        // TODO: Load Scan Context database from file
        std::cout << "[GlobalLoc] Map loaded. Database loading not yet implemented."
                 << std::endl;
        std::cout << "[GlobalLoc] You may need to rebuild keyframes from the map."
                 << std::endl;

        mapping_finalized_ = true;
        return true;
    }

    /**
     * @brief Rebuild Scan Context database from loaded map
     *
     * Use this after loading a map that doesn't have a saved database.
     * Creates keyframes at regular grid positions.
     */
    void rebuildDatabaseFromMap(double keyframe_spacing = 5.0) {
        if (!engine_.isLocalizationMode()) {
            std::cerr << "[GlobalLoc] Must load map first" << std::endl;
            return;
        }

        auto world_points = engine_.getMapPoints();
        if (world_points.empty()) return;

        // Find map bounds
        V3D min_pt(std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max());
        V3D max_pt(std::numeric_limits<double>::lowest(),
                   std::numeric_limits<double>::lowest(),
                   std::numeric_limits<double>::lowest());

        for (const auto& pt : world_points) {
            min_pt.x() = std::min(min_pt.x(), static_cast<double>(pt.x));
            min_pt.y() = std::min(min_pt.y(), static_cast<double>(pt.y));
            min_pt.z() = std::min(min_pt.z(), static_cast<double>(pt.z));
            max_pt.x() = std::max(max_pt.x(), static_cast<double>(pt.x));
            max_pt.y() = std::max(max_pt.y(), static_cast<double>(pt.y));
            max_pt.z() = std::max(max_pt.z(), static_cast<double>(pt.z));
        }

        std::cout << "[GlobalLoc] Rebuilding database from map..." << std::endl;
        std::cout << "  Map bounds: [" << min_pt.x() << ", " << min_pt.y() << "] to ["
                 << max_pt.x() << ", " << max_pt.y() << "]" << std::endl;

        // Convert all points
        std::vector<V3D> all_points;
        all_points.reserve(world_points.size());
        for (const auto& wp : world_points) {
            all_points.emplace_back(wp.x, wp.y, wp.z);
        }

        // Create synthetic keyframes at grid positions
        int keyframes_added = 0;
        double floor_z = min_pt.z() + 1.0;  // Assume 1m above floor

        for (double x = min_pt.x(); x <= max_pt.x(); x += keyframe_spacing) {
            for (double y = min_pt.y(); y <= max_pt.y(); y += keyframe_spacing) {
                // Check if this position has enough nearby points
                V3D position(x, y, floor_z);

                // Collect points in sensor range
                std::vector<V3D> local_points;
                for (const auto& pt : all_points) {
                    V3D relative = pt - position;
                    double range = relative.head<2>().norm();
                    if (range < global_config_.sc_config.max_radius &&
                        range > global_config_.sc_config.min_radius) {
                        local_points.push_back(relative);
                    }
                }

                if (local_points.size() < 1000) continue;  // Not enough points

                // Create keyframe at multiple yaw angles
                for (double yaw = 0; yaw < 2 * M_PI; yaw += M_PI / 2) {
                    M3D R = Eigen::AngleAxisd(yaw, V3D::UnitZ()).toRotationMatrix();
                    M4D pose = M4D::Identity();
                    pose.block<3,3>(0,0) = R;
                    pose.block<3,1>(0,3) = position;

                    // Rotate points to sensor frame
                    std::vector<V3D> rotated_points;
                    rotated_points.reserve(local_points.size());
                    M3D R_inv = R.transpose();
                    for (const auto& pt : local_points) {
                        rotated_points.push_back(R_inv * pt);
                    }

                    if (globalLocalizer_->database().addKeyFrame(rotated_points, pose, 0.0)) {
                        keyframes_added++;
                    }
                }
            }
        }

        std::cout << "  Added " << keyframes_added << " synthetic keyframes" << std::endl;
        mapping_finalized_ = true;
    }

    //=========================================================================
    // Configuration
    //=========================================================================

    /**
     * @brief Set minimum distance between keyframes (during mapping)
     */
    void setKeyFrameDistanceThreshold(double distance) {
        keyframe_distance_threshold_ = distance;
        globalLocalizer_->database().setMinKeyFrameDistance(distance);
    }

    /**
     * @brief Get global localizer configuration
     */
    const GlobalLocalizerConfig& globalConfig() const { return global_config_; }

    /**
     * @brief Check if mapping has been finalized
     */
    bool isMappingFinalized() const { return mapping_finalized_; }

    /**
     * @brief Get number of keyframes in database
     */
    size_t numKeyFrames() const {
        return globalLocalizer_ ? globalLocalizer_->database().size() : 0;
    }

private:
    SlamEngine engine_;
    SlamConfig slam_config_;
    GlobalLocalizerConfig global_config_;

    std::unique_ptr<GlobalLocalizer> globalLocalizer_;

    // Keyframe tracking during mapping
    V3D last_keyframe_position_ = V3D::Zero();
    int keyframe_count_ = 0;
    double keyframe_distance_threshold_ = 5.0;  // meters

    bool mapping_finalized_ = false;
};

} // namespace slam

#endif // SLAM_WITH_GLOBAL_LOCALIZATION_HPP
