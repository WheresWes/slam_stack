/**
 * @file slam_engine.hpp
 * @brief Main SLAM engine - ROS-free port of FAST-LIO
 *
 * This is the core SLAM processing engine, ported from laserMapping.cpp
 * with all ROS dependencies removed.
 *
 * Key features:
 * - Gravity alignment at startup
 * - Intensity preservation throughout pipeline
 * - Callback-based sensor interface
 * - Direct state access (no ROS publishers)
 */

#ifndef SLAM_ENGINE_HPP
#define SLAM_ENGINE_HPP

#include <mutex>
#include <deque>
#include <thread>
#include <atomic>
#include <functional>
#include <condition_variable>

#include <Eigen/Core>
#include <Eigen/Geometry>

// PCL for point cloud processing
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

#include "slam/types.hpp"
#include "slam/so3_math.hpp"
#include "slam/preprocess.hpp"
#include "slam/imu_processing.hpp"
#include "slam/ply_export.hpp"
#include "slam/icp.hpp"

// PCL I/O for map loading
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

// ikd-Tree for map storage
#include "slam/ikd-Tree/ikd_Tree.h"

// Point type with intensity for map storage
typedef pcl::PointXYZINormal PointType;
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;

// IKFoM state definitions
#include "slam/use_ikfom.hpp"

namespace slam {

//=============================================================================
// Configuration
//=============================================================================

struct SlamConfig {
    // IMU noise parameters
    double gyr_cov = 0.1;
    double acc_cov = 0.1;
    double b_gyr_cov = 0.0001;
    double b_acc_cov = 0.0001;

    // Map parameters
    double filter_size_surf = 0.5;      // Downsampling voxel size for scan
    double filter_size_map = 0.5;       // Map voxel size
    double cube_len = 200.0;            // Local map cube size
    double det_range = 300.0;           // Detection range
    double fov_deg = 180.0;             // Field of view

    // Optimization
    int max_iterations = 4;

    // LiDAR-IMU extrinsics
    V3D extrinsic_T = V3D::Zero();
    M3D extrinsic_R = M3D::Identity();
    bool extrinsic_est_en = false;

    // Motion compensation
    bool deskew_enabled = true;

    // Output
    bool dense_output = true;           // Output dense (vs downsampled) clouds
    bool save_map = false;
    std::string map_save_path = "map.ply";
};

//=============================================================================
// Callbacks
//=============================================================================

using StateCallback = std::function<void(const SlamState&)>;
using PointCloudCallback = std::function<void(const PointCloud&, const M4D&)>;
using TrajectoryCallback = std::function<void(const std::vector<M4D>&)>;

//=============================================================================
// SLAM Engine Class
//=============================================================================

/**
 * @brief Main SLAM processing engine
 *
 * Usage:
 *   SlamEngine slam;
 *   slam.init(config);
 *   slam.setStateCallback(...);  // Optional: receive state updates
 *
 *   // Feed sensor data:
 *   slam.addImuData(imu);
 *   slam.addPointCloud(cloud, timestamp);
 *
 *   // Or process synchronized data:
 *   slam.processMeasurement(meas);
 *
 *   // Get results:
 *   SlamState state = slam.getState();
 *   auto map_points = slam.getMapPoints();
 */
class SlamEngine {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Static callback wrapper for EKF (needed because capturing lambdas can't convert to function pointers)
    static SlamEngine* s_instance_;
    static void staticHShareModel(state_ikfom& s, esekfom::dyn_share_datastruct<double>& d) {
        if (s_instance_) s_instance_->hShareModel(s, d);
    }

    SlamEngine();
    ~SlamEngine();

    /**
     * @brief Initialize the SLAM engine
     */
    bool init(const SlamConfig& config);

    /**
     * @brief Reset to initial state
     */
    void reset();

    //=========================================================================
    // Sensor Data Input
    //=========================================================================

    /**
     * @brief Add raw IMU measurement
     */
    void addImuData(const ImuData& imu);

    /**
     * @brief Add preprocessed point cloud
     */
    void addPointCloud(const PointCloud& cloud);

    /**
     * @brief Add raw Livox SDK2 points (will preprocess internally)
     */
    void addRawPoints(const LivoxPointXYZRTLT* points, size_t count,
                      uint64_t timestamp_ns);

    /**
     * @brief Process a complete measurement group (synchronized LiDAR + IMU)
     */
    void processMeasurement(MeasureGroup& meas);

    /**
     * @brief Process all available buffered sensor data
     *
     * Call this periodically after adding IMU and point cloud data.
     * It will sync and process all complete measurement groups.
     *
     * @return Number of measurements processed
     */
    int process();

    //=========================================================================
    // State Output
    //=========================================================================

    /**
     * @brief Get current state estimate
     */
    SlamState getState() const;

    /**
     * @brief Get current pose as 4x4 matrix
     */
    M4D getPose() const;

    /**
     * @brief Get current position
     */
    V3D getPosition() const;

    /**
     * @brief Get current rotation matrix
     */
    M3D getRotation() const;

    /**
     * @brief Check if SLAM is initialized
     */
    bool isInitialized() const { return ekf_initialized_; }

    //=========================================================================
    // Map Access
    //=========================================================================

    /**
     * @brief Get all map points (with intensity)
     */
    std::vector<WorldPoint> getMapPoints() const;

    /**
     * @brief Get current scan in world frame (with intensity)
     */
    std::vector<WorldPoint> getCurrentScan() const;

    /**
     * @brief Get trajectory (all poses)
     */
    std::vector<M4D> getTrajectory() const;

    /**
     * @brief Get number of points in map
     */
    size_t getMapSize() const;

    //=========================================================================
    // Map Save/Load
    //=========================================================================

    /**
     * @brief Save map to PLY file (with intensity)
     */
    bool saveMap(const std::string& filename) const;

    /**
     * @brief Save trajectory to PLY file
     */
    bool saveTrajectory(const std::string& filename) const;

    //=========================================================================
    // Localization Mode
    //=========================================================================

    /**
     * @brief Load pre-built map for localization
     * @param filename Path to PLY or PCD file
     * @return true if map loaded successfully
     *
     * After loading, call setLocalizationMode(true) to enable localization.
     */
    bool loadMap(const std::string& filename);

    /**
     * @brief Enable/disable localization mode
     * @param enabled If true, disables map updates (localization only)
     *
     * In localization mode:
     * - Pre-built map is used for point-to-plane matching
     * - No new points are added to the map
     * - EKF still estimates pose based on map matches
     */
    void setLocalizationMode(bool enabled);

    /**
     * @brief Check if in localization mode
     */
    bool isLocalizationMode() const { return localization_mode_; }

    /**
     * @brief Set initial pose estimate (for localization mode)
     * @param pose 4x4 transformation matrix (world to body)
     *
     * Use this when starting localization with a rough initial guess.
     */
    void setInitialPose(const M4D& pose);

    /**
     * @brief Perform global re-localization using ICP
     * @param initial_guess Optional initial pose guess (uses current if not provided)
     * @return true if localization succeeded
     *
     * Runs multi-scale ICP against the pre-built map.
     * Updates the EKF state if successful.
     */
    bool globalRelocalize(const M4D& initial_guess = M4D::Identity());

    /**
     * @brief Get fitness score from last localization
     * @return Inlier ratio (0-1), higher is better
     */
    double getLocalizationFitness() const { return localization_fitness_; }

    //=========================================================================
    // Callbacks
    //=========================================================================

    /**
     * @brief Set callback for state updates
     */
    void setStateCallback(StateCallback cb) { state_callback_ = cb; }

    /**
     * @brief Set callback for processed point clouds
     */
    void setPointCloudCallback(PointCloudCallback cb) { cloud_callback_ = cb; }

private:
    //=========================================================================
    // Internal Processing
    //=========================================================================

    /**
     * @brief Try to sync sensor data into a measurement group
     */
    bool syncPackages(MeasureGroup& meas);

    /**
     * @brief Process one synchronized measurement
     */
    void processOneMeasurement(MeasureGroup& meas);

    /**
     * @brief Transform point from body to world frame (preserves intensity)
     */
    void pointBodyToWorld(const LidarPoint& pi, WorldPoint& po) const;

    /**
     * @brief Manage local map FOV
     */
    void lasermapFovSegment();

    /**
     * @brief Add points to map incrementally
     */
    void mapIncremental();

    /**
     * @brief EKF measurement model
     */
    void hShareModel(state_ikfom& s, esekfom::dyn_share_datastruct<double>& ekfom_data);

    //=========================================================================
    // Configuration
    //=========================================================================
    SlamConfig config_;
    bool initialized_ = false;

    //=========================================================================
    // Sensor Buffers
    //=========================================================================
    mutable std::mutex buffer_mutex_;
    std::deque<ImuData> imu_buffer_;
    std::deque<PointCloudPtr> lidar_buffer_;
    std::deque<double> time_buffer_;

    double last_timestamp_lidar_ = 0;
    double last_timestamp_imu_ = -1.0;
    bool lidar_pushed_ = false;
    double lidar_end_time_ = 0;
    double lidar_mean_scantime_ = 0.1;
    int scan_num_ = 0;

    //=========================================================================
    // Processing Components
    //=========================================================================
    std::shared_ptr<Preprocessor> preprocessor_;
    std::shared_ptr<ImuProcessor> imu_processor_;

    //=========================================================================
    // EKF State
    //=========================================================================
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;
    state_ikfom state_point_;
    bool ekf_initialized_ = false;
    double first_lidar_time_ = 0;
    bool first_scan_ = true;

    //=========================================================================
    // Map
    //=========================================================================
    mutable KD_TREE<PointType> ikdtree_;
    bool localmap_initialized_ = false;
    BoxPointType local_map_points_;

    // Nearest points for each feature
    std::vector<PointVector> nearest_points_;
    bool point_selected_surf_[100000];
    float res_last_[100000];

    //=========================================================================
    // Point Cloud Buffers (PCL-compatible for ikd-tree)
    //=========================================================================
    pcl::PointCloud<PointType>::Ptr feats_undistort_;
    pcl::PointCloud<PointType>::Ptr feats_down_body_;
    pcl::PointCloud<PointType>::Ptr feats_down_world_;
    pcl::PointCloud<PointType>::Ptr normvec_;
    pcl::PointCloud<PointType>::Ptr laser_cloud_ori_;
    pcl::PointCloud<PointType>::Ptr corr_normvect_;

    pcl::VoxelGrid<PointType> down_size_filter_surf_;
    pcl::VoxelGrid<PointType> down_size_filter_map_;

    int feats_down_size_ = 0;
    int effct_feat_num_ = 0;

    //=========================================================================
    // Trajectory
    //=========================================================================
    mutable std::mutex trajectory_mutex_;
    std::vector<M4D> trajectory_;

    //=========================================================================
    // Callbacks
    //=========================================================================
    StateCallback state_callback_;
    PointCloudCallback cloud_callback_;

    //=========================================================================
    // Localization Mode
    //=========================================================================
    bool localization_mode_ = false;
    double localization_fitness_ = 0.0;
    ICP icp_;
    MultiScaleICP multi_scale_icp_;
};

//=============================================================================
// Implementation
//=============================================================================

inline SlamEngine::SlamEngine() {
    // Initialize point cloud pointers
    feats_undistort_.reset(new pcl::PointCloud<PointType>());
    feats_down_body_.reset(new pcl::PointCloud<PointType>());
    feats_down_world_.reset(new pcl::PointCloud<PointType>());
    normvec_.reset(new pcl::PointCloud<PointType>(100000, 1));
    laser_cloud_ori_.reset(new pcl::PointCloud<PointType>(100000, 1));
    corr_normvect_.reset(new pcl::PointCloud<PointType>(100000, 1));

    // Initialize selection arrays
    std::fill(point_selected_surf_, point_selected_surf_ + 100000, true);
    std::fill(res_last_, res_last_ + 100000, -1000.0f);
}

inline SlamEngine::~SlamEngine() {
    // Save map on exit if configured
    if (config_.save_map && !trajectory_.empty()) {
        saveMap(config_.map_save_path);
    }
}

inline bool SlamEngine::init(const SlamConfig& config) {
    config_ = config;

    // Initialize preprocessor
    PreprocessConfig pre_config;
    pre_config.lidar_type = LidarType::LIVOX_MID360;
    pre_config.n_scans = 4;
    pre_config.blind_distance = 0.5;
    pre_config.point_filter_num = 1;
    preprocessor_ = std::make_shared<Preprocessor>(pre_config);

    // Initialize IMU processor
    imu_processor_ = std::make_shared<ImuProcessor>();
    imu_processor_->setDeskewEnabled(config_.deskew_enabled);
    imu_processor_->setExtrinsic(config_.extrinsic_T, config_.extrinsic_R);
    imu_processor_->setGyrCov(V3D(config_.gyr_cov, config_.gyr_cov, config_.gyr_cov));
    imu_processor_->setAccCov(V3D(config_.acc_cov, config_.acc_cov, config_.acc_cov));
    imu_processor_->setGyrBiasCov(V3D(config_.b_gyr_cov, config_.b_gyr_cov, config_.b_gyr_cov));
    imu_processor_->setAccBiasCov(V3D(config_.b_acc_cov, config_.b_acc_cov, config_.b_acc_cov));

    // Initialize downsampling filters
    down_size_filter_surf_.setLeafSize(config_.filter_size_surf,
                                        config_.filter_size_surf,
                                        config_.filter_size_surf);
    down_size_filter_map_.setLeafSize(config_.filter_size_map,
                                       config_.filter_size_map,
                                       config_.filter_size_map);

    // Initialize EKF
    double epsi[23];
    std::fill(epsi, epsi + 23, 0.001);
    s_instance_ = this;  // Set static instance pointer for callback
    kf_.init_dyn_share(get_f, df_dx, df_dw, staticHShareModel,
                       config_.max_iterations, epsi);

    // Set ikd-tree downsampling
    ikdtree_.set_downsample_param(config_.filter_size_map);

    initialized_ = true;
    std::cout << "[SlamEngine] Initialized successfully" << std::endl;
    std::cout << "  Deskew: " << (config_.deskew_enabled ? "enabled" : "disabled") << std::endl;
    std::cout << "  Extrinsic estimation: " << (config_.extrinsic_est_en ? "enabled" : "disabled") << std::endl;

    return true;
}

inline void SlamEngine::reset() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    imu_buffer_.clear();
    lidar_buffer_.clear();
    time_buffer_.clear();
    trajectory_.clear();

    imu_processor_->reset();

    // Reset ikd-tree (rebuild from scratch)
    ikdtree_.~KD_TREE();
    new (&ikdtree_) KD_TREE<PointType>();
    ikdtree_.set_downsample_param(config_.filter_size_map);

    ekf_initialized_ = false;
    first_scan_ = true;
    localmap_initialized_ = false;
    last_timestamp_lidar_ = 0;
    last_timestamp_imu_ = -1.0;
}

inline void SlamEngine::addImuData(const ImuData& imu) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    double timestamp = imu.timestamp_sec();

    if (timestamp < last_timestamp_imu_) {
        std::cerr << "[SlamEngine] IMU loop back, clearing buffer" << std::endl;
        imu_buffer_.clear();
    }

    last_timestamp_imu_ = timestamp;
    imu_buffer_.push_back(imu);
}

inline void SlamEngine::addPointCloud(const PointCloud& cloud) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    double timestamp = cloud.start_time_sec();

    if (timestamp < last_timestamp_lidar_) {
        std::cerr << "[SlamEngine] LiDAR loop back, clearing buffer" << std::endl;
        lidar_buffer_.clear();
    }

    last_timestamp_lidar_ = timestamp;

    // Convert to PCL format (preserving intensity)
    auto pcl_cloud = std::make_shared<PointCloud>(cloud);
    lidar_buffer_.push_back(pcl_cloud);
    time_buffer_.push_back(timestamp);
}

inline void SlamEngine::addRawPoints(const LivoxPointXYZRTLT* points, size_t count,
                                      uint64_t timestamp_ns) {
    PointCloud cloud;
    preprocessor_->process(points, count, timestamp_ns, cloud);
    addPointCloud(cloud);
}

inline bool SlamEngine::syncPackages(MeasureGroup& meas) {
    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    if (!lidar_pushed_) {
        auto& front_cloud = lidar_buffer_.front();
        meas.lidar_beg_time = time_buffer_.front();

        // Convert our PointCloud to internal format
        meas.lidar->clear();
        meas.lidar->points.reserve(front_cloud->size());
        for (const auto& pt : front_cloud->points) {
            meas.lidar->push_back(pt);
        }

        if (meas.lidar->size() <= 1) {
            lidar_end_time_ = meas.lidar_beg_time + lidar_mean_scantime_;
        } else {
            // Use time offset from last point
            float last_time_offset = meas.lidar->points.back().time_offset_ms / 1000.0;
            if (last_time_offset < 0.5 * lidar_mean_scantime_) {
                lidar_end_time_ = meas.lidar_beg_time + lidar_mean_scantime_;
            } else {
                scan_num_++;
                lidar_end_time_ = meas.lidar_beg_time + last_time_offset;
                lidar_mean_scantime_ += (last_time_offset - lidar_mean_scantime_) / scan_num_;
            }
        }

        meas.lidar_end_time = lidar_end_time_;
        lidar_pushed_ = true;
    }

    if (last_timestamp_imu_ < lidar_end_time_) {
        return false;
    }

    // Push IMU data
    meas.imu.clear();
    while (!imu_buffer_.empty()) {
        double imu_time = imu_buffer_.front().timestamp_sec();
        if (imu_time > lidar_end_time_) break;
        meas.imu.push_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_ = false;

    return true;
}

inline int SlamEngine::process() {
    int count = 0;
    MeasureGroup meas;
    meas.lidar = std::make_shared<PointCloud>();

    while (syncPackages(meas)) {
        processMeasurement(meas);
        count++;

        // Reset for next iteration
        meas.lidar = std::make_shared<PointCloud>();
        meas.imu.clear();
    }

    return count;
}

inline void SlamEngine::processMeasurement(MeasureGroup& meas) {
    if (meas.imu.empty() || meas.lidar->empty()) return;

    if (first_scan_) {
        first_lidar_time_ = meas.lidar_beg_time;
        imu_processor_->first_lidar_time = first_lidar_time_;
        first_scan_ = false;
        return;
    }

    // Convert our PointCloud to PCL format for IMU processing
    // This is needed because the EKF uses PCL types internally
    auto pcl_undistort = std::make_shared<PointCloud>();

    // Process IMU + undistort points
    imu_processor_->process(meas, kf_, pcl_undistort);

    // Convert back to PCL for rest of processing
    feats_undistort_->clear();
    feats_undistort_->reserve(pcl_undistort->size());
    for (const auto& pt : pcl_undistort->points) {
        PointType pcl_pt;
        pcl_pt.x = pt.x;
        pcl_pt.y = pt.y;
        pcl_pt.z = pt.z;
        pcl_pt.intensity = pt.intensity;  // PRESERVE INTENSITY
        pcl_pt.curvature = pt.time_offset_ms;
        feats_undistort_->push_back(pcl_pt);
    }

    state_point_ = kf_.get_x();

    if (feats_undistort_->empty()) {
        std::cerr << "[SlamEngine] No points after undistortion, skipping" << std::endl;
        return;
    }

    // Check if EKF is initialized (after INIT_TIME seconds)
    ekf_initialized_ = (meas.lidar_beg_time - first_lidar_time_) > 0.1;

    // Manage local map FOV
    lasermapFovSegment();

    // Downsample
    down_size_filter_surf_.setInputCloud(feats_undistort_);
    down_size_filter_surf_.filter(*feats_down_body_);
    feats_down_size_ = feats_down_body_->points.size();

    // Initialize map if needed
    if (ikdtree_.Root_Node == nullptr) {
        if (feats_down_size_ > 5) {
            feats_down_world_->resize(feats_down_size_);
            for (int i = 0; i < feats_down_size_; i++) {
                WorldPoint wp;
                LidarPoint lp;
                lp.x = feats_down_body_->points[i].x;
                lp.y = feats_down_body_->points[i].y;
                lp.z = feats_down_body_->points[i].z;
                lp.intensity = feats_down_body_->points[i].intensity;
                pointBodyToWorld(lp, wp);

                feats_down_world_->points[i].x = wp.x;
                feats_down_world_->points[i].y = wp.y;
                feats_down_world_->points[i].z = wp.z;
                feats_down_world_->points[i].intensity = wp.intensity;
            }
            ikdtree_.Build(feats_down_world_->points);
        }
        return;
    }

    if (feats_down_size_ < 5) {
        std::cerr << "[SlamEngine] Too few points, skipping" << std::endl;
        return;
    }

    // Resize buffers
    normvec_->resize(feats_down_size_);
    feats_down_world_->resize(feats_down_size_);
    nearest_points_.resize(feats_down_size_);

    // EKF update
    double solve_H_time = 0;
    kf_.update_iterated_dyn_share_modified(0.001, solve_H_time);

    state_point_ = kf_.get_x();

    // Add points to map (skip in localization mode)
    if (!localization_mode_) {
        mapIncremental();
    }

    // Update trajectory
    {
        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        M4D pose = M4D::Identity();
        pose.block<3, 3>(0, 0) = state_point_.rot.toRotationMatrix();
        pose.block<3, 1>(0, 3) = V3D(state_point_.pos[0], state_point_.pos[1], state_point_.pos[2]);
        trajectory_.push_back(pose);
    }

    // Invoke callbacks
    if (state_callback_) {
        state_callback_(getState());
    }

    if (cloud_callback_) {
        PointCloud current_scan;
        for (const auto& pt : feats_undistort_->points) {
            LidarPoint lp;
            lp.x = pt.x;
            lp.y = pt.y;
            lp.z = pt.z;
            lp.intensity = pt.intensity;
            current_scan.push_back(lp);
        }
        cloud_callback_(current_scan, getPose());
    }
}

inline void SlamEngine::pointBodyToWorld(const LidarPoint& pi, WorldPoint& po) const {
    V3D p_body(pi.x, pi.y, pi.z);
    M3D R_L_I = state_point_.offset_R_L_I.toRotationMatrix();
    V3D T_L_I(state_point_.offset_T_L_I[0], state_point_.offset_T_L_I[1], state_point_.offset_T_L_I[2]);
    V3D p_global = state_point_.rot.toRotationMatrix() * (R_L_I * p_body + T_L_I) +
                   V3D(state_point_.pos[0], state_point_.pos[1], state_point_.pos[2]);

    po.x = static_cast<float>(p_global(0));
    po.y = static_cast<float>(p_global(1));
    po.z = static_cast<float>(p_global(2));
    po.intensity = pi.intensity;  // PRESERVE INTENSITY
}

inline void SlamEngine::lasermapFovSegment() {
    V3D pos_LiD(state_point_.pos[0], state_point_.pos[1], state_point_.pos[2]);

    if (!localmap_initialized_) {
        for (int i = 0; i < 3; i++) {
            local_map_points_.vertex_min[i] = pos_LiD(i) - config_.cube_len / 2.0;
            local_map_points_.vertex_max[i] = pos_LiD(i) + config_.cube_len / 2.0;
        }
        localmap_initialized_ = true;
        return;
    }

    float mov_threshold = 1.5f;
    float dist_to_map_edge[3][2];
    bool need_move = false;

    for (int i = 0; i < 3; i++) {
        dist_to_map_edge[i][0] = std::fabs(pos_LiD(i) - local_map_points_.vertex_min[i]);
        dist_to_map_edge[i][1] = std::fabs(pos_LiD(i) - local_map_points_.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= mov_threshold * config_.det_range ||
            dist_to_map_edge[i][1] <= mov_threshold * config_.det_range) {
            need_move = true;
        }
    }

    if (!need_move) return;

    std::vector<BoxPointType> cub_needrm;
    BoxPointType new_local_map = local_map_points_;
    float mov_dist = std::max((config_.cube_len - 2.0 * mov_threshold * config_.det_range) * 0.5 * 0.9,
                              config_.det_range * (mov_threshold - 1));

    for (int i = 0; i < 3; i++) {
        BoxPointType tmp_box = local_map_points_;
        if (dist_to_map_edge[i][0] <= mov_threshold * config_.det_range) {
            new_local_map.vertex_max[i] -= mov_dist;
            new_local_map.vertex_min[i] -= mov_dist;
            tmp_box.vertex_min[i] = local_map_points_.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_box);
        } else if (dist_to_map_edge[i][1] <= mov_threshold * config_.det_range) {
            new_local_map.vertex_max[i] += mov_dist;
            new_local_map.vertex_min[i] += mov_dist;
            tmp_box.vertex_max[i] = local_map_points_.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_box);
        }
    }

    local_map_points_ = new_local_map;

    if (!cub_needrm.empty()) {
        ikdtree_.Delete_Point_Boxes(cub_needrm);
    }
}

inline void SlamEngine::mapIncremental() {
    PointVector point_to_add;
    PointVector point_no_need_downsample;
    point_to_add.reserve(feats_down_size_);
    point_no_need_downsample.reserve(feats_down_size_);

    for (int i = 0; i < feats_down_size_; i++) {
        // Transform to world frame (preserving intensity)
        PointType& point_body = feats_down_body_->points[i];
        PointType& point_world = feats_down_world_->points[i];

        V3D p_body(point_body.x, point_body.y, point_body.z);
        M3D R_L_I = state_point_.offset_R_L_I.toRotationMatrix();
        V3D T_L_I(state_point_.offset_T_L_I[0], state_point_.offset_T_L_I[1], state_point_.offset_T_L_I[2]);
        V3D p_global = state_point_.rot.toRotationMatrix() * (R_L_I * p_body + T_L_I) +
                       V3D(state_point_.pos[0], state_point_.pos[1], state_point_.pos[2]);

        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;  // PRESERVE INTENSITY

        // Decide if need to add to map
        if (!nearest_points_[i].empty() && ekf_initialized_) {
            const PointVector& points_near = nearest_points_[i];
            bool need_add = true;

            PointType mid_point;
            mid_point.x = std::floor(point_world.x / config_.filter_size_map) * config_.filter_size_map +
                          0.5 * config_.filter_size_map;
            mid_point.y = std::floor(point_world.y / config_.filter_size_map) * config_.filter_size_map +
                          0.5 * config_.filter_size_map;
            mid_point.z = std::floor(point_world.z / config_.filter_size_map) * config_.filter_size_map +
                          0.5 * config_.filter_size_map;

            float dx = point_world.x - mid_point.x;
            float dy = point_world.y - mid_point.y;
            float dz = point_world.z - mid_point.z;
            float dist = dx * dx + dy * dy + dz * dz;

            if (std::fabs(points_near[0].x - mid_point.x) > 0.5 * config_.filter_size_map &&
                std::fabs(points_near[0].y - mid_point.y) > 0.5 * config_.filter_size_map &&
                std::fabs(points_near[0].z - mid_point.z) > 0.5 * config_.filter_size_map) {
                point_no_need_downsample.push_back(point_world);
                continue;
            }

            for (size_t j = 0; j < std::min(points_near.size(), size_t(NUM_MATCH_POINTS)); j++) {
                float ndx = points_near[j].x - mid_point.x;
                float ndy = points_near[j].y - mid_point.y;
                float ndz = points_near[j].z - mid_point.z;
                if (ndx * ndx + ndy * ndy + ndz * ndz < dist) {
                    need_add = false;
                    break;
                }
            }

            if (need_add) point_to_add.push_back(point_world);
        } else {
            point_to_add.push_back(point_world);
        }
    }

    ikdtree_.Add_Points(point_to_add, true);
    ikdtree_.Add_Points(point_no_need_downsample, false);
}

inline void SlamEngine::hShareModel(state_ikfom& s,
                                     esekfom::dyn_share_datastruct<double>& ekfom_data) {
    laser_cloud_ori_->clear();
    corr_normvect_->clear();
    double total_residual = 0.0;

    // Find correspondences and compute residuals
    #ifdef MP_EN
    omp_set_num_threads(4);
    #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size_; i++) {
        PointType& point_body = feats_down_body_->points[i];
        PointType& point_world = feats_down_world_->points[i];

        // Transform to world frame
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global = s.rot.toRotationMatrix() *
                       (s.offset_R_L_I.toRotationMatrix() * p_body +
                        V3D(s.offset_T_L_I[0], s.offset_T_L_I[1], s.offset_T_L_I[2])) +
                       V3D(s.pos[0], s.pos[1], s.pos[2]);

        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        std::vector<float> point_search_sq_dis(NUM_MATCH_POINTS);
        auto& points_near = nearest_points_[i];

        if (ekfom_data.converge) {
            ikdtree_.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, point_search_sq_dis);
            point_selected_surf_[i] = points_near.size() >= NUM_MATCH_POINTS &&
                                      point_search_sq_dis[NUM_MATCH_POINTS - 1] <= 5.0;
        }

        if (!point_selected_surf_[i]) continue;

        // Estimate plane
        Eigen::Matrix<float, 4, 1> pabcd;
        point_selected_surf_[i] = false;

        Eigen::Matrix<float, NUM_MATCH_POINTS, 3> A;
        Eigen::Matrix<float, NUM_MATCH_POINTS, 1> b;
        A.setZero();
        b.setOnes();
        b *= -1.0f;

        for (int j = 0; j < NUM_MATCH_POINTS; j++) {
            A(j, 0) = points_near[j].x;
            A(j, 1) = points_near[j].y;
            A(j, 2) = points_near[j].z;
        }

        Eigen::Matrix<float, 3, 1> normvec = A.colPivHouseholderQr().solve(b);
        float n = normvec.norm();
        pabcd(0) = normvec(0) / n;
        pabcd(1) = normvec(1) / n;
        pabcd(2) = normvec(2) / n;
        pabcd(3) = 1.0f / n;

        bool valid = true;
        for (int j = 0; j < NUM_MATCH_POINTS; j++) {
            if (std::fabs(pabcd(0) * points_near[j].x + pabcd(1) * points_near[j].y +
                          pabcd(2) * points_near[j].z + pabcd(3)) > 0.1f) {
                valid = false;
                break;
            }
        }

        if (valid) {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y +
                        pabcd(2) * point_world.z + pabcd(3);
            float score = 1.0f - 0.9f * std::fabs(pd2) / std::sqrt(p_body.norm());

            if (score > 0.9f) {
                point_selected_surf_[i] = true;
                normvec_->points[i].x = pabcd(0);
                normvec_->points[i].y = pabcd(1);
                normvec_->points[i].z = pabcd(2);
                normvec_->points[i].intensity = pd2;
                res_last_[i] = std::abs(pd2);
            }
        }
    }

    // Collect valid correspondences
    effct_feat_num_ = 0;
    for (int i = 0; i < feats_down_size_; i++) {
        if (point_selected_surf_[i]) {
            laser_cloud_ori_->points[effct_feat_num_] = feats_down_body_->points[i];
            corr_normvect_->points[effct_feat_num_] = normvec_->points[i];
            total_residual += res_last_[i];
            effct_feat_num_++;
        }
    }

    if (effct_feat_num_ < 1) {
        ekfom_data.valid = false;
        std::cerr << "[SlamEngine] No effective points!" << std::endl;
        return;
    }

    // Build measurement Jacobian
    ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num_, 12);
    ekfom_data.h.resize(effct_feat_num_);

    for (int i = 0; i < effct_feat_num_; i++) {
        const PointType& laser_p = laser_cloud_ori_->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat = skew_sym_mat(point_this_be);

        V3D point_this = s.offset_R_L_I.toRotationMatrix() * point_this_be +
                         V3D(s.offset_T_L_I[0], s.offset_T_L_I[1], s.offset_T_L_I[2]);
        M3D point_crossmat = skew_sym_mat(point_this);

        const PointType& norm_p = corr_normvect_->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        V3D C = s.rot.toRotationMatrix().transpose() * norm_vec;
        V3D A = point_crossmat * C;

        if (config_.extrinsic_est_en) {
            V3D B = point_be_crossmat * s.offset_R_L_I.toRotationMatrix().transpose() * C;
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
                A(0), A(1), A(2), B(0), B(1), B(2), C(0), C(1), C(2);
        } else {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
                A(0), A(1), A(2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        ekfom_data.h(i) = -norm_p.intensity;
    }
}

inline SlamState SlamEngine::getState() const {
    SlamState state;
    state.timestamp_ns = static_cast<uint64_t>(lidar_end_time_ * 1e9);
    state.rot = state_point_.rot.toRotationMatrix();
    state.pos = V3D(state_point_.pos[0], state_point_.pos[1], state_point_.pos[2]);
    state.vel = V3D(state_point_.vel[0], state_point_.vel[1], state_point_.vel[2]);
    state.bias_gyro = V3D(state_point_.bg[0], state_point_.bg[1], state_point_.bg[2]);
    state.bias_acc = V3D(state_point_.ba[0], state_point_.ba[1], state_point_.ba[2]);
    state.gravity = V3D(state_point_.grav[0], state_point_.grav[1], state_point_.grav[2]);
    return state;
}

inline M4D SlamEngine::getPose() const {
    M4D pose = M4D::Identity();
    pose.block<3, 3>(0, 0) = state_point_.rot.toRotationMatrix();
    pose.block<3, 1>(0, 3) = V3D(state_point_.pos[0], state_point_.pos[1], state_point_.pos[2]);
    return pose;
}

inline V3D SlamEngine::getPosition() const {
    return V3D(state_point_.pos[0], state_point_.pos[1], state_point_.pos[2]);
}

inline M3D SlamEngine::getRotation() const {
    return state_point_.rot.toRotationMatrix();
}

inline std::vector<WorldPoint> SlamEngine::getMapPoints() const {
    std::vector<WorldPoint> points;

    if (ikdtree_.Root_Node == nullptr) return points;

    PointVector storage;
    ikdtree_.flatten(ikdtree_.Root_Node, storage, NOT_RECORD);

    points.reserve(storage.size());
    for (const auto& pt : storage) {
        points.emplace_back(pt.x, pt.y, pt.z, pt.intensity);
    }

    return points;
}

inline std::vector<WorldPoint> SlamEngine::getCurrentScan() const {
    std::vector<WorldPoint> points;

    if (!feats_undistort_ || feats_undistort_->empty()) return points;

    points.reserve(feats_undistort_->size());
    for (const auto& pt : feats_undistort_->points) {
        LidarPoint lp;
        lp.x = pt.x;
        lp.y = pt.y;
        lp.z = pt.z;
        lp.intensity = pt.intensity;

        WorldPoint wp;
        pointBodyToWorld(lp, wp);
        points.push_back(wp);
    }

    return points;
}

inline std::vector<M4D> SlamEngine::getTrajectory() const {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    return trajectory_;
}

inline size_t SlamEngine::getMapSize() const {
    return ikdtree_.validnum();
}

inline bool SlamEngine::saveMap(const std::string& filename) const {
    auto points = getMapPoints();
    if (points.empty()) {
        std::cerr << "[SlamEngine] No map points to save" << std::endl;
        return false;
    }

    PlyExportOptions opts;
    opts.format = PlyFormat::BINARY_LITTLE_ENDIAN;
    opts.include_intensity = true;

    return exportToPly(points, filename, opts);
}

inline bool SlamEngine::saveTrajectory(const std::string& filename) const {
    auto poses = getTrajectory();
    if (poses.empty()) {
        std::cerr << "[SlamEngine] No trajectory to save" << std::endl;
        return false;
    }

    return exportTrajectoryToPly(poses, filename);
}

//=============================================================================
// Localization Mode Implementation
//=============================================================================

inline bool SlamEngine::loadMap(const std::string& filename) {
    std::cout << "[SlamEngine] Loading map from: " << filename << std::endl;

    pcl::PointCloud<PointType>::Ptr map_cloud(new pcl::PointCloud<PointType>());

    // Determine file type and load
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    int result = -1;
    if (ext == "pcd") {
        result = pcl::io::loadPCDFile<PointType>(filename, *map_cloud);
    } else if (ext == "ply") {
        result = pcl::io::loadPLYFile<PointType>(filename, *map_cloud);
    } else {
        std::cerr << "[SlamEngine] Unsupported map format: " << ext << std::endl;
        return false;
    }

    if (result < 0 || map_cloud->empty()) {
        std::cerr << "[SlamEngine] Failed to load map or map is empty" << std::endl;
        return false;
    }

    std::cout << "[SlamEngine] Loaded " << map_cloud->size() << " points" << std::endl;

    // Build ikd-tree from loaded map
    ikdtree_.~KD_TREE();
    new (&ikdtree_) KD_TREE<PointType>();
    ikdtree_.set_downsample_param(config_.filter_size_map);
    ikdtree_.Build(map_cloud->points);

    localmap_initialized_ = true;

    std::cout << "[SlamEngine] Map loaded into ikd-tree, size: " << ikdtree_.validnum() << std::endl;

    return true;
}

inline void SlamEngine::setLocalizationMode(bool enabled) {
    localization_mode_ = enabled;

    if (enabled) {
        std::cout << "[SlamEngine] Localization mode ENABLED" << std::endl;
        std::cout << "  - Map updates disabled" << std::endl;
        std::cout << "  - Using pre-built map for matching" << std::endl;

        if (ikdtree_.Root_Node == nullptr) {
            std::cerr << "[SlamEngine] WARNING: No map loaded! Call loadMap() first." << std::endl;
        }
    } else {
        std::cout << "[SlamEngine] Localization mode DISABLED (SLAM mode)" << std::endl;
    }
}

inline void SlamEngine::setInitialPose(const M4D& pose) {
    std::cout << "[SlamEngine] Setting initial pose" << std::endl;

    // Extract rotation and translation from 4x4 matrix
    M3D R = pose.block<3, 3>(0, 0);
    V3D t = pose.block<3, 1>(0, 3);

    // Update EKF state
    state_point_.rot = MTK::SO3<double>(Eigen::Quaterniond(R));
    state_point_.pos = t;

    // Reset velocities and biases to zero
    state_point_.vel = V3D::Zero();
    state_point_.bg = V3D::Zero();
    state_point_.ba = V3D::Zero();

    // Update the EKF
    kf_.change_x(state_point_);

    // Mark as initialized
    ekf_initialized_ = true;

    std::cout << "  Position: [" << t.x() << ", " << t.y() << ", " << t.z() << "]" << std::endl;

    // Extract euler angles for display
    V3D euler = so3_math::rotationToEuler(R);
    std::cout << "  Orientation (RPY deg): ["
              << euler.x() * 180.0 / M_PI << ", "
              << euler.y() * 180.0 / M_PI << ", "
              << euler.z() * 180.0 / M_PI << "]" << std::endl;
}

inline bool SlamEngine::globalRelocalize(const M4D& initial_guess) {
    if (ikdtree_.Root_Node == nullptr) {
        std::cerr << "[SlamEngine] Cannot relocalize: no map loaded" << std::endl;
        return false;
    }

    if (!feats_down_body_ || feats_down_body_->empty()) {
        std::cerr << "[SlamEngine] Cannot relocalize: no scan available" << std::endl;
        return false;
    }

    std::cout << "[SlamEngine] Running global re-localization..." << std::endl;

    // Convert current scan to vector of V3D
    std::vector<V3D> scan_points;
    scan_points.reserve(feats_down_body_->size());
    for (const auto& pt : feats_down_body_->points) {
        scan_points.emplace_back(pt.x, pt.y, pt.z);
    }

    // Use current pose if no initial guess provided
    M4D guess = initial_guess;
    if (guess.isIdentity()) {
        guess = getPose();
    }

    // Run multi-scale ICP
    ICPResult result = multi_scale_icp_.align(scan_points, ikdtree_, guess, {2.0, 1.0, 0.5});

    localization_fitness_ = result.fitness_score;

    std::cout << "  Iterations: " << result.num_iterations << std::endl;
    std::cout << "  Inliers: " << result.num_inliers << "/" << scan_points.size() << std::endl;
    std::cout << "  Fitness: " << result.fitness_score << std::endl;
    std::cout << "  RMSE: " << result.rmse << std::endl;
    std::cout << "  Converged: " << (result.converged ? "yes" : "no") << std::endl;

    if (result.converged && result.fitness_score > 0.3) {
        // Update pose with ICP result
        setInitialPose(result.transformation);
        std::cout << "[SlamEngine] Re-localization SUCCESSFUL" << std::endl;
        return true;
    } else {
        std::cerr << "[SlamEngine] Re-localization FAILED (poor fitness)" << std::endl;
        return false;
    }
}

// Static member definition
inline SlamEngine* SlamEngine::s_instance_ = nullptr;

} // namespace slam

#endif // SLAM_ENGINE_HPP
