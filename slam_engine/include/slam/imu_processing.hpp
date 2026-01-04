/**
 * @file imu_processing.hpp
 * @brief IMU processing and motion compensation for LiDAR-inertial SLAM
 *
 * This is a ROS-free port of FAST-LIO's IMU processing with added:
 * - Gravity vector alignment at startup (fixes tilted map issue)
 * - Clean C++ interface using custom types
 *
 * Key modification: When the IMU starts in a non-level position, we use
 * the measured gravity vector to compute a correction rotation, ensuring
 * the map Z-axis aligns with true vertical regardless of initial orientation.
 */

#ifndef SLAM_IMU_PROCESSING_HPP
#define SLAM_IMU_PROCESSING_HPP

#include <cmath>
#include <deque>
#include <mutex>
#include <vector>
#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "slam/types.hpp"
#include "slam/so3_math.hpp"

// Include the IKFOM toolkit and state definitions
#include "slam/use_ikfom.hpp"

namespace slam {

//=============================================================================
// Constants
//=============================================================================

constexpr int MAX_INI_COUNT = 20;  // Increased from 10 for better gravity estimation

//=============================================================================
// IMU Processor Class
//=============================================================================

/**
 * @brief IMU processing and point cloud motion compensation
 *
 * Handles:
 * - IMU initialization with gravity alignment
 * - IMU forward propagation
 * - Point cloud undistortion (deskewing)
 */
class ImuProcessor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcessor();
    ~ImuProcessor() = default;

    /**
     * @brief Reset the IMU processor state
     */
    void reset();

    /**
     * @brief Set LiDAR-to-IMU extrinsic transformation
     */
    void setExtrinsic(const V3D& translation, const M3D& rotation);
    void setExtrinsic(const M4D& T);

    /**
     * @brief Enable/disable point cloud deskewing
     */
    void setDeskewEnabled(bool enabled) { deskew_enabled_ = enabled; }

    /**
     * @brief Set IMU noise parameters
     */
    void setGyrCov(const V3D& cov) { cov_gyr_scale_ = cov; }
    void setAccCov(const V3D& cov) { cov_acc_scale_ = cov; }
    void setGyrBiasCov(const V3D& cov) { cov_bias_gyr_ = cov; }
    void setAccBiasCov(const V3D& cov) { cov_bias_acc_ = cov; }

    /**
     * @brief Process synchronized IMU and LiDAR measurements
     * @param meas Measurement group containing IMU and LiDAR data
     * @param kf_state Kalman filter state (modified in-place)
     * @param pcl_undistorted Output undistorted point cloud
     */
    void process(const MeasureGroup& meas,
                 esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state,
                 PointCloudPtr pcl_undistorted);

    /**
     * @brief Check if IMU initialization is complete
     */
    bool isInitialized() const { return !imu_need_init_; }

    /**
     * @brief Get the gravity-aligned initial rotation
     * This is the rotation computed during initialization to align
     * the measured gravity with world -Z axis
     */
    M3D getInitialRotation() const { return initial_rotation_; }

    // Process noise matrix
    Eigen::Matrix<double, 12, 12> Q;

    // Noise covariances (for external access)
    V3D cov_acc;
    V3D cov_gyr;

    double first_lidar_time = 0;

private:
    /**
     * @brief Initialize IMU state from measurements
     *
     * KEY MODIFICATION: This function now computes a gravity-aligned
     * initial rotation so that the map Z-axis is vertical regardless
     * of the initial sensor orientation.
     */
    void imuInit(const MeasureGroup& meas,
                 esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state,
                 int& N);

    /**
     * @brief Compute rotation to align measured gravity with world vertical
     * @param measured_gravity Gravity vector measured in sensor frame
     * @return Rotation matrix that aligns measured gravity with -Z
     */
    M3D computeGravityAlignment(const V3D& measured_gravity);

    /**
     * @brief Undistort point cloud using IMU poses
     */
    void undistortPointCloud(const MeasureGroup& meas,
                             esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state,
                             PointCloud& pcl_out);

    // LiDAR-IMU extrinsics
    M3D Lidar_R_wrt_IMU_;
    V3D Lidar_T_wrt_IMU_;

    // Initialization state
    V3D mean_acc_;
    V3D mean_gyr_;
    V3D angvel_last_;
    V3D acc_s_last_;

    // IMU data buffer
    ImuData last_imu_;
    std::deque<ImuData> v_imu_;
    std::vector<Pose6D> IMUpose_;

    // State flags
    double start_timestamp_ = -1;
    double last_lidar_end_time_ = 0;
    int init_iter_num_ = 1;
    bool b_first_frame_ = true;
    bool imu_need_init_ = true;
    bool deskew_enabled_ = true;

    // Noise scale parameters
    V3D cov_acc_scale_;
    V3D cov_gyr_scale_;
    V3D cov_bias_gyr_;
    V3D cov_bias_acc_;

    // Gravity-aligned initial rotation (computed during init)
    M3D initial_rotation_;

    // Debug output
    std::ofstream fout_imu_;
};

//=============================================================================
// Implementation
//=============================================================================

inline ImuProcessor::ImuProcessor()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1) {

    init_iter_num_ = 1;
    Q = process_noise_cov();

    cov_acc = V3D(0.1, 0.1, 0.1);
    cov_gyr = V3D(0.1, 0.1, 0.1);
    cov_bias_gyr_ = V3D(0.0001, 0.0001, 0.0001);
    cov_bias_acc_ = V3D(0.0001, 0.0001, 0.0001);
    cov_acc_scale_ = cov_acc;
    cov_gyr_scale_ = cov_gyr;

    mean_acc_ = V3D(0, 0, -1.0);
    mean_gyr_ = V3D(0, 0, 0);
    angvel_last_ = V3D::Zero();
    acc_s_last_ = V3D::Zero();

    Lidar_T_wrt_IMU_ = V3D::Zero();
    Lidar_R_wrt_IMU_ = M3D::Identity();

    initial_rotation_ = M3D::Identity();
}

inline void ImuProcessor::reset() {
    mean_acc_ = V3D(0, 0, -1.0);
    mean_gyr_ = V3D(0, 0, 0);
    angvel_last_ = V3D::Zero();
    imu_need_init_ = true;
    start_timestamp_ = -1;
    init_iter_num_ = 1;
    v_imu_.clear();
    IMUpose_.clear();
    last_imu_ = ImuData();
    initial_rotation_ = M3D::Identity();
}

inline void ImuProcessor::setExtrinsic(const V3D& translation, const M3D& rotation) {
    Lidar_T_wrt_IMU_ = translation;
    Lidar_R_wrt_IMU_ = rotation;
}

inline void ImuProcessor::setExtrinsic(const M4D& T) {
    Lidar_T_wrt_IMU_ = T.block<3, 1>(0, 3);
    Lidar_R_wrt_IMU_ = T.block<3, 3>(0, 0);
}

inline M3D ImuProcessor::computeGravityAlignment(const V3D& measured_gravity) {
    // Normalize the measured gravity vector
    V3D g_measured = measured_gravity.normalized();

    // World frame gravity direction (pointing down in -Z)
    V3D g_world(0, 0, -1);

    // Compute rotation that aligns measured gravity with world gravity
    // R * g_measured = g_world
    // This means: after applying R, the sensor's "down" becomes world "down"

    return rotationFromTwoVectors<double>(g_measured, g_world);
}

inline void ImuProcessor::imuInit(
    const MeasureGroup& meas,
    esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state,
    int& N) {

    // 1. Initialize gravity, gyro bias, and covariances
    // 2. Normalize acceleration measurements to unit gravity
    // 3. ADDED: Compute gravity-aligned initial rotation

    V3D cur_acc, cur_gyr;

    if (b_first_frame_) {
        reset();
        N = 1;
        b_first_frame_ = false;

        const auto& first_imu = meas.imu.front();
        mean_acc_ = first_imu.acc;
        mean_gyr_ = first_imu.gyro;
        first_lidar_time = meas.lidar_beg_time;
    }

    // Accumulate IMU measurements for averaging
    for (const auto& imu : meas.imu) {
        cur_acc = imu.acc;
        cur_gyr = imu.gyro;

        // Running mean update (Welford's algorithm)
        mean_acc_ += (cur_acc - mean_acc_) / static_cast<double>(N);
        mean_gyr_ += (cur_gyr - mean_gyr_) / static_cast<double>(N);

        // Running variance update
        cov_acc = cov_acc * (N - 1.0) / N +
                  (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_) * (N - 1.0) / (N * N);
        cov_gyr = cov_gyr * (N - 1.0) / N +
                  (cur_gyr - mean_gyr_).cwiseProduct(cur_gyr - mean_gyr_) * (N - 1.0) / (N * N);

        N++;
    }

    state_ikfom init_state = kf_state.get_x();

    // Set gravity direction (normalized, scaled to G)
    init_state.grav = S2(-mean_acc_ / mean_acc_.norm() * G_m_s2);

    //=========================================================================
    // GRAVITY ALIGNMENT FIX
    //=========================================================================
    // Compute the rotation that aligns the measured gravity (mean_acc_)
    // with the world vertical direction (-Z). This ensures that even if
    // the sensor starts tilted, the map will be built with Z-up orientation.

    initial_rotation_ = computeGravityAlignment(mean_acc_);

    // Set the initial rotation in the state
    // This is the KEY FIX - without this, rot starts as identity regardless
    // of the initial sensor orientation
    init_state.rot = SO3(initial_rotation_);

    std::cout << "[ImuProcessor] Gravity alignment rotation computed:" << std::endl;
    std::cout << "  Measured gravity (body): ["
              << mean_acc_.x() << ", " << mean_acc_.y() << ", " << mean_acc_.z()
              << "] (norm: " << mean_acc_.norm() << ")" << std::endl;
    std::cout << "  Initial rotation (RPY deg): ["
              << rad2deg(RotMtoEuler(initial_rotation_)(0)) << ", "
              << rad2deg(RotMtoEuler(initial_rotation_)(1)) << ", "
              << rad2deg(RotMtoEuler(initial_rotation_)(2)) << "]" << std::endl;

    //=========================================================================

    // Set gyro bias from mean angular velocity (assuming static start)
    init_state.bg = mean_gyr_;

    // Set LiDAR-IMU extrinsics
    init_state.offset_T_L_I = Lidar_T_wrt_IMU_;
    init_state.offset_R_L_I = SO3(Lidar_R_wrt_IMU_);

    kf_state.change_x(init_state);

    // Initialize covariance
    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
    init_P.setIdentity();
    init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
    init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
    init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
    init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
    init_P(21, 21) = init_P(22, 22) = 0.00001;
    kf_state.change_P(init_P);

    // Save last IMU for next iteration
    last_imu_ = meas.imu.back();
}

inline void ImuProcessor::undistortPointCloud(
    const MeasureGroup& meas,
    esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state,
    PointCloud& pcl_out) {

    // Add the last IMU from previous frame to current IMU buffer
    std::deque<ImuData> v_imu = meas.imu;
    v_imu.push_front(last_imu_);

    const double imu_beg_time = v_imu.front().timestamp_sec();
    const double imu_end_time = v_imu.back().timestamp_sec();
    const double pcl_beg_time = meas.lidar_beg_time;
    const double pcl_end_time = meas.lidar_end_time;

    // Copy and sort point cloud by time offset
    pcl_out = *meas.lidar;
    std::sort(pcl_out.points.begin(), pcl_out.points.end(),
              [](const LidarPoint& a, const LidarPoint& b) {
                  return a.time_offset_ms < b.time_offset_ms;
              });

    // Initialize IMU pose buffer
    state_ikfom imu_state = kf_state.get_x();
    IMUpose_.clear();
    IMUpose_.push_back(makePose6D(0.0, acc_s_last_, angvel_last_,
                                   V3D(imu_state.vel[0], imu_state.vel[1], imu_state.vel[2]),
                                   V3D(imu_state.pos[0], imu_state.pos[1], imu_state.pos[2]),
                                   imu_state.rot.toRotationMatrix()));

    // Forward propagation at each IMU sample
    V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
    M3D R_imu;
    double dt = 0;

    input_ikfom in;
    for (auto it = v_imu.begin(); it < v_imu.end() - 1; it++) {
        const auto& head = *it;
        const auto& tail = *(it + 1);

        if (tail.timestamp_sec() < last_lidar_end_time_) continue;

        // Average IMU measurements
        angvel_avr = 0.5 * (head.gyro + tail.gyro);
        acc_avr = 0.5 * (head.acc + tail.acc);

        // Scale acceleration to proper units
        acc_avr = acc_avr * G_m_s2 / mean_acc_.norm();

        // Compute time step
        if (head.timestamp_sec() < last_lidar_end_time_) {
            dt = tail.timestamp_sec() - last_lidar_end_time_;
        } else {
            dt = tail.timestamp_sec() - head.timestamp_sec();
        }

        // Propagate state
        in.acc = acc_avr;
        in.gyro = angvel_avr;
        Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
        Q.block<3, 3>(3, 3).diagonal() = cov_acc;
        Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr_;
        Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc_;
        kf_state.predict(dt, Q, in);

        // Save pose at this IMU timestamp
        imu_state = kf_state.get_x();
        angvel_last_ = angvel_avr - V3D(imu_state.bg[0], imu_state.bg[1], imu_state.bg[2]);
        acc_s_last_ = imu_state.rot.toRotationMatrix() *
                      (acc_avr - V3D(imu_state.ba[0], imu_state.ba[1], imu_state.ba[2]));
        for (int i = 0; i < 3; i++) {
            acc_s_last_[i] += imu_state.grav[i];
        }

        double offs_t = tail.timestamp_sec() - pcl_beg_time;
        IMUpose_.push_back(makePose6D(offs_t, acc_s_last_, angvel_last_,
                                       V3D(imu_state.vel[0], imu_state.vel[1], imu_state.vel[2]),
                                       V3D(imu_state.pos[0], imu_state.pos[1], imu_state.pos[2]),
                                       imu_state.rot.toRotationMatrix()));
    }

    // Propagate to scan end time
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    kf_state.predict(dt, Q, in);

    imu_state = kf_state.get_x();
    last_imu_ = meas.imu.back();
    last_lidar_end_time_ = pcl_end_time;

    // Skip undistortion if disabled
    if (!deskew_enabled_) return;

    // Backward propagation to undistort each point
    if (pcl_out.empty()) return;

    auto it_pcl = pcl_out.points.end() - 1;
    for (auto it_kp = IMUpose_.end() - 1; it_kp != IMUpose_.begin(); it_kp--) {
        auto head = it_kp - 1;
        auto tail = it_kp;

        R_imu = head->rot;
        vel_imu = head->vel;
        pos_imu = head->pos;
        acc_imu = tail->acc;
        angvel_avr = tail->gyro;

        for (; it_pcl >= pcl_out.points.begin() &&
               it_pcl->time_offset_ms > head->offset_time * 1000.0; it_pcl--) {

            dt = it_pcl->time_offset_ms / 1000.0 - head->offset_time;

            // Compute rotation at point timestamp
            M3D R_i = R_imu * Exp(angvel_avr, dt);

            // Point position in body frame
            V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);

            // Translation from point time to scan end time
            V3D T_ei = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt -
                       V3D(imu_state.pos[0], imu_state.pos[1], imu_state.pos[2]);

            // Transform point to scan-end frame
            M3D offset_R = imu_state.offset_R_L_I.toRotationMatrix();
            V3D offset_T(imu_state.offset_T_L_I[0], imu_state.offset_T_L_I[1], imu_state.offset_T_L_I[2]);

            V3D P_compensate = offset_R.transpose() *
                (imu_state.rot.toRotationMatrix().transpose() *
                 (R_i * (offset_R * P_i + offset_T) + T_ei) - offset_T);

            it_pcl->x = static_cast<float>(P_compensate(0));
            it_pcl->y = static_cast<float>(P_compensate(1));
            it_pcl->z = static_cast<float>(P_compensate(2));

            if (it_pcl == pcl_out.points.begin()) break;
        }
    }
}

inline void ImuProcessor::process(
    const MeasureGroup& meas,
    esekfom::esekf<state_ikfom, 12, input_ikfom>& kf_state,
    PointCloudPtr pcl_undistorted) {

    if (meas.imu.empty()) return;
    if (!meas.lidar || meas.lidar->empty()) return;

    if (imu_need_init_) {
        // IMU initialization phase
        imuInit(meas, kf_state, init_iter_num_);

        imu_need_init_ = true;
        last_imu_ = meas.imu.back();

        if (init_iter_num_ > MAX_INI_COUNT) {
            cov_acc *= std::pow(G_m_s2 / mean_acc_.norm(), 2);
            imu_need_init_ = false;

            cov_acc = cov_acc_scale_;
            cov_gyr = cov_gyr_scale_;

            std::cout << "[ImuProcessor] IMU initialization complete!" << std::endl;
            std::cout << "  Gravity magnitude: " << mean_acc_.norm() << " m/s^2" << std::endl;
            std::cout << "  Gyro bias: [" << mean_gyr_.transpose() << "] rad/s" << std::endl;
        }
        return;
    }

    // Normal processing - undistort point cloud
    undistortPointCloud(meas, kf_state, *pcl_undistorted);
}

} // namespace slam

#endif // SLAM_IMU_PROCESSING_HPP
