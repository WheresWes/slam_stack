/**
 * @file types.hpp
 * @brief Core data types for the SLAM stack, replacing ROS message types
 *
 * This file defines all the fundamental data structures used throughout
 * the SLAM system, designed for use without ROS dependencies.
 */

#ifndef SLAM_TYPES_HPP
#define SLAM_TYPES_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <deque>
#include <cstdint>
#include <memory>

namespace slam {

//=============================================================================
// Constants
//=============================================================================

constexpr double G_m_s2 = 9.81;           // Gravity constant (m/s^2)
constexpr double PI = 3.14159265358979323846;
constexpr int DIM_STATE = 18;             // State dimension
constexpr int DIM_PROC_N = 12;            // Process noise dimension
constexpr int NUM_MATCH_POINTS = 5;       // Points for plane fitting

//=============================================================================
// Type Aliases
//=============================================================================

using V3D = Eigen::Vector3d;
using V3F = Eigen::Vector3f;
using M3D = Eigen::Matrix3d;
using M3F = Eigen::Matrix3f;
using M4D = Eigen::Matrix4d;
using M4F = Eigen::Matrix4f;
using Quaterniond = Eigen::Quaterniond;

//=============================================================================
// IMU Data Structure
//=============================================================================

/**
 * @brief IMU measurement data
 * Replaces sensor_msgs::Imu
 */
struct ImuData {
    uint64_t timestamp_ns;      // Timestamp in nanoseconds
    V3D acc;                    // Linear acceleration (m/s^2)
    V3D gyro;                   // Angular velocity (rad/s)

    ImuData() : timestamp_ns(0), acc(V3D::Zero()), gyro(V3D::Zero()) {}

    ImuData(uint64_t ts, const V3D& a, const V3D& g)
        : timestamp_ns(ts), acc(a), gyro(g) {}

    // Convert timestamp to seconds
    double timestamp_sec() const {
        return static_cast<double>(timestamp_ns) * 1e-9;
    }
};

//=============================================================================
// LiDAR Point Structure
//=============================================================================

/**
 * @brief Single LiDAR point with intensity and timing
 * Replaces pcl::PointXYZINormal for Livox data
 */
struct LidarPoint {
    float x, y, z;              // 3D coordinates (meters)
    float intensity;            // Reflectivity (0-255 for Livox)
    float time_offset_ms;       // Time offset within scan (milliseconds)
    uint8_t tag;                // Livox point tag
    uint8_t line;               // Scan line number

    LidarPoint() : x(0), y(0), z(0), intensity(0),
                   time_offset_ms(0), tag(0), line(0) {}

    // Get distance from origin
    float range() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    // Get 2D range (horizontal)
    float range2d() const {
        return std::sqrt(x*x + y*y);
    }

    // Convert to Eigen vector
    V3F toEigen() const {
        return V3F(x, y, z);
    }

    V3D toEigenDouble() const {
        return V3D(x, y, z);
    }
};

/**
 * @brief Point cloud container
 */
struct PointCloud {
    std::vector<LidarPoint> points;
    uint64_t timestamp_ns;      // Scan start timestamp
    uint64_t timestamp_end_ns;  // Scan end timestamp

    PointCloud() : timestamp_ns(0), timestamp_end_ns(0) {}

    void clear() { points.clear(); }
    void reserve(size_t n) { points.reserve(n); }
    size_t size() const { return points.size(); }
    bool empty() const { return points.empty(); }

    void push_back(const LidarPoint& pt) { points.push_back(pt); }

    LidarPoint& operator[](size_t i) { return points[i]; }
    const LidarPoint& operator[](size_t i) const { return points[i]; }

    auto begin() { return points.begin(); }
    auto end() { return points.end(); }
    auto begin() const { return points.begin(); }
    auto end() const { return points.end(); }

    double start_time_sec() const {
        return static_cast<double>(timestamp_ns) * 1e-9;
    }

    double end_time_sec() const {
        return static_cast<double>(timestamp_end_ns) * 1e-9;
    }
};

using PointCloudPtr = std::shared_ptr<PointCloud>;

//=============================================================================
// Pose6D Structure (for IMU preintegration)
//=============================================================================

/**
 * @brief 6DOF pose with velocity and IMU data
 * Replaces fast_lio::Pose6D message
 */
struct Pose6D {
    double offset_time;         // Time offset from scan start
    V3D acc;                    // Acceleration at this pose
    V3D gyro;                   // Angular velocity at this pose
    V3D vel;                    // Velocity at this pose
    V3D pos;                    // Position at this pose
    M3D rot;                    // Rotation matrix at this pose

    Pose6D() : offset_time(0), acc(V3D::Zero()), gyro(V3D::Zero()),
               vel(V3D::Zero()), pos(V3D::Zero()), rot(M3D::Identity()) {}
};

//=============================================================================
// Measurement Group (synchronized sensor data)
//=============================================================================

/**
 * @brief Synchronized LiDAR and IMU measurements for one processing cycle
 * Replaces the MeasureGroup struct
 */
struct MeasureGroup {
    double lidar_beg_time;      // LiDAR scan start time (seconds)
    double lidar_end_time;      // LiDAR scan end time (seconds)
    PointCloudPtr lidar;        // LiDAR point cloud
    std::deque<ImuData> imu;    // IMU measurements during this scan

    MeasureGroup() : lidar_beg_time(0), lidar_end_time(0) {
        lidar = std::make_shared<PointCloud>();
    }

    void clear() {
        lidar->clear();
        imu.clear();
        lidar_beg_time = 0;
        lidar_end_time = 0;
    }
};

//=============================================================================
// SLAM State
//=============================================================================

/**
 * @brief Full SLAM state including pose, velocity, and biases
 */
struct SlamState {
    uint64_t timestamp_ns;      // State timestamp
    M3D rot;                    // Rotation (world frame)
    V3D pos;                    // Position (world frame)
    V3D vel;                    // Velocity (world frame)
    V3D bias_gyro;              // Gyroscope bias
    V3D bias_acc;               // Accelerometer bias
    V3D gravity;                // Estimated gravity vector
    Eigen::Matrix<double, DIM_STATE, DIM_STATE> cov;  // State covariance

    SlamState() : timestamp_ns(0),
                  rot(M3D::Identity()),
                  pos(V3D::Zero()),
                  vel(V3D::Zero()),
                  bias_gyro(V3D::Zero()),
                  bias_acc(V3D::Zero()),
                  gravity(V3D(0, 0, -G_m_s2)) {
        cov.setIdentity();
    }

    // Get pose as 4x4 transformation matrix
    M4D getPoseMatrix() const {
        M4D T = M4D::Identity();
        T.block<3,3>(0,0) = rot;
        T.block<3,1>(0,3) = pos;
        return T;
    }

    // Set pose from 4x4 transformation matrix
    void setPoseMatrix(const M4D& T) {
        rot = T.block<3,3>(0,0);
        pos = T.block<3,1>(0,3);
    }

    double timestamp_sec() const {
        return static_cast<double>(timestamp_ns) * 1e-9;
    }
};

//=============================================================================
// Wheel Odometry
//=============================================================================

/**
 * @brief Wheel odometry data from VESC motor controllers
 */
struct WheelOdometry {
    uint64_t timestamp_ns;
    double left_distance;       // Cumulative left wheel distance (meters)
    double right_distance;      // Cumulative right wheel distance (meters)
    double left_velocity;       // Left wheel velocity (m/s)
    double right_velocity;      // Right wheel velocity (m/s)

    WheelOdometry() : timestamp_ns(0), left_distance(0), right_distance(0),
                      left_velocity(0), right_velocity(0) {}

    // Get linear velocity (average of wheels)
    double linear_velocity() const {
        return (left_velocity + right_velocity) / 2.0;
    }

    // Get angular velocity (differential)
    double angular_velocity(double wheel_base) const {
        return (right_velocity - left_velocity) / wheel_base;
    }
};

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Create Pose6D from state components
 */
inline Pose6D makePose6D(double t, const V3D& a, const V3D& g,
                         const V3D& v, const V3D& p, const M3D& R) {
    Pose6D pose;
    pose.offset_time = t;
    pose.acc = a;
    pose.gyro = g;
    pose.vel = v;
    pose.pos = p;
    pose.rot = R;
    return pose;
}

/**
 * @brief Convert degrees to radians
 */
template<typename T>
inline T deg2rad(T degrees) {
    return degrees * static_cast<T>(PI) / static_cast<T>(180);
}

/**
 * @brief Convert radians to degrees
 */
template<typename T>
inline T rad2deg(T radians) {
    return radians * static_cast<T>(180) / static_cast<T>(PI);
}

} // namespace slam

#endif // SLAM_TYPES_HPP
