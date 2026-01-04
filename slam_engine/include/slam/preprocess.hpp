/**
 * @file preprocess.hpp
 * @brief Point cloud preprocessing for Livox Mid-360 (SDK2 compatible)
 *
 * This is a ROS-free port of FAST-LIO's preprocessor, designed to work
 * directly with Livox SDK2 data structures.
 *
 * Key features:
 * - Intensity/reflectivity preservation throughout the pipeline
 * - Point filtering and blind zone removal
 * - Time offset storage for motion compensation
 */

#ifndef SLAM_PREPROCESS_HPP
#define SLAM_PREPROCESS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

#include <Eigen/Core>

#include "slam/types.hpp"

namespace slam {

//=============================================================================
// Livox SDK2 Compatible Point Types
//=============================================================================

/**
 * @brief Livox SDK2 Cartesian point format
 * Matches LivoxLidarCartesianHighRawPoint from Livox SDK2
 */
struct LivoxPointXYZRTLT {
    int32_t x;              // X coordinate in mm
    int32_t y;              // Y coordinate in mm
    int32_t z;              // Z coordinate in mm
    uint8_t reflectivity;   // Reflectivity (0-255)
    uint8_t tag;            // Point tag (return type, etc.)
    uint8_t line;           // Scan line number
    uint32_t offset_time;   // Time offset from scan start (nanoseconds)
};

/**
 * @brief Livox SDK2 IMU raw data format
 * Matches LivoxLidarImuRawPoint from Livox SDK2
 */
struct LivoxImuPoint {
    float gyro_x;           // rad/s
    float gyro_y;           // rad/s
    float gyro_z;           // rad/s
    float acc_x;            // g (multiply by 9.81 for m/s^2)
    float acc_y;            // g
    float acc_z;            // g
};

//=============================================================================
// Enums
//=============================================================================

enum class LidarType {
    LIVOX_MID360 = 1,
    VELODYNE = 2,
    OUSTER = 3
};

enum class TimeUnit {
    SEC = 0,
    MS = 1,
    US = 2,
    NS = 3
};

//=============================================================================
// Preprocessor Configuration
//=============================================================================

struct PreprocessConfig {
    LidarType lidar_type = LidarType::LIVOX_MID360;
    int n_scans = 4;                    // Number of scan lines (4 for Mid-360)
    double blind_distance = 0.5;        // Minimum distance (meters)
    double max_distance = 100.0;        // Maximum distance (meters)
    int point_filter_num = 1;           // Keep every Nth point
    bool feature_enabled = false;       // Enable feature extraction
    TimeUnit time_unit = TimeUnit::NS;  // Input timestamp unit
};

//=============================================================================
// Preprocessor Class
//=============================================================================

/**
 * @brief Point cloud preprocessor for Livox Mid-360
 *
 * Converts raw Livox SDK2 point data to our internal format,
 * applying filtering and preserving intensity/reflectivity.
 */
class Preprocessor {
public:
    Preprocessor() = default;
    explicit Preprocessor(const PreprocessConfig& config) : config_(config) {
        updateTimeScale();
    }

    /**
     * @brief Set configuration
     */
    void setConfig(const PreprocessConfig& config) {
        config_ = config;
        updateTimeScale();
    }

    /**
     * @brief Process raw Livox SDK2 points
     * @param raw_points Raw point data from Livox SDK2
     * @param num_points Number of points in the array
     * @param scan_timestamp Scan start timestamp (nanoseconds)
     * @param output Output point cloud
     */
    void process(const LivoxPointXYZRTLT* raw_points,
                 size_t num_points,
                 uint64_t scan_timestamp,
                 PointCloud& output) {

        output.clear();
        output.reserve(num_points / config_.point_filter_num);
        output.timestamp_ns = scan_timestamp;

        uint32_t max_offset = 0;
        size_t valid_count = 0;

        for (size_t i = 0; i < num_points; i++) {
            const auto& pt = raw_points[i];

            // Check line number
            if (pt.line >= config_.n_scans) continue;

            // Check tag for valid return
            // Tag format: bits 4-5 indicate return type
            // 0x00 = single return, 0x10 = first of multiple returns
            uint8_t return_type = pt.tag & 0x30;
            if (return_type != 0x00 && return_type != 0x10) continue;

            // Convert from mm to meters
            float x = static_cast<float>(pt.x) * 0.001f;
            float y = static_cast<float>(pt.y) * 0.001f;
            float z = static_cast<float>(pt.z) * 0.001f;

            // Check distance
            float dist_sq = x*x + y*y + z*z;
            float blind_sq = static_cast<float>(config_.blind_distance * config_.blind_distance);
            float max_sq = static_cast<float>(config_.max_distance * config_.max_distance);

            if (dist_sq < blind_sq || dist_sq > max_sq) continue;

            // Point filtering
            valid_count++;
            if (valid_count % config_.point_filter_num != 0) continue;

            // Create output point
            LidarPoint out_pt;
            out_pt.x = x;
            out_pt.y = y;
            out_pt.z = z;

            // PRESERVE INTENSITY/REFLECTIVITY
            out_pt.intensity = static_cast<float>(pt.reflectivity);

            // Convert time offset to milliseconds (for motion compensation)
            out_pt.time_offset_ms = static_cast<float>(pt.offset_time) * time_scale_;

            out_pt.tag = pt.tag;
            out_pt.line = pt.line;

            output.push_back(out_pt);

            // Track max offset for end timestamp
            if (pt.offset_time > max_offset) {
                max_offset = pt.offset_time;
            }
        }

        // Set scan end timestamp
        output.timestamp_end_ns = scan_timestamp + static_cast<uint64_t>(max_offset);
    }

    /**
     * @brief Process raw Livox SDK2 points (vector interface)
     */
    void process(const std::vector<LivoxPointXYZRTLT>& raw_points,
                 uint64_t scan_timestamp,
                 PointCloud& output) {
        process(raw_points.data(), raw_points.size(), scan_timestamp, output);
    }

    /**
     * @brief Convert Livox IMU data to our format
     * @param raw_imu Raw IMU data from Livox SDK2
     * @param timestamp Timestamp in nanoseconds
     * @return Converted IMU data
     */
    static ImuData processImu(const LivoxImuPoint& raw_imu, uint64_t timestamp) {
        ImuData imu;
        imu.timestamp_ns = timestamp;

        // Gyroscope (already in rad/s)
        imu.gyro.x() = raw_imu.gyro_x;
        imu.gyro.y() = raw_imu.gyro_y;
        imu.gyro.z() = raw_imu.gyro_z;

        // Accelerometer (convert from g to m/s^2)
        imu.acc.x() = raw_imu.acc_x * G_m_s2;
        imu.acc.y() = raw_imu.acc_y * G_m_s2;
        imu.acc.z() = raw_imu.acc_z * G_m_s2;

        return imu;
    }

    /**
     * @brief Get current configuration
     */
    const PreprocessConfig& getConfig() const { return config_; }

private:
    void updateTimeScale() {
        // Convert input time units to milliseconds
        switch (config_.time_unit) {
            case TimeUnit::SEC: time_scale_ = 1000.0f; break;
            case TimeUnit::MS:  time_scale_ = 1.0f; break;
            case TimeUnit::US:  time_scale_ = 0.001f; break;
            case TimeUnit::NS:  time_scale_ = 0.000001f; break;
        }
    }

    PreprocessConfig config_;
    float time_scale_ = 0.000001f;  // Default: nanoseconds to milliseconds
};

//=============================================================================
// Point Cloud Utilities
//=============================================================================

/**
 * @brief Transform point cloud by a pose matrix
 * Preserves intensity during transformation
 * @param input Input point cloud
 * @param pose 4x4 transformation matrix
 * @param output Output point cloud (can be same as input)
 */
inline void transformPointCloud(const PointCloud& input,
                                 const M4D& pose,
                                 PointCloud& output) {
    M3D R = pose.block<3, 3>(0, 0);
    V3D t = pose.block<3, 1>(0, 3);

    if (&input != &output) {
        output = input;
    }

    for (auto& pt : output.points) {
        V3D p(pt.x, pt.y, pt.z);
        V3D p_transformed = R * p + t;
        pt.x = static_cast<float>(p_transformed.x());
        pt.y = static_cast<float>(p_transformed.y());
        pt.z = static_cast<float>(p_transformed.z());
        // Intensity is preserved automatically (not modified)
    }
}

/**
 * @brief Transform single point by pose
 * @param pt Input point
 * @param R Rotation matrix
 * @param t Translation vector
 * @return Transformed point (intensity preserved)
 */
inline LidarPoint transformPoint(const LidarPoint& pt,
                                  const M3D& R,
                                  const V3D& t) {
    LidarPoint out = pt;  // Copy all fields including intensity
    V3D p(pt.x, pt.y, pt.z);
    V3D p_transformed = R * p + t;
    out.x = static_cast<float>(p_transformed.x());
    out.y = static_cast<float>(p_transformed.y());
    out.z = static_cast<float>(p_transformed.z());
    return out;
}

/**
 * @brief Downsample point cloud using voxel grid
 * @param input Input point cloud
 * @param voxel_size Voxel size in meters
 * @param output Output downsampled cloud
 */
inline void voxelDownsample(const PointCloud& input,
                            double voxel_size,
                            PointCloud& output) {
    if (input.empty()) {
        output.clear();
        return;
    }

    // Simple spatial hashing approach
    struct VoxelKey {
        int64_t x, y, z;
        bool operator==(const VoxelKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelKeyHash {
        size_t operator()(const VoxelKey& k) const {
            return std::hash<int64_t>()(k.x) ^
                   (std::hash<int64_t>()(k.y) << 1) ^
                   (std::hash<int64_t>()(k.z) << 2);
        }
    };

    std::unordered_map<VoxelKey, std::pair<LidarPoint, int>, VoxelKeyHash> voxels;
    double inv_size = 1.0 / voxel_size;

    for (const auto& pt : input.points) {
        VoxelKey key{
            static_cast<int64_t>(std::floor(pt.x * inv_size)),
            static_cast<int64_t>(std::floor(pt.y * inv_size)),
            static_cast<int64_t>(std::floor(pt.z * inv_size))
        };

        auto it = voxels.find(key);
        if (it == voxels.end()) {
            voxels[key] = {pt, 1};
        } else {
            // Accumulate for averaging
            auto& [acc_pt, count] = it->second;
            acc_pt.x += pt.x;
            acc_pt.y += pt.y;
            acc_pt.z += pt.z;
            acc_pt.intensity += pt.intensity;  // Average intensity too
            count++;
        }
    }

    output.clear();
    output.reserve(voxels.size());
    output.timestamp_ns = input.timestamp_ns;
    output.timestamp_end_ns = input.timestamp_end_ns;

    for (auto& [key, val] : voxels) {
        auto& [pt, count] = val;
        float inv_count = 1.0f / static_cast<float>(count);
        pt.x *= inv_count;
        pt.y *= inv_count;
        pt.z *= inv_count;
        pt.intensity *= inv_count;  // Average intensity
        output.push_back(pt);
    }
}

} // namespace slam

#endif // SLAM_PREPROCESS_HPP
