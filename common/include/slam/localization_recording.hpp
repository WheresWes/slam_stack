/**
 * Localization Session Recording Format (v2)
 *
 * Binary format for recording raw sensor data streams during localization.
 * Extends the FUSN v1 format (from record_fusion_data.cpp) with:
 * - Session header (map filename, position hint, SLAM config)
 * - Efficient batch point cloud records
 * - VESC odometry records
 *
 * Used by:
 * - slam_gui.cpp (recording)
 * - replay_localization.cpp (replay)
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <chrono>

namespace slam {
namespace recording {

// File format constants
constexpr char MAGIC[4] = {'F', 'U', 'S', 'N'};
constexpr uint32_t VERSION = 2;

// ============================================================================
// Record Types
// ============================================================================

enum class RecordType : uint8_t {
    // v1 types (backward compatible)
    IMU = 1,
    POINT_CLOUD_START = 2,   // Not used in v2, but reserved
    POINT_CLOUD_POINT = 3,   // Not used in v2, but reserved
    POINT_CLOUD_END = 4,     // Not used in v2, but reserved
    VESC_STATUS = 5,
    MOTION_COMMAND = 6,
    WHEEL_ODOM = 7,

    // v2 types for localization replay
    POINT_CLOUD_BATCH = 10,  // Entire scan as single record
    VESC_ODOM = 11,          // VESC odometry (velocity, ERPM, tachometer)
};

// ============================================================================
// Record Header (precedes every record)
// ============================================================================

#pragma pack(push, 1)

struct RecordHeader {
    uint64_t timestamp_us;   // Microseconds from recording start
    RecordType type;
    uint32_t size;           // Size of payload following this header
};

// ============================================================================
// Session Header (written once after file header)
// ============================================================================

struct SessionHeader {
    // Map filename (length-prefixed, string follows this struct)
    uint16_t map_filename_len;

    // Position hint
    float hint_x;
    float hint_y;
    float hint_heading_rad;
    float hint_search_radius_m;
    uint8_t hint_valid;
    uint8_t hint_heading_known;

    // SLAM config snapshot
    float voxel_size;
    float gyr_cov;
    float acc_cov;
    float imu_lpf_alpha;
    float blind_distance;
    uint8_t point_filter;
    uint8_t deskew_enabled;
    int32_t max_iterations;
    int32_t max_points_icp;
    float max_position_jump;
    float max_rotation_jump_deg;
};

// ============================================================================
// Record Payloads
// ============================================================================

// IMU record - accelerometer and gyroscope
// NOTE: acc values are in m/s^2 (already converted from g-units)
struct ImuRecord {
    float acc_x, acc_y, acc_z;    // m/s^2
    float gyro_x, gyro_y, gyro_z; // rad/s
};

// Point cloud batch header (followed by LidarPointRecord[num_points])
struct PointCloudBatchHeader {
    uint64_t scan_timestamp_ns;   // Original LiDAR timestamp
    uint32_t num_points;
};

// Individual point within a batch
struct LidarPointRecord {
    float x, y, z;               // meters
    float intensity;             // 0-255
    float time_offset_ms;        // time offset within scan
    uint8_t tag;                 // return type
    uint8_t line;                // scan line
};

// VESC odometry record
struct VescOdomRecord {
    float velocity_left_mps;
    float velocity_right_mps;
    int32_t erpm_left;
    int32_t erpm_right;
    int32_t tach_left;
    int32_t tach_right;
    float linear_vel;            // (left + right) / 2
    float angular_vel;           // from wheel differential
};

#pragma pack(pop)

// ============================================================================
// Helper: Write file header
// ============================================================================
inline bool writeFileHeader(std::ofstream& file) {
    file.write(MAGIC, 4);
    file.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
    return file.good();
}

// ============================================================================
// Helper: Write a record with header
// ============================================================================
template<typename T>
inline bool writeRecord(std::ofstream& file, uint64_t timestamp_us,
                        RecordType type, const T& payload) {
    RecordHeader hdr;
    hdr.timestamp_us = timestamp_us;
    hdr.type = type;
    hdr.size = sizeof(T);
    file.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    file.write(reinterpret_cast<const char*>(&payload), sizeof(payload));
    return file.good();
}

// ============================================================================
// Helper: Compute elapsed microseconds from recording start
// ============================================================================
inline uint64_t elapsedUs(const std::chrono::steady_clock::time_point& start) {
    auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now - start).count());
}

} // namespace recording
} // namespace slam
