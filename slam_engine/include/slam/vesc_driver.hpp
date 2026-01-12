#pragma once

/**
 * VESC CAN Bus Driver for Differential/Skid-Steer Drive
 *
 * Features:
 * - Open-loop duty cycle control for smooth operation
 * - Per-wheel scaling calibration (lookup table + interpolation)
 * - Minimum duty thresholds (starting vs keeping)
 * - Odometry from tachometer
 * - Combined calibration + localization sequence
 */

#include <cstdint>
#include <vector>
#include <array>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>
#include <string>

namespace slam {

// Forward declarations
class VescCanInterface;

/**
 * Robot physical parameters
 */
struct RobotGeometry {
    float track_width_mm = 120.0f;      // Center-to-center of treads
    float wheel_base_mm = 95.0f;        // Front to rear axle
    float tread_width_mm = 40.0f;       // Width of each tread
    float wheel_diameter_mm = 0.0f;     // If known, otherwise derived from calibration

    // Derived/calibrated values
    float effective_track_mm = 156.0f;  // Includes scrub factor (~1.3x)
    float ticks_per_meter = 14093.0f;   // From odometry calibration
};

/**
 * Per-wheel calibration data
 */
struct WheelCalibration {
    uint8_t vesc_id = 0;
    float ticks_per_meter = 14093.0f;

    // Duty scaling lookup table: (duty, scale_factor)
    // Scale factor multiplies duty to match the slower wheel
    std::vector<std::pair<float, float>> duty_scale_table = {
        {0.030f, 0.622f},
        {0.035f, 0.800f},
        {0.040f, 0.816f},
        {0.050f, 0.867f},
        {0.060f, 0.878f},
        {0.070f, 0.883f},
    };

    // Minimum duty thresholds
    float min_duty_start = 0.040f;  // From standstill (static friction)
    float min_duty_keep = 0.030f;   // While moving (kinetic friction)

    // Interpolate scale factor for given duty
    float getScaleFactor(float duty) const;
};

/**
 * Motor calibration results
 */
struct CalibrationResult {
    WheelCalibration left;
    WheelCalibration right;
    RobotGeometry geometry;

    // Rotation calibration
    float ticks_per_radian = 0.0f;  // Average of both wheels
    float effective_track_mm = 0.0f;

    bool valid = false;
    std::string error_message;
};

/**
 * VESC status data
 */
struct VescStatus {
    int32_t erpm = 0;
    float current = 0.0f;
    float duty = 0.0f;
    float voltage = 0.0f;
    float temp_fet = 0.0f;
    float temp_motor = 0.0f;
    int32_t tachometer = 0;
    uint64_t last_update_us = 0;
};

/**
 * VESC Odometry data (raw motor encoder readings)
 */
struct VescOdometry {
    float distance_left_m = 0.0f;
    float distance_right_m = 0.0f;
    float velocity_left_mps = 0.0f;
    float velocity_right_mps = 0.0f;
    int32_t tach_left = 0;
    int32_t tach_right = 0;
    uint64_t timestamp_us = 0;
};

/**
 * Callback types
 */
using OdometryCallback = std::function<void(const VescOdometry&)>;
using StatusCallback = std::function<void(uint8_t vesc_id, const VescStatus&)>;

/**
 * VESC Motor Controller
 *
 * Handles communication with two VESCs for differential/skid-steer drive.
 */
class VescDriver {
public:
    VescDriver();
    ~VescDriver();

    /**
     * Initialize the driver
     * @param port CAN interface (e.g., "COM3" on Windows, "can0" on Linux)
     * @param vesc_left VESC ID for left motor(s)
     * @param vesc_right VESC ID for right motor(s)
     * @return true if successful
     */
    bool init(const std::string& port, uint8_t vesc_left = 1, uint8_t vesc_right = 126);

    /**
     * Shutdown the driver
     */
    void shutdown();

    /**
     * Check if driver is connected
     */
    bool isConnected() const { return connected_; }

    // ========== Motor Control ==========

    /**
     * Set duty cycle for both wheels with calibration applied
     * @param duty Target duty cycle (0.0 to 1.0)
     * @param apply_scaling Whether to apply per-wheel scaling
     */
    void setDuty(float duty, bool apply_scaling = true);

    /**
     * Set raw duty cycle to each wheel (for calibration)
     */
    void setDutyRaw(float duty_left, float duty_right);

    /**
     * Set differential duty for turning
     * @param linear Forward/backward component (-1 to 1)
     * @param angular Rotation component (-1 to 1, positive = left)
     */
    void setDutyDifferential(float linear, float angular);

    /**
     * Stop both motors immediately
     */
    void stop();

    /**
     * Ramp to target duty over duration
     */
    void rampTo(float target_duty, float duration_s = 1.0f, bool apply_scaling = true);

    /**
     * Set RPM (ERPM) for both wheels - closed-loop velocity control
     * VESC PID handles motor asymmetry automatically
     * @param erpm_left Target ERPM for left wheel
     * @param erpm_right Target ERPM for right wheel
     */
    void setRPM(int32_t erpm_left, int32_t erpm_right);

    // ========== Odometry ==========

    /**
     * Get current wheel odometry
     */
    VescOdometry getOdometry() const;

    /**
     * Reset odometry to zero
     */
    void resetOdometry();

    /**
     * Set odometry callback (called from CAN receive thread)
     */
    void setOdometryCallback(OdometryCallback callback);

    // ========== Status ==========

    /**
     * Get VESC status
     */
    VescStatus getStatus(uint8_t vesc_id) const;

    /**
     * Set status callback
     */
    void setStatusCallback(StatusCallback callback);

    // ========== Calibration ==========

    /**
     * Load calibration from file
     */
    bool loadCalibration(const std::string& path);

    /**
     * Save calibration to file
     */
    bool saveCalibration(const std::string& path) const;

    /**
     * Apply calibration
     */
    void applyCalibration(const CalibrationResult& cal);

    /**
     * Get current calibration
     */
    CalibrationResult getCalibration() const;

    /**
     * Run duty scaling calibration
     * Robot will drive at various duty levels to measure ERPM per wheel.
     * @param progress_callback Called with progress (0-100) and status message
     * @return Calibration result
     */
    CalibrationResult runDutyCalibration(
        std::function<void(int, const std::string&)> progress_callback = nullptr);

    /**
     * Run minimum duty threshold calibration
     * Tests increasing duty until each wheel starts moving.
     */
    CalibrationResult runMinDutyCalibration(
        std::function<void(int, const std::string&)> progress_callback = nullptr);

    /**
     * Run rotation calibration
     * Robot will rotate to measure ticks per radian and effective track width.
     * Requires IMU data for accurate angle measurement.
     * @param imu_yaw_callback Function that returns current IMU yaw in radians
     */
    CalibrationResult runRotationCalibration(
        std::function<float()> imu_yaw_callback,
        std::function<void(int, const std::string&)> progress_callback = nullptr);

    /**
     * Run combined calibration + localization sequence
     * Movement pattern: Forward → Turn 90° Left → Center → Turn 90° Right
     * Captures point cloud data during movement for global localization.
     *
     * @param imu_yaw_callback Function that returns current IMU yaw in radians
     * @param point_cloud_callback Called with captured point cloud after sequence
     * @param progress_callback Called with progress updates
     */
    CalibrationResult runCalibrationWithLocalization(
        std::function<float()> imu_yaw_callback,
        std::function<void(const std::vector<std::array<float, 3>>&)> point_cloud_callback,
        std::function<void(int, const std::string&)> progress_callback = nullptr);

private:
    // CAN communication
    std::unique_ptr<VescCanInterface> can_;
    std::atomic<bool> connected_{false};

    // VESC IDs
    uint8_t vesc_left_ = 1;
    uint8_t vesc_right_ = 126;

    // Calibration data
    WheelCalibration cal_left_;
    WheelCalibration cal_right_;
    RobotGeometry geometry_;

    // State
    std::atomic<bool> is_moving_{false};
    float current_duty_left_ = 0.0f;
    float current_duty_right_ = 0.0f;

    // Status
    mutable std::mutex status_mutex_;
    VescStatus status_left_;
    VescStatus status_right_;

    // Odometry
    mutable std::mutex odom_mutex_;
    int32_t tach_offset_left_ = 0;
    int32_t tach_offset_right_ = 0;
    int32_t last_tach_left_ = 0;
    int32_t last_tach_right_ = 0;
    uint64_t last_odom_time_us_ = 0;
    bool has_tach_left_ = false;   // True after first STATUS_5 received
    bool has_tach_right_ = false;
    bool odom_reset_pending_ = false;  // Deferred reset until STATUS_5 arrives

    // Callbacks
    OdometryCallback odom_callback_;
    StatusCallback status_callback_;

    // Internal methods
    void sendDutyCommand(uint8_t vesc_id, float duty);
    void sendRPMCommand(uint8_t vesc_id, int32_t erpm);
    void processCanMessage(uint32_t can_id, const uint8_t* data, uint8_t len);
    float interpolateScale(float duty, const WheelCalibration& cal) const;
    void applyThresholds(float& duty_left, float& duty_right);
};

} // namespace slam
