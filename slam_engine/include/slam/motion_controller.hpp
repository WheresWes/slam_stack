#pragma once

/**
 * Motion Controller - Calibration-aware motor control
 *
 * Sits between high-level commands (GUI/autonomy) and VescDriver.
 * Applies calibration parameters to ensure accurate, smooth motion.
 *
 * Features:
 * - Per-wheel duty scaling (forward/reverse aware)
 * - Minimum duty threshold enforcement (start vs keep)
 * - Smooth ramping (prevents jerky motion)
 * - Open-loop ↔ closed-loop transitions
 * - Differential drive kinematics
 * - Wheel odometry integration
 */

#include <string>
#include <vector>
#include <mutex>
#include <cstdint>

namespace slam {

// Forward declaration
class VescDriver;

// ============================================================================
// Calibration Data Structures
// ============================================================================

struct DutyScalePoint {
    float duty;
    float scale_right;  // Multiply right wheel duty by this
};

struct DutyErpmPoint {
    float duty;
    int32_t erpm;  // Expected ERPM at this duty (left wheel reference)
};

struct MinDutyThresholds {
    float start_left = 0.0f;   // Duty to overcome static friction
    float start_right = 0.0f;
    float keep_left = 0.0f;    // Duty to maintain motion
    float keep_right = 0.0f;

    // Apply a margin multiplier (e.g., 1.1 = 10% margin)
    MinDutyThresholds withMargin(float margin) const {
        return {
            start_left * margin,
            start_right * margin,
            keep_left * margin,
            keep_right * margin
        };
    }
};

struct MotionCalibration {
    // Duty scaling tables (duty → scale factor for right wheel)
    std::vector<DutyScalePoint> forward_scaling;
    std::vector<DutyScalePoint> reverse_scaling;

    // Duty to ERPM lookup (left wheel reference, for closed-loop control)
    // Used to convert duty command to target ERPM when switching modes
    std::vector<DutyErpmPoint> duty_to_erpm = {
        {0.030f, 891},
        {0.035f, 1563},
        {0.040f, 2046},
        {0.050f, 2801},
        {0.060f, 3877},
        {0.070f, 4645},
        {0.100f, 6600},  // Extrapolated
        {0.120f, 7900},  // Extrapolated
    };

    // Minimum duty thresholds for straight-line motion
    MinDutyThresholds forward_min_duty;
    MinDutyThresholds reverse_min_duty;

    // Minimum duty thresholds for turning (higher friction due to scrubbing)
    // If not calibrated, these default to forward/reverse values + margin
    MinDutyThresholds turning_forward_min_duty;
    MinDutyThresholds turning_reverse_min_duty;
    bool turning_thresholds_calibrated = false;

    // Safety margin to add on top of calibrated min duty values
    // Applied as multiplier: actual_threshold = calibrated * (1 + margin)
    float min_duty_margin = 0.10f;  // 10% margin by default

    // Robot geometry
    float effective_track_m = 0.180f;   // For differential drive kinematics
    float ticks_per_meter = 14093.0f;   // Encoder ticks per meter traveled
    float ticks_per_radian = 1300.0f;   // Encoder ticks per radian turned

    // VESC IDs
    uint8_t vesc_id_left = 1;
    uint8_t vesc_id_right = 126;

    bool valid = false;
};

// ============================================================================
// Motion State
// ============================================================================

struct Pose2D {
    float x = 0.0f;      // meters
    float y = 0.0f;      // meters
    float theta = 0.0f;  // radians
};

struct Velocity2D {
    float linear = 0.0f;   // m/s
    float angular = 0.0f;  // rad/s
};

enum class ControlMode {
    STOPPED,
    OPEN_LOOP,
    CLOSED_LOOP
};

// ============================================================================
// Motion Controller
// ============================================================================

class MotionController {
public:
    MotionController();
    ~MotionController();

    /**
     * Initialize with VESC driver and calibration file
     * @param vesc Pointer to initialized VescDriver (not owned)
     * @param calibration_file Path to vesc_calibration.ini
     */
    bool init(VescDriver* vesc, const std::string& calibration_file);

    /**
     * Initialize with VESC driver and calibration struct
     */
    bool init(VescDriver* vesc, const MotionCalibration& calibration);

    /**
     * Check if initialized
     */
    bool isInitialized() const { return initialized_; }

    // ========== Control Commands ==========

    /**
     * Set velocity (converts to duty internally)
     * Uses closed-loop RPM control when above threshold
     * @param linear_mps Forward velocity in m/s
     * @param angular_radps Angular velocity in rad/s (positive = left)
     */
    void setVelocity(float linear_mps, float angular_radps);

    /**
     * Set duty cycle with differential steering
     * @param linear Forward/backward (-1 to 1)
     * @param angular Turn rate (-1 to 1, positive = left)
     */
    void setDuty(float linear, float angular);

    /**
     * Set per-wheel duty (calibration applied)
     * @param left Left wheel duty (-1 to 1)
     * @param right Right wheel duty (-1 to 1)
     */
    void setWheelDuty(float left, float right);

    /**
     * Set per-wheel duty (bypasses calibration - for testing)
     */
    void setWheelDutyRaw(float left, float right);

    /**
     * Stop with ramp-down
     * @param immediate If true, stop instantly without ramping
     */
    void stop(bool immediate = false);

    /**
     * Emergency stop - immediate, bypasses everything
     */
    void emergencyStop();

    // ========== Update Loop ==========

    /**
     * Update loop - MUST be called periodically (~50Hz)
     * - Applies ramping
     * - Sends commands to VESCs
     * - Updates odometry
     * @param dt Time since last update (seconds)
     * @return false if communication error
     */
    bool update(float dt);

    // ========== Feedback ==========

    Pose2D getPose() const;
    Velocity2D getVelocity() const;
    bool isMoving() const;
    ControlMode getMode() const { return mode_; }

    // Current wheel state
    float getCurrentDutyLeft() const { return current_duty_left_; }
    float getCurrentDutyRight() const { return current_duty_right_; }
    float getTargetDutyLeft() const { return target_duty_left_; }
    float getTargetDutyRight() const { return target_duty_right_; }

    /**
     * Reset odometry to given pose
     */
    void resetPose(const Pose2D& pose = {});

    // ========== Configuration ==========

    /**
     * Set duty ramp rate (duty units per second)
     * Default: 0.3/s (0→0.06 in 200ms)
     */
    void setRampRate(float duty_per_sec);

    /**
     * Set maximum allowed duty cycle. Default: 0.15
     */
    void setMaxDuty(float max_duty);

    /**
     * Set ERPM threshold to consider wheel "moving". Default: 100
     */
    void setMovingThreshold(int erpm);

    /**
     * Get current calibration (for display/debugging)
     */
    const MotionCalibration& getCalibration() const { return cal_; }

    /**
     * Load calibration from file
     */
    bool loadCalibration(const std::string& path);

private:
    VescDriver* vesc_ = nullptr;
    MotionCalibration cal_;
    bool initialized_ = false;

    // Control state
    ControlMode mode_ = ControlMode::STOPPED;
    float target_duty_left_ = 0.0f;
    float target_duty_right_ = 0.0f;
    float current_duty_left_ = 0.0f;
    float current_duty_right_ = 0.0f;
    bool raw_mode_ = false;  // Bypass calibration
    bool straight_line_mode_ = false;  // True when both wheels should go same speed
    bool turning_mode_ = false;  // True when differential steering (use higher min duty)

    // For closed-loop RPM control
    int32_t target_erpm_left_ = 0;
    int32_t target_erpm_right_ = 0;

    // Wheel state (from VESC feedback)
    bool moving_left_ = false;
    bool moving_right_ = false;
    int32_t last_erpm_left_ = 0;
    int32_t last_erpm_right_ = 0;
    float last_duty_left_ = 0.0f;
    float last_duty_right_ = 0.0f;

    // Odometry
    mutable std::mutex odom_mutex_;
    Pose2D pose_;
    Velocity2D velocity_;
    int32_t last_tach_left_ = 0;
    int32_t last_tach_right_ = 0;
    bool odom_initialized_ = false;

    // Configuration
    float ramp_rate_ = 0.3f;           // duty/sec
    float max_duty_ = 0.15f;
    int erpm_moving_threshold_ = 100;

    // Hybrid control thresholds (with hysteresis)
    static constexpr int32_t ERPM_THRESHOLD_UP = 2000;    // Switch to closed-loop above this
    static constexpr int32_t ERPM_THRESHOLD_DOWN = 1800;  // Switch to open-loop below this

    // Transition state (for open↔closed-loop handoff)
    float transition_duty_left_ = 0.0f;
    float transition_duty_right_ = 0.0f;
    int32_t erpm_offset_left_ = 0;   // Offset between actual ERPM and dutyToErpm() at transition
    int32_t erpm_offset_right_ = 0;

    // ========== Internal Methods ==========

    /**
     * Apply duty scaling based on calibration
     * @param duty Input duty (can be negative)
     * @param is_right True for right wheel
     * @return Scaled duty
     */
    float applyScaling(float duty, bool is_right);

    /**
     * Apply minimum duty threshold
     * @param duty Input duty
     * @param is_right True for right wheel
     * @param is_moving True if wheel is currently moving
     * @return Thresholded duty (may be zeroed)
     */
    float applyThreshold(float duty, bool is_right, bool is_moving);

    /**
     * Interpolate scale factor from calibration table
     */
    float interpolateScale(float duty, const std::vector<DutyScalePoint>& table);

    /**
     * Update odometry from wheel encoders
     */
    void updateOdometry(float dt);

    /**
     * Ramp current value toward target
     */
    float rampToward(float current, float target, float rate, float dt);

    /**
     * Convert duty to target ERPM using calibration lookup table
     * @param duty Absolute duty value (0 to 1)
     * @return Target ERPM (always positive, caller handles sign)
     */
    int32_t dutyToErpm(float duty);

    /**
     * Check ERPM and handle mode transitions
     * Called from update() after reading VESC status
     */
    void updateControlMode();

    /**
     * Parse INI-style calibration file
     */
    bool parseCalibrationFile(const std::string& path);
};

} // namespace slam
