#pragma once

/**
 * Sensor Fusion - Wheel Odometry + SLAM Pose Fusion
 *
 * Combines wheel odometry and LiDAR SLAM with motion-state-aware fusion:
 * - STATIONARY: Freeze pose, reject SLAM jitter
 * - STRAIGHT_LINE: Dead-reckon with wheels, slow SLAM correction
 * - TURNING: Trust SLAM for rotation (wheel slip makes odom unreliable)
 *
 * Optional hull surface constraint for deployment on boat hulls:
 * - Projects position onto mesh surface
 * - Aligns roll/pitch to surface normal
 * - Constrains velocity to tangent plane
 */

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <mutex>
#include <memory>
#include <functional>
#include <vector>

namespace slam {

// ============================================================================
// Motion State
// ============================================================================

enum class MotionState {
    STATIONARY,      // Wheels not moving - freeze pose
    STRAIGHT_LINE,   // Forward/backward motion - trust wheel velocity
    TURNING          // Significant angular velocity - trust SLAM rotation
};

// ============================================================================
// Pose Types
// ============================================================================

struct Pose3D {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float roll = 0.0f;   // radians
    float pitch = 0.0f;  // radians
    float yaw = 0.0f;    // radians (heading)

    // Convert to/from Eigen for math operations
    Eigen::Vector3f position() const { return Eigen::Vector3f(x, y, z); }
    void setPosition(const Eigen::Vector3f& p) { x = p.x(); y = p.y(); z = p.z(); }

    Eigen::Quaternionf orientation() const;
    void setOrientation(const Eigen::Quaternionf& q);

    // Convenience: 2D pose (x, y, yaw)
    float heading() const { return yaw; }
};

struct Velocity3D {
    float linear_x = 0.0f;   // m/s forward
    float linear_y = 0.0f;   // m/s lateral (usually 0 for differential drive)
    float linear_z = 0.0f;   // m/s vertical
    float angular_z = 0.0f;  // rad/s yaw rate
};

// ============================================================================
// Configuration
// ============================================================================

struct FusionConfig {
    // Motion state detection thresholds
    int stationary_erpm_threshold = 50;        // ERPM below this = stationary
    float turning_angular_threshold = 0.05f;   // rad/s - above this = turning

    // SLAM low-pass filter coefficients (0 = no filter, 1 = frozen)
    // These determine how much of the new SLAM reading to incorporate
    // alpha=0.15 means: new_smoothed = 0.85 * old_smoothed + 0.15 * new_raw
    float slam_position_alpha = 0.15f;         // Heavy smoothing for position
    float slam_heading_alpha = 0.20f;          // Slightly less for heading

    // Correction rates (per second) in STRAIGHT_LINE mode
    // These determine how fast the fused pose converges to SLAM
    float straight_heading_correction = 0.5f;  // 50% per second
    float straight_position_correction = 0.3f; // 30% per second

    // Correction rate in TURNING mode (faster since wheel odom unreliable)
    float turning_position_correction = 2.0f;  // 200% per second

    // State transition hysteresis (prevents oscillation)
    float stationary_to_moving_erpm = 100;     // Need this ERPM to start moving
    float moving_to_stationary_erpm = 50;      // Below this to stop

    // Hull surface constraint (optional, for deployment)
    bool enable_hull_constraint = false;
    float hull_snap_distance = 0.1f;           // Max distance to snap to surface (m)
};

// ============================================================================
// Hull Surface Constraint (forward declaration)
// ============================================================================

class HullConstraint;

// ============================================================================
// Sensor Fusion Class
// ============================================================================

class SensorFusion {
public:
    SensorFusion();
    ~SensorFusion();

    /**
     * Initialize with configuration
     */
    void init(const FusionConfig& config = FusionConfig());

    /**
     * Reset fusion state to given pose
     */
    void reset(const Pose3D& initial_pose = Pose3D());

    // ========== Sensor Updates ==========

    /**
     * Update from wheel odometry (call at ~50Hz from MotionController)
     * @param linear_velocity Forward velocity in m/s (positive = forward)
     * @param angular_command Angular command in rad/s (positive = left)
     * @param erpm_left Left wheel ERPM (for stationary detection)
     * @param erpm_right Right wheel ERPM (for stationary detection)
     * @param dt Time since last wheel update (seconds)
     */
    void updateWheelOdometry(float linear_velocity, float angular_command,
                             int32_t erpm_left, int32_t erpm_right, float dt);

    /**
     * Update from SLAM (call at ~10Hz from SlamEngine)
     * @param slam_pose 6DOF pose from SLAM (x, y, z, roll, pitch, yaw)
     * @param timestamp_us Timestamp in microseconds (for latency compensation)
     */
    void updateSlamPose(const Pose3D& slam_pose, uint64_t timestamp_us = 0);

    /**
     * Update from SLAM using 4x4 transformation matrix
     */
    void updateSlamPose(const Eigen::Matrix4f& slam_transform, uint64_t timestamp_us = 0);

    // ========== Output ==========

    /**
     * Get fused pose (thread-safe)
     */
    Pose3D getFusedPose() const;

    /**
     * Get fused velocity (thread-safe)
     */
    Velocity3D getFusedVelocity() const;

    /**
     * Get current motion state
     */
    MotionState getMotionState() const;

    /**
     * Get motion state as string (for debugging)
     */
    const char* getMotionStateString() const;

    /**
     * Get raw (unfiltered) SLAM pose
     */
    Pose3D getRawSlamPose() const;

    /**
     * Get smoothed SLAM pose (after low-pass filter)
     */
    Pose3D getSmoothedSlamPose() const;

    // ========== Configuration ==========

    /**
     * Update configuration
     */
    void setConfig(const FusionConfig& config);

    /**
     * Get current configuration
     */
    FusionConfig getConfig() const;

    // ========== Hull Constraint ==========

    /**
     * Set hull mesh for surface constraint
     * @param mesh Triangle mesh representing hull surface
     */
    void setHullMesh(std::shared_ptr<HullConstraint> hull);

    /**
     * Enable/disable hull constraint
     */
    void setHullConstraintEnabled(bool enabled);

    // ========== Callbacks ==========

    /**
     * Callback when motion state changes
     */
    using MotionStateCallback = std::function<void(MotionState old_state, MotionState new_state)>;
    void setMotionStateCallback(MotionStateCallback callback);

    // ========== Debugging ==========

    /**
     * Get debug info as formatted string
     */
    std::string getDebugInfo() const;

private:
    FusionConfig config_;

    // Fused state (output)
    mutable std::mutex mutex_;
    Pose3D fused_pose_;
    Velocity3D fused_velocity_;
    MotionState motion_state_ = MotionState::STATIONARY;

    // SLAM state
    Pose3D slam_raw_;           // Latest raw SLAM pose
    Pose3D slam_smoothed_;      // Low-pass filtered SLAM pose
    uint64_t last_slam_time_us_ = 0;
    bool slam_initialized_ = false;

    // Wheel odometry state
    float wheel_linear_vel_ = 0.0f;
    float wheel_angular_cmd_ = 0.0f;
    int32_t erpm_left_ = 0;
    int32_t erpm_right_ = 0;

    // Hull constraint (optional)
    std::shared_ptr<HullConstraint> hull_constraint_;

    // Callback
    MotionStateCallback state_callback_;

    // ========== Internal Methods ==========

    /**
     * Detect motion state from wheel feedback
     */
    MotionState detectMotionState() const;

    /**
     * Apply fusion based on current motion state
     */
    void applyFusion(float dt);

    /**
     * Apply hull surface constraint
     */
    void applyHullConstraint();

    /**
     * Normalize angle to [-pi, pi]
     */
    static float normalizeAngle(float angle);

    /**
     * Compute angle difference (handles wraparound)
     */
    static float angleDiff(float target, float current);

    /**
     * Linear interpolation
     */
    static float lerp(float a, float b, float t);

    /**
     * Angle interpolation (handles wraparound)
     */
    static float lerpAngle(float a, float b, float t);
};

// ============================================================================
// Hull Constraint Interface
// ============================================================================

/**
 * Interface for hull surface constraint
 * Implement this for specific mesh representations (e.g., Open3D, custom)
 */
class HullConstraint {
public:
    virtual ~HullConstraint() = default;

    /**
     * Project a 3D point onto the nearest surface point
     * @param point Input point
     * @param max_distance Maximum search distance (return input if no surface found)
     * @return Closest point on surface
     */
    virtual Eigen::Vector3f projectToSurface(
        const Eigen::Vector3f& point,
        float max_distance = 0.5f) const = 0;

    /**
     * Get surface normal at a point
     * @param point Point on or near surface
     * @return Unit normal vector (pointing outward from hull)
     */
    virtual Eigen::Vector3f getSurfaceNormal(const Eigen::Vector3f& point) const = 0;

    /**
     * Check if a point is within a certain distance of the surface
     */
    virtual bool isOnSurface(const Eigen::Vector3f& point, float tolerance = 0.05f) const = 0;

    /**
     * Get the distance from a point to the nearest surface
     */
    virtual float distanceToSurface(const Eigen::Vector3f& point) const = 0;
};

/**
 * Simple planar hull constraint (for testing)
 * Assumes hull is a flat plane at z=0
 */
class PlanarHullConstraint : public HullConstraint {
public:
    PlanarHullConstraint(float z_height = 0.0f) : z_height_(z_height) {}

    Eigen::Vector3f projectToSurface(const Eigen::Vector3f& point, float) const override {
        return Eigen::Vector3f(point.x(), point.y(), z_height_);
    }

    Eigen::Vector3f getSurfaceNormal(const Eigen::Vector3f&) const override {
        return Eigen::Vector3f(0, 0, 1);  // Upward normal
    }

    bool isOnSurface(const Eigen::Vector3f& point, float tolerance) const override {
        return std::abs(point.z() - z_height_) < tolerance;
    }

    float distanceToSurface(const Eigen::Vector3f& point) const override {
        return std::abs(point.z() - z_height_);
    }

private:
    float z_height_;
};

} // namespace slam
