#include "slam/sensor_fusion.hpp"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace slam {

// ============================================================================
// Pose3D Helper Methods
// ============================================================================

Eigen::Quaternionf Pose3D::orientation() const {
    // ZYX Euler angles to quaternion
    Eigen::AngleAxisf roll_angle(roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitch_angle(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yaw_angle(yaw, Eigen::Vector3f::UnitZ());
    return yaw_angle * pitch_angle * roll_angle;
}

void Pose3D::setOrientation(const Eigen::Quaternionf& q) {
    // Quaternion to ZYX Euler angles
    Eigen::Vector3f euler = q.toRotationMatrix().eulerAngles(2, 1, 0);
    yaw = euler[0];
    pitch = euler[1];
    roll = euler[2];
}

// ============================================================================
// SensorFusion Implementation
// ============================================================================

SensorFusion::SensorFusion() = default;
SensorFusion::~SensorFusion() = default;

void SensorFusion::init(const FusionConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    fused_pose_ = Pose3D();
    fused_velocity_ = Velocity3D();
    motion_state_ = MotionState::STATIONARY;
    slam_initialized_ = false;
}

void SensorFusion::reset(const Pose3D& initial_pose) {
    std::lock_guard<std::mutex> lock(mutex_);
    fused_pose_ = initial_pose;
    slam_smoothed_ = initial_pose;
    slam_raw_ = initial_pose;
    fused_velocity_ = Velocity3D();
    motion_state_ = MotionState::STATIONARY;
    slam_initialized_ = true;
}

void SensorFusion::updateWheelOdometry(float linear_velocity, float angular_command,
                                        int32_t erpm_left, int32_t erpm_right, float dt) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Store wheel state
    wheel_linear_vel_ = linear_velocity;
    wheel_angular_cmd_ = angular_command;
    erpm_left_ = erpm_left;
    erpm_right_ = erpm_right;

    // Detect motion state
    MotionState new_state = detectMotionState();

    // Handle state transitions with callback
    if (new_state != motion_state_) {
        MotionState old_state = motion_state_;
        motion_state_ = new_state;
        if (state_callback_) {
            state_callback_(old_state, new_state);
        }
    }

    // Apply fusion based on motion state
    if (slam_initialized_ && dt > 0.0f) {
        applyFusion(dt);
    }

    // Update velocity estimate
    fused_velocity_.linear_x = linear_velocity;
    fused_velocity_.linear_y = 0.0f;  // No lateral velocity for diff drive
    fused_velocity_.linear_z = 0.0f;
    // Angular velocity: use SLAM-derived during turns, wheel command during straight
    if (motion_state_ == MotionState::TURNING) {
        // Will be updated by SLAM
    } else {
        fused_velocity_.angular_z = angular_command;
    }
}

void SensorFusion::updateSlamPose(const Pose3D& slam_pose, uint64_t timestamp_us) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Store raw SLAM pose
    slam_raw_ = slam_pose;

    // Initialize if first SLAM update
    if (!slam_initialized_) {
        slam_smoothed_ = slam_pose;
        fused_pose_ = slam_pose;
        slam_initialized_ = true;
        last_slam_time_us_ = timestamp_us;
        return;
    }

    // Apply low-pass filter to SLAM pose
    float pos_alpha = config_.slam_position_alpha;
    float heading_alpha = config_.slam_heading_alpha;

    slam_smoothed_.x = lerp(slam_smoothed_.x, slam_raw_.x, pos_alpha);
    slam_smoothed_.y = lerp(slam_smoothed_.y, slam_raw_.y, pos_alpha);
    slam_smoothed_.z = lerp(slam_smoothed_.z, slam_raw_.z, pos_alpha);
    slam_smoothed_.roll = lerpAngle(slam_smoothed_.roll, slam_raw_.roll, pos_alpha);
    slam_smoothed_.pitch = lerpAngle(slam_smoothed_.pitch, slam_raw_.pitch, pos_alpha);
    slam_smoothed_.yaw = lerpAngle(slam_smoothed_.yaw, slam_raw_.yaw, heading_alpha);

    // Compute SLAM angular velocity (for turning state)
    if (timestamp_us > last_slam_time_us_) {
        float dt_slam = (timestamp_us - last_slam_time_us_) / 1e6f;
        if (dt_slam > 0.0f && dt_slam < 1.0f) {  // Sanity check
            // Note: This is a rough estimate, could be improved with proper differentiation
            // fused_velocity_.angular_z = angleDiff(slam_raw_.yaw, slam_smoothed_.yaw) / dt_slam;
        }
    }
    last_slam_time_us_ = timestamp_us;
}

void SensorFusion::updateSlamPose(const Eigen::Matrix4f& slam_transform, uint64_t timestamp_us) {
    // Extract pose from 4x4 transformation matrix
    Pose3D pose;
    pose.x = slam_transform(0, 3);
    pose.y = slam_transform(1, 3);
    pose.z = slam_transform(2, 3);

    // Extract rotation (assuming ZYX Euler)
    Eigen::Matrix3f rot = slam_transform.block<3, 3>(0, 0);
    Eigen::Vector3f euler = rot.eulerAngles(2, 1, 0);  // ZYX order
    pose.yaw = euler[0];
    pose.pitch = euler[1];
    pose.roll = euler[2];

    updateSlamPose(pose, timestamp_us);
}

Pose3D SensorFusion::getFusedPose() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return fused_pose_;
}

Velocity3D SensorFusion::getFusedVelocity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return fused_velocity_;
}

MotionState SensorFusion::getMotionState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return motion_state_;
}

const char* SensorFusion::getMotionStateString() const {
    std::lock_guard<std::mutex> lock(mutex_);
    switch (motion_state_) {
        case MotionState::STATIONARY: return "STATIONARY";
        case MotionState::STRAIGHT_LINE: return "STRAIGHT_LINE";
        case MotionState::TURNING: return "TURNING";
        default: return "UNKNOWN";
    }
}

Pose3D SensorFusion::getRawSlamPose() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return slam_raw_;
}

Pose3D SensorFusion::getSmoothedSlamPose() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return slam_smoothed_;
}

void SensorFusion::setConfig(const FusionConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

FusionConfig SensorFusion::getConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

void SensorFusion::setHullMesh(std::shared_ptr<HullConstraint> hull) {
    std::lock_guard<std::mutex> lock(mutex_);
    hull_constraint_ = hull;
}

void SensorFusion::setHullConstraintEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.enable_hull_constraint = enabled;
}

void SensorFusion::setMotionStateCallback(MotionStateCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    state_callback_ = callback;
}

std::string SensorFusion::getDebugInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);

    ss << "=== Sensor Fusion Debug ===" << "\n";
    ss << "Motion State: " << getMotionStateString() << "\n";
    ss << "ERPM: L=" << erpm_left_ << " R=" << erpm_right_ << "\n";
    ss << "Wheel: vel=" << wheel_linear_vel_ << " m/s, ang=" << wheel_angular_cmd_ << " rad/s\n";
    ss << "\n";
    ss << "Fused Pose:\n";
    ss << "  x=" << fused_pose_.x << " y=" << fused_pose_.y << " z=" << fused_pose_.z << "\n";
    ss << "  yaw=" << (fused_pose_.yaw * 180.0f / M_PI) << " deg\n";
    ss << "\n";
    ss << "SLAM Raw:\n";
    ss << "  x=" << slam_raw_.x << " y=" << slam_raw_.y << " z=" << slam_raw_.z << "\n";
    ss << "  yaw=" << (slam_raw_.yaw * 180.0f / M_PI) << " deg\n";
    ss << "\n";
    ss << "SLAM Smoothed:\n";
    ss << "  x=" << slam_smoothed_.x << " y=" << slam_smoothed_.y << " z=" << slam_smoothed_.z << "\n";
    ss << "  yaw=" << (slam_smoothed_.yaw * 180.0f / M_PI) << " deg\n";

    return ss.str();
}

// ============================================================================
// Internal Methods
// ============================================================================

MotionState SensorFusion::detectMotionState() const {
    // Use absolute ERPM values
    int abs_erpm_left = std::abs(erpm_left_);
    int abs_erpm_right = std::abs(erpm_right_);

    // Hysteresis for stationary detection
    int threshold;
    if (motion_state_ == MotionState::STATIONARY) {
        threshold = static_cast<int>(config_.stationary_to_moving_erpm);
    } else {
        threshold = static_cast<int>(config_.moving_to_stationary_erpm);
    }

    // Check if stationary
    if (abs_erpm_left < threshold && abs_erpm_right < threshold) {
        return MotionState::STATIONARY;
    }

    // Check if turning (based on angular command, not wheel difference)
    // Using command is more reliable than wheel ERPM difference
    if (std::abs(wheel_angular_cmd_) > config_.turning_angular_threshold) {
        return MotionState::TURNING;
    }

    // Otherwise, straight line motion
    return MotionState::STRAIGHT_LINE;
}

void SensorFusion::applyFusion(float dt) {
    switch (motion_state_) {
        case MotionState::STATIONARY:
            // ========== STATIONARY: Freeze pose ==========
            // Don't update anything - we trust wheels saying no motion
            // This completely rejects SLAM jitter when stationary
            break;

        case MotionState::STRAIGHT_LINE: {
            // ========== STRAIGHT LINE: Wheel dead-reckoning + SLAM correction ==========

            // 1. Dead-reckon position using wheel velocity
            //    Apply "no lateral motion" constraint implicitly by only using forward delta
            float forward_delta = wheel_linear_vel_ * dt;
            fused_pose_.x += forward_delta * std::cos(fused_pose_.yaw);
            fused_pose_.y += forward_delta * std::sin(fused_pose_.yaw);

            // 2. Slowly correct heading from smoothed SLAM
            float heading_error = angleDiff(slam_smoothed_.yaw, fused_pose_.yaw);
            fused_pose_.yaw += heading_error * config_.straight_heading_correction * dt;
            fused_pose_.yaw = normalizeAngle(fused_pose_.yaw);

            // 3. Slowly correct position from smoothed SLAM
            float pos_error_x = slam_smoothed_.x - fused_pose_.x;
            float pos_error_y = slam_smoothed_.y - fused_pose_.y;
            fused_pose_.x += pos_error_x * config_.straight_position_correction * dt;
            fused_pose_.y += pos_error_y * config_.straight_position_correction * dt;

            // 4. Z, roll, pitch: follow SLAM (we don't have wheel info for these)
            fused_pose_.z = lerp(fused_pose_.z, slam_smoothed_.z, config_.straight_position_correction * dt);
            fused_pose_.roll = lerpAngle(fused_pose_.roll, slam_smoothed_.roll, config_.straight_position_correction * dt);
            fused_pose_.pitch = lerpAngle(fused_pose_.pitch, slam_smoothed_.pitch, config_.straight_position_correction * dt);

            break;
        }

        case MotionState::TURNING: {
            // ========== TURNING: Trust SLAM ==========
            // Wheel odometry is unreliable during skid-steer turns

            // 1. Heading: Trust smoothed SLAM completely
            fused_pose_.yaw = slam_smoothed_.yaw;

            // 2. Position: Fast correction toward SLAM
            float correction_rate = config_.turning_position_correction * dt;
            correction_rate = std::min(correction_rate, 1.0f);  // Clamp to prevent overshoot

            fused_pose_.x = lerp(fused_pose_.x, slam_smoothed_.x, correction_rate);
            fused_pose_.y = lerp(fused_pose_.y, slam_smoothed_.y, correction_rate);
            fused_pose_.z = lerp(fused_pose_.z, slam_smoothed_.z, correction_rate);

            // 3. Roll/pitch: Follow SLAM
            fused_pose_.roll = lerpAngle(fused_pose_.roll, slam_smoothed_.roll, correction_rate);
            fused_pose_.pitch = lerpAngle(fused_pose_.pitch, slam_smoothed_.pitch, correction_rate);

            break;
        }
    }

    // Apply hull surface constraint if enabled
    if (config_.enable_hull_constraint) {
        applyHullConstraint();
    }
}

void SensorFusion::applyHullConstraint() {
    if (!hull_constraint_) return;

    Eigen::Vector3f pos = fused_pose_.position();

    // Check if we're close enough to the surface
    float dist = hull_constraint_->distanceToSurface(pos);
    if (dist > config_.hull_snap_distance) {
        // Too far from surface - don't constrain (might be in transition)
        return;
    }

    // 1. Project position onto hull surface
    Eigen::Vector3f projected = hull_constraint_->projectToSurface(pos, config_.hull_snap_distance);
    fused_pose_.setPosition(projected);

    // 2. Align roll/pitch to surface normal (keep yaw)
    Eigen::Vector3f normal = hull_constraint_->getSurfaceNormal(projected);

    // Compute roll and pitch from surface normal
    // Assuming robot's "up" vector should align with surface normal
    // This is a simplification - full implementation would use robot frame
    if (std::abs(normal.z()) > 0.1f) {  // Surface not vertical
        fused_pose_.roll = std::atan2(-normal.y(), normal.z());
        fused_pose_.pitch = std::atan2(normal.x(), normal.z());
    }
}

float SensorFusion::normalizeAngle(float angle) {
    constexpr float PI_F = static_cast<float>(M_PI);
    while (angle > PI_F) angle -= 2.0f * PI_F;
    while (angle < -PI_F) angle += 2.0f * PI_F;
    return angle;
}

float SensorFusion::angleDiff(float target, float current) {
    float diff = target - current;
    return normalizeAngle(diff);
}

float SensorFusion::lerp(float a, float b, float t) {
    return a + t * (b - a);
}

float SensorFusion::lerpAngle(float a, float b, float t) {
    float diff = angleDiff(b, a);
    return normalizeAngle(a + t * diff);
}

} // namespace slam
