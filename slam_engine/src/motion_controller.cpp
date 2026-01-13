#include "slam/motion_controller.hpp"
#include "slam/vesc_driver.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace slam {

// ============================================================================
// Construction / Initialization
// ============================================================================

MotionController::MotionController() = default;

MotionController::~MotionController() {
    if (initialized_ && vesc_) {
        emergencyStop();
    }
}

bool MotionController::init(VescDriver* vesc, const std::string& calibration_file) {
    if (!vesc) {
        std::cerr << "[MotionController] Error: null VescDriver" << std::endl;
        return false;
    }

    vesc_ = vesc;

    if (!loadCalibration(calibration_file)) {
        std::cerr << "[MotionController] Warning: using default calibration" << std::endl;
        // Set reasonable defaults
        cal_.forward_scaling = {{0.035f, 0.85f}, {0.05f, 0.88f}, {0.065f, 0.90f}};
        cal_.reverse_scaling = {{0.035f, 1.15f}, {0.05f, 1.12f}, {0.065f, 1.10f}};
        cal_.forward_min_duty = {0.025f, 0.020f, 0.020f, 0.018f};
        cal_.reverse_min_duty = {0.025f, 0.020f, 0.020f, 0.020f};
        cal_.valid = true;
    }

    // Initialize odometry from current tachometer
    if (vesc_->isConnected()) {
        VescOdometry odom = vesc_->getOdometry();
        last_tach_left_ = odom.tach_left;
        last_tach_right_ = odom.tach_right;
        odom_initialized_ = true;
    }

    initialized_ = true;
    std::cout << "[MotionController] Initialized" << std::endl;
    std::cout << "  Effective track: " << (cal_.effective_track_m * 1000) << " mm" << std::endl;
    std::cout << "  Forward scaling points: " << cal_.forward_scaling.size() << std::endl;
    std::cout << "  Reverse scaling points: " << cal_.reverse_scaling.size() << std::endl;
    std::cout << "  Turning thresholds: " << (cal_.turning_thresholds_calibrated ? "calibrated" : "using margin") << std::endl;
    std::cout << "  Min duty margin: " << (cal_.min_duty_margin * 100) << "%" << std::endl;

    return true;
}

bool MotionController::init(VescDriver* vesc, const MotionCalibration& calibration) {
    if (!vesc) {
        std::cerr << "[MotionController] Error: null VescDriver" << std::endl;
        return false;
    }

    vesc_ = vesc;
    cal_ = calibration;

    if (vesc_->isConnected()) {
        VescOdometry odom = vesc_->getOdometry();
        last_tach_left_ = odom.tach_left;
        last_tach_right_ = odom.tach_right;
        odom_initialized_ = true;
    }

    initialized_ = true;
    return true;
}

// ============================================================================
// Calibration Loading
// ============================================================================

bool MotionController::loadCalibration(const std::string& path) {
    return parseCalibrationFile(path);
}

bool MotionController::parseCalibrationFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[MotionController] Cannot open calibration file: " << path << std::endl;
        return false;
    }

    std::string line;
    std::string current_section;

    cal_.forward_scaling.clear();
    cal_.reverse_scaling.clear();

    while (std::getline(file, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        // Skip comments
        if (line[0] == '#' || line[0] == ';') continue;

        // Section header
        if (line[0] == '[') {
            size_t end = line.find(']');
            if (end != std::string::npos) {
                current_section = line.substr(1, end - 1);
            }
            continue;
        }

        // Key=value
        size_t eq = line.find('=');
        if (eq == std::string::npos) {
            // Could be duty:scale format in scaling section
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                float duty = std::stof(line.substr(0, colon));
                float scale = std::stof(line.substr(colon + 1));

                if (current_section == "forward_duty_scale") {
                    cal_.forward_scaling.push_back({duty, scale});
                } else if (current_section == "reverse_duty_scale") {
                    cal_.reverse_scaling.push_back({duty, scale});
                }
            }
            continue;
        }

        std::string key = line.substr(0, eq);
        std::string value = line.substr(eq + 1);

        // Trim key
        size_t key_end = key.find_last_not_of(" \t");
        if (key_end != std::string::npos) key = key.substr(0, key_end + 1);

        // Trim value
        size_t val_start = value.find_first_not_of(" \t");
        if (val_start != std::string::npos) value = value.substr(val_start);

        // Parse by section
        if (current_section == "geometry") {
            if (key == "effective_track_mm") {
                cal_.effective_track_m = std::stof(value) / 1000.0f;
            }
        }
        else if (current_section == "rotation") {
            if (key == "ticks_per_radian_left" || key == "ticks_per_radian_right") {
                // Average them
                cal_.ticks_per_radian = std::stof(value);
            }
        }
        else if (current_section == "vesc") {
            if (key == "vesc_id_left") {
                cal_.vesc_id_left = static_cast<uint8_t>(std::stoi(value));
            } else if (key == "vesc_id_right") {
                cal_.vesc_id_right = static_cast<uint8_t>(std::stoi(value));
            }
        }
        else if (current_section == "forward_min_duty") {
            if (key == "start_left") cal_.forward_min_duty.start_left = std::stof(value);
            else if (key == "start_right") cal_.forward_min_duty.start_right = std::stof(value);
            else if (key == "keep_left") cal_.forward_min_duty.keep_left = std::stof(value);
            else if (key == "keep_right") cal_.forward_min_duty.keep_right = std::stof(value);
        }
        else if (current_section == "reverse_min_duty") {
            if (key == "start_left") cal_.reverse_min_duty.start_left = std::stof(value);
            else if (key == "start_right") cal_.reverse_min_duty.start_right = std::stof(value);
            else if (key == "keep_left") cal_.reverse_min_duty.keep_left = std::stof(value);
            else if (key == "keep_right") cal_.reverse_min_duty.keep_right = std::stof(value);
        }
        else if (current_section == "turning_forward_min_duty") {
            cal_.turning_thresholds_calibrated = true;
            if (key == "start_left") cal_.turning_forward_min_duty.start_left = std::stof(value);
            else if (key == "start_right") cal_.turning_forward_min_duty.start_right = std::stof(value);
            else if (key == "keep_left") cal_.turning_forward_min_duty.keep_left = std::stof(value);
            else if (key == "keep_right") cal_.turning_forward_min_duty.keep_right = std::stof(value);
        }
        else if (current_section == "turning_reverse_min_duty") {
            cal_.turning_thresholds_calibrated = true;
            if (key == "start_left") cal_.turning_reverse_min_duty.start_left = std::stof(value);
            else if (key == "start_right") cal_.turning_reverse_min_duty.start_right = std::stof(value);
            else if (key == "keep_left") cal_.turning_reverse_min_duty.keep_left = std::stof(value);
            else if (key == "keep_right") cal_.turning_reverse_min_duty.keep_right = std::stof(value);
        }
        else if (current_section == "safety") {
            if (key == "min_duty_margin") cal_.min_duty_margin = std::stof(value);
        }
    }

    // Sort scaling tables by duty
    std::sort(cal_.forward_scaling.begin(), cal_.forward_scaling.end(),
              [](const auto& a, const auto& b) { return a.duty < b.duty; });
    std::sort(cal_.reverse_scaling.begin(), cal_.reverse_scaling.end(),
              [](const auto& a, const auto& b) { return a.duty < b.duty; });

    cal_.valid = !cal_.forward_scaling.empty() || !cal_.reverse_scaling.empty();

    std::cout << "[MotionController] Loaded calibration from " << path << std::endl;
    return cal_.valid;
}

// ============================================================================
// Control Commands
// ============================================================================

void MotionController::setVelocity(float linear_mps, float angular_radps) {
    if (!initialized_) return;

    // Convert to wheel velocities using differential drive kinematics
    // v_left = v_linear - omega * track/2
    // v_right = v_linear + omega * track/2
    float half_track = cal_.effective_track_m / 2.0f;
    float v_left = linear_mps - angular_radps * half_track;
    float v_right = linear_mps + angular_radps * half_track;

    // Convert velocity to duty cycle
    // Calibration baseline: duty 0.15 gives ~0.25 m/s
    // Scale proportionally with max_duty_ setting
    const float baseline_duty = 0.15f;
    const float baseline_velocity = 0.25f;
    float max_vel_at_max_duty = baseline_velocity * (max_duty_ / baseline_duty);

    float duty_left = (v_left / max_vel_at_max_duty) * max_duty_;
    float duty_right = (v_right / max_vel_at_max_duty) * max_duty_;

    // Ensure any non-zero velocity command produces duty above start threshold
    // This prevents a "dead zone" where small but intentional commands don't move
    const float min_start_duty = 0.028f;  // Slightly above cal default 0.025

    if (std::abs(v_left) > 0.005f && std::abs(duty_left) < min_start_duty) {
        duty_left = std::copysign(min_start_duty, v_left);
    }
    if (std::abs(v_right) > 0.005f && std::abs(duty_right) < min_start_duty) {
        duty_right = std::copysign(min_start_duty, v_right);
    }

    setWheelDuty(duty_left, duty_right);
}

void MotionController::setDuty(float linear, float angular) {
    if (!initialized_) return;

    // Track if this is straight-line motion (for hybrid control)
    // Straight line = angular is near zero
    straight_line_mode_ = (std::abs(angular) < 0.05f);

    // Track if we're turning (for higher min duty thresholds)
    // Turning = significant angular command relative to linear
    turning_mode_ = (std::abs(angular) > 0.15f) ||
                    (std::abs(linear) > 0.01f && std::abs(angular / linear) > 0.3f);

    // Differential drive: convert (linear, angular) to wheel duties
    float left = linear - angular;
    float right = linear + angular;

    // Scale to max_duty
    left *= max_duty_;
    right *= max_duty_;

    setWheelDuty(left, right);
}

void MotionController::setWheelDuty(float left, float right) {
    if (!initialized_) return;

    raw_mode_ = false;

    // Clamp to max duty
    left = std::clamp(left, -max_duty_, max_duty_);
    right = std::clamp(right, -max_duty_, max_duty_);

    // Determine direction for scaling
    bool is_forward = (left + right) >= 0;

    // Apply scaling to right wheel
    float scaled_left = left;
    float scaled_right = applyScaling(right, true);

    // Apply thresholds
    float thresh_left = applyThreshold(scaled_left, false, moving_left_);
    float thresh_right = applyThreshold(scaled_right, true, moving_right_);

    // Key decision: if EITHER wheel would stall, stop BOTH
    // This prevents unexpected rotation when one wheel stops
    bool left_would_stall = (std::abs(scaled_left) > 0.001f && thresh_left == 0);
    bool right_would_stall = (std::abs(scaled_right) > 0.001f && thresh_right == 0);

    if (left_would_stall || right_would_stall) {
        target_duty_left_ = 0;
        target_duty_right_ = 0;
    } else {
        target_duty_left_ = thresh_left;
        target_duty_right_ = thresh_right;
    }

    // Update mode
    if (target_duty_left_ == 0 && target_duty_right_ == 0) {
        mode_ = ControlMode::STOPPED;
    } else {
        mode_ = ControlMode::OPEN_LOOP;
    }
}

void MotionController::setWheelDutyRaw(float left, float right) {
    if (!initialized_) return;

    raw_mode_ = true;
    target_duty_left_ = std::clamp(left, -max_duty_, max_duty_);
    target_duty_right_ = std::clamp(right, -max_duty_, max_duty_);

    if (target_duty_left_ == 0 && target_duty_right_ == 0) {
        mode_ = ControlMode::STOPPED;
    } else {
        mode_ = ControlMode::OPEN_LOOP;
    }
}

void MotionController::stop(bool immediate) {
    target_duty_left_ = 0;
    target_duty_right_ = 0;
    mode_ = ControlMode::STOPPED;

    if (immediate && vesc_) {
        current_duty_left_ = 0;
        current_duty_right_ = 0;
        vesc_->stop();
    }
}

void MotionController::emergencyStop() {
    target_duty_left_ = 0;
    target_duty_right_ = 0;
    current_duty_left_ = 0;
    current_duty_right_ = 0;
    mode_ = ControlMode::STOPPED;

    if (vesc_) {
        vesc_->stop();
    }
}

// ============================================================================
// Update Loop
// ============================================================================

bool MotionController::update(float dt) {
    if (!initialized_ || !vesc_) return false;

    // Read wheel status first (needed for mode decisions)
    VescStatus status_left = vesc_->getStatus(cal_.vesc_id_left);
    VescStatus status_right = vesc_->getStatus(cal_.vesc_id_right);

    last_erpm_left_ = status_left.erpm;
    last_erpm_right_ = status_right.erpm;
    last_duty_left_ = status_left.duty;
    last_duty_right_ = status_right.duty;

    // Update moving state
    moving_left_ = std::abs(status_left.erpm) > erpm_moving_threshold_;
    moving_right_ = std::abs(status_right.erpm) > erpm_moving_threshold_;

    // Update control mode based on ERPM (with hysteresis)
    updateControlMode();

    // Ramp current duty toward target
    current_duty_left_ = rampToward(current_duty_left_, target_duty_left_, ramp_rate_, dt);
    current_duty_right_ = rampToward(current_duty_right_, target_duty_right_, ramp_rate_, dt);

    // Send commands based on control mode
    switch (mode_) {
        case ControlMode::STOPPED:
            // Ensure motors are stopped
            vesc_->setDutyRaw(0.0f, 0.0f);
            break;

        case ControlMode::OPEN_LOOP:
            // Use duty cycle control (with calibrated scaling already applied in setWheelDuty)
            vesc_->setDutyRaw(current_duty_left_, current_duty_right_);
            break;

        case ControlMode::CLOSED_LOOP:
            // Use RPM control - VESC PID handles motor asymmetry
            // CRITICAL: For straight-line motion, send SAME ERPM to both wheels
            // The VESC PID will compensate for motor differences automatically
            {
                int32_t erpm_left = dutyToErpm(current_duty_left_);
                if (current_duty_left_ < 0) erpm_left = -erpm_left;

                int32_t erpm_right;
                if (straight_line_mode_) {
                    // Straight line - SAME ERPM for both wheels (use left as reference)
                    // This is the key fix: ignore the scaled right duty, use left for both
                    erpm_right = erpm_left;
                } else {
                    // Turning - compute right ERPM from its duty
                    // The duty difference was set intentionally for differential steering
                    erpm_right = dutyToErpm(current_duty_right_);
                    if (current_duty_right_ < 0) erpm_right = -erpm_right;
                }

                vesc_->setRPM(erpm_left, erpm_right);
            }
            break;
    }

    // Update odometry
    updateOdometry(dt);

    return true;
}

// ============================================================================
// Feedback
// ============================================================================

Pose2D MotionController::getPose() const {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    return pose_;
}

Velocity2D MotionController::getVelocity() const {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    return velocity_;
}

bool MotionController::isMoving() const {
    return moving_left_ || moving_right_;
}

void MotionController::resetPose(const Pose2D& pose) {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    pose_ = pose;
    velocity_ = {};

    // Reset VescDriver odometry (zeros the tachometer baseline)
    if (vesc_) {
        vesc_->resetOdometry();
    }

    // Reset our tracking
    last_tach_left_ = 0;
    last_tach_right_ = 0;
    odom_initialized_ = false;  // Re-initialize on next update
}

// ============================================================================
// Configuration
// ============================================================================

void MotionController::setRampRate(float duty_per_sec) {
    ramp_rate_ = std::max(0.01f, duty_per_sec);
}

void MotionController::setMaxDuty(float max_duty) {
    max_duty_ = std::clamp(max_duty, 0.01f, 1.0f);
}

void MotionController::setMovingThreshold(int erpm) {
    erpm_moving_threshold_ = std::max(10, erpm);
}

// ============================================================================
// Internal Methods
// ============================================================================

float MotionController::applyScaling(float duty, bool is_right) {
    if (!is_right) {
        return duty;  // Left wheel is reference, no scaling
    }

    float abs_duty = std::abs(duty);
    int sign = (duty >= 0) ? 1 : -1;

    // Determine direction from overall command context
    // Use the sign of the duty being processed
    bool is_forward = (duty >= 0);

    const auto& table = is_forward ? cal_.forward_scaling : cal_.reverse_scaling;

    if (table.empty()) {
        return duty;  // No calibration, pass through
    }

    float scale = interpolateScale(abs_duty, table);
    return sign * abs_duty * scale;
}

float MotionController::applyThreshold(float duty, bool is_right, bool is_moving) {
    float abs_duty = std::abs(duty);
    int sign = (duty >= 0) ? 1 : -1;
    bool is_forward = (duty >= 0);

    // Select appropriate thresholds based on motion type
    MinDutyThresholds thresholds;
    if (turning_mode_ && cal_.turning_thresholds_calibrated) {
        // Use turning-specific thresholds (higher due to scrubbing friction)
        thresholds = is_forward ? cal_.turning_forward_min_duty : cal_.turning_reverse_min_duty;
    } else if (turning_mode_) {
        // Turning but not calibrated: use straight-line thresholds with extra margin
        const auto& base = is_forward ? cal_.forward_min_duty : cal_.reverse_min_duty;
        thresholds = base.withMargin(1.0f + cal_.min_duty_margin * 2.0f);  // Double margin for uncalibrated turning
    } else {
        // Straight-line motion
        thresholds = is_forward ? cal_.forward_min_duty : cal_.reverse_min_duty;
    }

    // Apply safety margin
    float margin_mult = 1.0f + cal_.min_duty_margin;
    float start_thresh = (is_right ? thresholds.start_right : thresholds.start_left) * margin_mult;
    float keep_thresh = (is_right ? thresholds.keep_right : thresholds.keep_left) * margin_mult;

    if (is_moving) {
        // Already moving - use keep threshold
        if (abs_duty < keep_thresh) {
            return 0;  // Below keep threshold, will stall
        }
    } else {
        // Starting from standstill - use start threshold
        if (abs_duty < start_thresh) {
            return 0;  // Below start threshold, won't overcome static friction
        }
    }

    return duty;  // Above threshold, use as-is
}

float MotionController::interpolateScale(float duty, const std::vector<DutyScalePoint>& table) {
    if (table.empty()) return 1.0f;

    // Below first point
    if (duty <= table.front().duty) {
        return table.front().scale_right;
    }

    // Above last point
    if (duty >= table.back().duty) {
        return table.back().scale_right;
    }

    // Linear interpolation between points
    for (size_t i = 0; i < table.size() - 1; i++) {
        if (duty >= table[i].duty && duty <= table[i + 1].duty) {
            float t = (duty - table[i].duty) / (table[i + 1].duty - table[i].duty);
            return table[i].scale_right + t * (table[i + 1].scale_right - table[i].scale_right);
        }
    }

    return 1.0f;  // Fallback
}

void MotionController::updateOdometry(float dt) {
    if (!vesc_ || !vesc_->isConnected()) return;

    VescOdometry odom = vesc_->getOdometry();

    if (!odom_initialized_) {
        last_tach_left_ = odom.tach_left;
        last_tach_right_ = odom.tach_right;
        odom_initialized_ = true;
        return;
    }

    // Calculate distance traveled by each wheel
    int32_t delta_left = odom.tach_left - last_tach_left_;
    int32_t delta_right = odom.tach_right - last_tach_right_;

    float dist_left = delta_left / cal_.ticks_per_meter;
    float dist_right = delta_right / cal_.ticks_per_meter;

    // Differential drive odometry
    float dist_center = (dist_left + dist_right) / 2.0f;
    float delta_theta = (dist_right - dist_left) / cal_.effective_track_m;

    {
        std::lock_guard<std::mutex> lock(odom_mutex_);

        // Update pose (using midpoint approximation)
        float avg_theta = pose_.theta + delta_theta / 2.0f;
        pose_.x += dist_center * std::cos(avg_theta);
        pose_.y += dist_center * std::sin(avg_theta);
        pose_.theta += delta_theta;

        // Normalize theta to [-pi, pi]
        constexpr float PI_F = 3.14159265358979323846f;
        constexpr float TWO_PI_F = 2.0f * PI_F;
        while (pose_.theta > PI_F) pose_.theta -= TWO_PI_F;
        while (pose_.theta < -PI_F) pose_.theta += TWO_PI_F;

        // Update velocity (if dt > 0)
        if (dt > 0.001f) {
            velocity_.linear = dist_center / dt;
            velocity_.angular = delta_theta / dt;
        }
    }

    last_tach_left_ = odom.tach_left;
    last_tach_right_ = odom.tach_right;
}

float MotionController::rampToward(float current, float target, float rate, float dt) {
    float max_change = rate * dt;
    float diff = target - current;

    if (std::abs(diff) <= max_change) {
        return target;  // Close enough, snap to target
    }

    return current + std::copysign(max_change, diff);
}

int32_t MotionController::dutyToErpm(float duty) {
    // Convert absolute duty value to expected ERPM using calibration lookup
    float abs_duty = std::abs(duty);
    const auto& table = cal_.duty_to_erpm;

    if (table.empty()) {
        // Fallback: linear approximation (roughly 66000 * duty)
        return static_cast<int32_t>(abs_duty * 66000.0f);
    }

    // Below first point
    if (abs_duty <= table.front().duty) {
        // Linear extrapolation from origin
        float ratio = table.front().erpm / table.front().duty;
        return static_cast<int32_t>(abs_duty * ratio);
    }

    // Above last point
    if (abs_duty >= table.back().duty) {
        // Linear extrapolation from last two points
        if (table.size() >= 2) {
            const auto& p1 = table[table.size() - 2];
            const auto& p2 = table.back();
            float slope = (p2.erpm - p1.erpm) / (p2.duty - p1.duty);
            return static_cast<int32_t>(p2.erpm + slope * (abs_duty - p2.duty));
        }
        return table.back().erpm;
    }

    // Linear interpolation between points
    for (size_t i = 0; i < table.size() - 1; i++) {
        if (abs_duty >= table[i].duty && abs_duty <= table[i + 1].duty) {
            float t = (abs_duty - table[i].duty) / (table[i + 1].duty - table[i].duty);
            float erpm = table[i].erpm + t * (table[i + 1].erpm - table[i].erpm);
            return static_cast<int32_t>(erpm);
        }
    }

    return 0;  // Fallback
}

void MotionController::updateControlMode() {
    // Get current ERPM from both wheels
    int32_t erpm_left = std::abs(last_erpm_left_);
    int32_t erpm_right = std::abs(last_erpm_right_);

    switch (mode_) {
        case ControlMode::STOPPED:
            // Stay stopped until we have non-zero target
            if (target_duty_left_ != 0 || target_duty_right_ != 0) {
                mode_ = ControlMode::OPEN_LOOP;
            }
            break;

        case ControlMode::OPEN_LOOP:
            // Switch to closed-loop when BOTH wheels are above threshold
            if (erpm_left >= ERPM_THRESHOLD_UP && erpm_right >= ERPM_THRESHOLD_UP) {
                mode_ = ControlMode::CLOSED_LOOP;
                // Calculate target ERPM from current duty commands
                target_erpm_left_ = dutyToErpm(current_duty_left_);
                if (current_duty_left_ < 0) target_erpm_left_ = -target_erpm_left_;
                target_erpm_right_ = dutyToErpm(current_duty_right_);
                if (current_duty_right_ < 0) target_erpm_right_ = -target_erpm_right_;
            }
            // Check for stop
            if (target_duty_left_ == 0 && target_duty_right_ == 0 &&
                current_duty_left_ == 0 && current_duty_right_ == 0) {
                mode_ = ControlMode::STOPPED;
            }
            break;

        case ControlMode::CLOSED_LOOP:
            // Switch back to open-loop if EITHER wheel drops below threshold (hysteresis)
            if (erpm_left < ERPM_THRESHOLD_DOWN || erpm_right < ERPM_THRESHOLD_DOWN) {
                mode_ = ControlMode::OPEN_LOOP;
                // Keep current duty values for smooth transition
            }
            // Check for stop
            if (target_duty_left_ == 0 && target_duty_right_ == 0) {
                mode_ = ControlMode::OPEN_LOOP;  // Use duty ramping to stop smoothly
            }
            break;
    }
}

} // namespace slam
