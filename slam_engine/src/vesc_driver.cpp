/**
 * VESC CAN Bus Driver Implementation
 *
 * Differential/skid-steer drive control with:
 * - Open-loop duty cycle control
 * - Per-wheel scaling calibration
 * - Minimum duty thresholds
 * - Tachometer-based odometry
 */

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "slam/vesc_driver.hpp"
#include "slam/vesc_can_interface.hpp"
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>

namespace slam {

// ============================================================================
// WheelCalibration
// ============================================================================

float WheelCalibration::getScaleFactor(float duty) const {
    if (duty_scale_table.empty()) {
        return 1.0f;
    }

    // Find interpolation points
    if (duty <= duty_scale_table.front().first) {
        return duty_scale_table.front().second;
    }
    if (duty >= duty_scale_table.back().first) {
        return duty_scale_table.back().second;
    }

    // Linear interpolation
    for (size_t i = 0; i < duty_scale_table.size() - 1; i++) {
        if (duty >= duty_scale_table[i].first && duty <= duty_scale_table[i + 1].first) {
            float t = (duty - duty_scale_table[i].first) /
                      (duty_scale_table[i + 1].first - duty_scale_table[i].first);
            return duty_scale_table[i].second + t *
                   (duty_scale_table[i + 1].second - duty_scale_table[i].second);
        }
    }

    return 1.0f;
}

// ============================================================================
// VescDriver
// ============================================================================

VescDriver::VescDriver() {
    can_ = std::make_unique<VescCanInterface>();

    // Initialize with default calibration (from January 2026 testing)
    cal_left_.vesc_id = 1;
    cal_left_.ticks_per_meter = 14052.0f;
    cal_left_.min_duty_start = 0.040f;
    cal_left_.min_duty_keep = 0.030f;
    cal_left_.duty_scale_table = {
        {0.030f, 1.0f},  // Left is reference, scale = 1.0
        {0.035f, 1.0f},
        {0.040f, 1.0f},
        {0.050f, 1.0f},
        {0.060f, 1.0f},
        {0.070f, 1.0f},
    };

    cal_right_.vesc_id = 126;
    cal_right_.ticks_per_meter = 14133.0f;
    cal_right_.min_duty_start = 0.035f;
    cal_right_.min_duty_keep = 0.020f;
    cal_right_.duty_scale_table = {
        {0.030f, 0.622f},  // Right needs scaling to match left
        {0.035f, 0.800f},
        {0.040f, 0.816f},
        {0.050f, 0.867f},
        {0.060f, 0.878f},
        {0.070f, 0.883f},
    };
}

VescDriver::~VescDriver() {
    shutdown();
}

bool VescDriver::init(const std::string& port, uint8_t vesc_left, uint8_t vesc_right) {
    vesc_left_ = vesc_left;
    vesc_right_ = vesc_right;
    cal_left_.vesc_id = vesc_left;
    cal_right_.vesc_id = vesc_right;

    if (!can_->open(port, 500000)) {
        return false;
    }

    // Set up CAN receive callback
    can_->setReceiveCallback([this](const CanMessage& msg) {
        processCanMessage(msg.id, msg.data, msg.dlc);
    });

    connected_ = true;
    return true;
}

void VescDriver::shutdown() {
    if (!connected_) return;

    stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    can_->close();
    connected_ = false;
}

// ============================================================================
// Motor Control
// ============================================================================

void VescDriver::sendDutyCommand(uint8_t vesc_id, float duty) {
    uint32_t can_id = vesc::makeCanId(vesc::Command::SET_DUTY, vesc_id);
    uint8_t data[4];
    vesc::encodeDuty(duty, data);
    can_->send(can_id, data, 4, true);
}

void VescDriver::setDuty(float duty, bool apply_scaling) {
    if (!connected_) return;

    float duty_left = duty;
    float duty_right = duty;

    if (apply_scaling && duty > 0) {
        // Apply interpolated scaling to right wheel
        float scale = cal_right_.getScaleFactor(duty);
        duty_right = duty * scale;
    }

    applyThresholds(duty_left, duty_right);

    sendDutyCommand(vesc_left_, duty_left);
    sendDutyCommand(vesc_right_, duty_right);

    current_duty_left_ = duty_left;
    current_duty_right_ = duty_right;
}

void VescDriver::setDutyRaw(float duty_left, float duty_right) {
    if (!connected_) return;

    sendDutyCommand(vesc_left_, duty_left);
    sendDutyCommand(vesc_right_, duty_right);

    current_duty_left_ = duty_left;
    current_duty_right_ = duty_right;
    is_moving_ = (duty_left != 0.0f || duty_right != 0.0f);
}

void VescDriver::sendRPMCommand(uint8_t vesc_id, int32_t erpm) {
    uint32_t can_id = vesc::makeCanId(vesc::Command::SET_RPM, vesc_id);
    uint8_t data[4];
    vesc::encodeRpm(erpm, data);
    can_->send(can_id, data, 4, true);
}

void VescDriver::setRPM(int32_t erpm_left, int32_t erpm_right) {
    if (!connected_) return;

    sendRPMCommand(vesc_left_, erpm_left);
    sendRPMCommand(vesc_right_, erpm_right);

    is_moving_ = (erpm_left != 0 || erpm_right != 0);
}

void VescDriver::setDutyDifferential(float linear, float angular) {
    if (!connected_) return;

    // Mix linear and angular components
    float duty_left = linear + angular;
    float duty_right = linear - angular;

    // Normalize if exceeding limits
    float max_duty = std::max(std::abs(duty_left), std::abs(duty_right));
    if (max_duty > 1.0f) {
        duty_left /= max_duty;
        duty_right /= max_duty;
    }

    // Apply scaling for forward motion
    if (linear > 0 && std::abs(angular) < 0.1f) {
        float scale = cal_right_.getScaleFactor(std::abs(duty_right));
        duty_right *= scale;
    }

    applyThresholds(duty_left, duty_right);

    sendDutyCommand(vesc_left_, duty_left);
    sendDutyCommand(vesc_right_, duty_right);

    current_duty_left_ = duty_left;
    current_duty_right_ = duty_right;
}

void VescDriver::stop() {
    setDutyRaw(0.0f, 0.0f);
    is_moving_ = false;
}

void VescDriver::rampTo(float target_duty, float duration_s, bool apply_scaling) {
    if (!connected_) return;

    // Estimate current "nominal" duty
    float start_duty = std::max(current_duty_left_, current_duty_right_);
    if (apply_scaling && current_duty_right_ > 0) {
        float scale = cal_right_.getScaleFactor(start_duty);
        if (scale > 0) {
            start_duty = current_duty_right_ / scale;
        }
    }

    int steps = static_cast<int>(duration_s * 50);  // 50 Hz
    if (steps < 1) steps = 1;

    for (int i = 0; i < steps; i++) {
        float t = static_cast<float>(i + 1) / steps;
        float duty = start_duty + (target_duty - start_duty) * t;
        setDuty(duty, apply_scaling);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

void VescDriver::applyThresholds(float& duty_left, float& duty_right) {
    if (duty_left > 0 || duty_right > 0) {
        float min_left, min_right;

        if (is_moving_) {
            // Already moving - use lower "keep" thresholds
            min_left = cal_left_.min_duty_keep;
            min_right = cal_right_.min_duty_keep;
        } else {
            // Starting from standstill - use higher "start" thresholds
            min_left = cal_left_.min_duty_start;
            min_right = cal_right_.min_duty_start;
        }

        // If EITHER wheel is below its threshold, stop BOTH
        if (duty_left < min_left || duty_right < min_right) {
            duty_left = 0.0f;
            duty_right = 0.0f;
            is_moving_ = false;
        } else {
            is_moving_ = true;
        }
    } else {
        duty_left = 0.0f;
        duty_right = 0.0f;
        is_moving_ = false;
    }
}

// ============================================================================
// Odometry
// ============================================================================

VescOdometry VescDriver::getOdometry() const {
    std::lock_guard<std::mutex> lock(odom_mutex_);

    VescOdometry odom;
    odom.tach_left = status_left_.tachometer - tach_offset_left_;
    odom.tach_right = status_right_.tachometer - tach_offset_right_;
    odom.distance_left_m = odom.tach_left / cal_left_.ticks_per_meter;
    odom.distance_right_m = odom.tach_right / cal_right_.ticks_per_meter;

    // Calculate velocity from ERPM
    // ERPM = electrical RPM, mechanical RPM = ERPM / pole_pairs
    // For VESC motors, typically 7 pole pairs
    // velocity = RPM * circumference / 60
    // But we'll derive from tachometer delta instead
    if (last_odom_time_us_ > 0) {
        auto now = std::chrono::steady_clock::now();
        uint64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();
        float dt = (now_us - last_odom_time_us_) / 1000000.0f;

        if (dt > 0) {
            int32_t delta_left = status_left_.tachometer - last_tach_left_;
            int32_t delta_right = status_right_.tachometer - last_tach_right_;
            odom.velocity_left_mps = (delta_left / cal_left_.ticks_per_meter) / dt;
            odom.velocity_right_mps = (delta_right / cal_right_.ticks_per_meter) / dt;
        }
    }

    odom.timestamp_us = status_left_.last_update_us;
    return odom;
}

void VescDriver::resetOdometry() {
    std::lock_guard<std::mutex> lock(odom_mutex_);

    // If we've received STATUS_5 from both VESCs, capture current tach as offset
    if (has_tach_left_ && has_tach_right_) {
        tach_offset_left_ = status_left_.tachometer;
        tach_offset_right_ = status_right_.tachometer;
        odom_reset_pending_ = false;
    } else {
        // STATUS_5 hasn't arrived yet - defer reset until it does
        odom_reset_pending_ = true;
    }
}

void VescDriver::setOdometryCallback(OdometryCallback callback) {
    odom_callback_ = callback;
}

// ============================================================================
// Status
// ============================================================================

VescStatus VescDriver::getStatus(uint8_t vesc_id) const {
    std::lock_guard<std::mutex> lock(status_mutex_);
    if (vesc_id == vesc_left_) {
        return status_left_;
    } else if (vesc_id == vesc_right_) {
        return status_right_;
    }
    return VescStatus{};
}

void VescDriver::setStatusCallback(StatusCallback callback) {
    status_callback_ = callback;
}

void VescDriver::processCanMessage(uint32_t can_id, const uint8_t* data, uint8_t len) {
    vesc::Command cmd;
    uint8_t vesc_id;
    vesc::parseCanId(can_id, cmd, vesc_id);

    VescStatus* status = nullptr;
    {
        std::lock_guard<std::mutex> lock(status_mutex_);
        if (vesc_id == vesc_left_) {
            status = &status_left_;
        } else if (vesc_id == vesc_right_) {
            status = &status_right_;
        }
    }

    if (!status) return;

    auto now = std::chrono::steady_clock::now();
    uint64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();

    {
        std::lock_guard<std::mutex> lock(status_mutex_);

        switch (cmd) {
            case vesc::Command::STATUS:
                if (len >= 8) {
                    vesc::decodeStatus(data, status->erpm, status->current, status->duty);
                }
                break;

            case vesc::Command::STATUS_4:
                if (len >= 8) {
                    float pid_pos;
                    vesc::decodeStatus4(data, status->temp_fet, status->temp_motor,
                                       status->current, pid_pos);
                }
                break;

            case vesc::Command::STATUS_5:
                if (len >= 6) {
                    vesc::decodeStatus5(data, status->tachometer, status->voltage);
                    // Mark that we've received tachometer data for this VESC
                    if (vesc_id == vesc_left_) {
                        has_tach_left_ = true;
                    } else if (vesc_id == vesc_right_) {
                        has_tach_right_ = true;
                    }
                }
                break;

            default:
                break;
        }

        status->last_update_us = now_us;
    }

    // Update odometry tracking
    {
        std::lock_guard<std::mutex> lock(odom_mutex_);

        // If reset was pending and we now have tach from both VESCs, apply it
        if (odom_reset_pending_ && has_tach_left_ && has_tach_right_) {
            tach_offset_left_ = status_left_.tachometer;
            tach_offset_right_ = status_right_.tachometer;
            odom_reset_pending_ = false;
        }

        last_tach_left_ = status_left_.tachometer;
        last_tach_right_ = status_right_.tachometer;
        last_odom_time_us_ = now_us;
    }

    // Notify callbacks
    if (status_callback_) {
        status_callback_(vesc_id, *status);
    }

    if (odom_callback_) {
        odom_callback_(getOdometry());
    }
}

// ============================================================================
// Calibration I/O
// ============================================================================

bool VescDriver::loadCalibration(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    std::string section;

    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        // Check for section headers
        if (line[0] == '[') {
            size_t end = line.find(']');
            if (end != std::string::npos) {
                section = line.substr(1, end - 1);
            }
            continue;
        }

        // Parse key=value
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string value = line.substr(eq + 1);

        // Remove whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        WheelCalibration* cal = nullptr;
        if (section == "left") {
            cal = &cal_left_;
        } else if (section == "right") {
            cal = &cal_right_;
        } else if (section == "geometry") {
            if (key == "track_width_mm") geometry_.track_width_mm = std::stof(value);
            else if (key == "wheel_base_mm") geometry_.wheel_base_mm = std::stof(value);
            else if (key == "tread_width_mm") geometry_.tread_width_mm = std::stof(value);
            else if (key == "effective_track_mm") geometry_.effective_track_mm = std::stof(value);
            continue;
        }

        if (cal) {
            if (key == "vesc_id") cal->vesc_id = std::stoi(value);
            else if (key == "ticks_per_meter") cal->ticks_per_meter = std::stof(value);
            else if (key == "min_duty_start") cal->min_duty_start = std::stof(value);
            else if (key == "min_duty_keep") cal->min_duty_keep = std::stof(value);
            else if (key == "duty_scale") {
                // Parse comma-separated pairs: duty1:scale1,duty2:scale2,...
                cal->duty_scale_table.clear();
                std::stringstream ss(value);
                std::string pair;
                while (std::getline(ss, pair, ',')) {
                    size_t colon = pair.find(':');
                    if (colon != std::string::npos) {
                        float duty = std::stof(pair.substr(0, colon));
                        float scale = std::stof(pair.substr(colon + 1));
                        cal->duty_scale_table.push_back({duty, scale});
                    }
                }
            }
        }
    }

    vesc_left_ = cal_left_.vesc_id;
    vesc_right_ = cal_right_.vesc_id;

    return true;
}

bool VescDriver::saveCalibration(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    file << "# VESC Calibration Data\n\n";

    file << "[geometry]\n";
    file << "track_width_mm=" << geometry_.track_width_mm << "\n";
    file << "wheel_base_mm=" << geometry_.wheel_base_mm << "\n";
    file << "tread_width_mm=" << geometry_.tread_width_mm << "\n";
    file << "effective_track_mm=" << geometry_.effective_track_mm << "\n";
    file << "\n";

    auto writeWheel = [&file](const std::string& name, const WheelCalibration& cal) {
        file << "[" << name << "]\n";
        file << "vesc_id=" << static_cast<int>(cal.vesc_id) << "\n";
        file << "ticks_per_meter=" << cal.ticks_per_meter << "\n";
        file << "min_duty_start=" << cal.min_duty_start << "\n";
        file << "min_duty_keep=" << cal.min_duty_keep << "\n";

        file << "duty_scale=";
        for (size_t i = 0; i < cal.duty_scale_table.size(); i++) {
            if (i > 0) file << ",";
            file << cal.duty_scale_table[i].first << ":"
                 << cal.duty_scale_table[i].second;
        }
        file << "\n\n";
    };

    writeWheel("left", cal_left_);
    writeWheel("right", cal_right_);

    return true;
}

void VescDriver::applyCalibration(const CalibrationResult& cal) {
    cal_left_ = cal.left;
    cal_right_ = cal.right;
    geometry_ = cal.geometry;
    vesc_left_ = cal.left.vesc_id;
    vesc_right_ = cal.right.vesc_id;
}

CalibrationResult VescDriver::getCalibration() const {
    CalibrationResult result;
    result.left = cal_left_;
    result.right = cal_right_;
    result.geometry = geometry_;
    result.valid = true;
    return result;
}

// ============================================================================
// Calibration Procedures
// ============================================================================

CalibrationResult VescDriver::runDutyCalibration(
    std::function<void(int, const std::string&)> progress_callback) {

    CalibrationResult result;
    result.left = cal_left_;
    result.right = cal_right_;
    result.geometry = geometry_;

    if (!connected_) {
        result.error_message = "Not connected";
        return result;
    }

    auto report = [&](int pct, const std::string& msg) {
        if (progress_callback) progress_callback(pct, msg);
    };

    std::vector<float> duty_points = {0.025f, 0.030f, 0.035f, 0.040f, 0.050f, 0.060f, 0.070f};
    std::vector<std::pair<float, float>> left_table;
    std::vector<std::pair<float, float>> right_table;

    report(0, "Starting duty calibration...");

    // Ramp up to first point
    for (int d = 0; d <= static_cast<int>(duty_points[0] * 1000); d += 2) {
        setDutyRaw(d / 1000.0f, d / 1000.0f);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    for (size_t i = 0; i < duty_points.size(); i++) {
        float duty = duty_points[i];
        int progress = static_cast<int>((i + 1) * 100 / duty_points.size());
        report(progress, "Testing duty " + std::to_string(duty));

        // Ramp to this duty
        float current = current_duty_left_;
        int steps = std::max(1, static_cast<int>(std::abs(duty - current) * 500));
        for (int j = 0; j < steps; j++) {
            float d = current + (duty - current) * (j + 1) / steps;
            setDutyRaw(d, d);
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        }

        // Hold and let stabilize
        for (int j = 0; j < 30; j++) {
            setDutyRaw(duty, duty);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        // Collect ERPM samples
        std::vector<int32_t> erpm_left, erpm_right;
        auto start = std::chrono::steady_clock::now();

        while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(800)) {
            setDutyRaw(duty, duty);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));

            VescStatus sl = getStatus(vesc_left_);
            VescStatus sr = getStatus(vesc_right_);

            if (sl.erpm != 0) erpm_left.push_back(sl.erpm);
            if (sr.erpm != 0) erpm_right.push_back(sr.erpm);
        }

        if (!erpm_left.empty() && !erpm_right.empty()) {
            float avg_left = 0, avg_right = 0;
            for (auto e : erpm_left) avg_left += e;
            for (auto e : erpm_right) avg_right += e;
            avg_left /= erpm_left.size();
            avg_right /= erpm_right.size();

            if (avg_left > 100) {
                // Left is reference (scale = 1.0)
                left_table.push_back({duty, 1.0f});

                // Right gets scaled to match left
                float scale = (avg_right > 0) ? (avg_left / avg_right) : 1.0f;
                right_table.push_back({duty, scale});
            }
        }
    }

    // Ramp down
    for (int d = static_cast<int>(duty_points.back() * 1000); d >= 0; d -= 2) {
        setDutyRaw(d / 1000.0f, d / 1000.0f);
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    stop();

    // Update calibration
    if (!left_table.empty()) {
        result.left.duty_scale_table = left_table;
        result.right.duty_scale_table = right_table;
        result.valid = true;
    }

    report(100, "Duty calibration complete");
    return result;
}

CalibrationResult VescDriver::runMinDutyCalibration(
    std::function<void(int, const std::string&)> progress_callback) {

    CalibrationResult result;
    result.left = cal_left_;
    result.right = cal_right_;
    result.geometry = geometry_;

    if (!connected_) {
        result.error_message = "Not connected";
        return result;
    }

    auto report = [&](int pct, const std::string& msg) {
        if (progress_callback) progress_callback(pct, msg);
    };

    report(0, "Finding minimum starting duty...");

    // Find minimum starting duty for each wheel
    for (float duty = 0.020f; duty <= 0.060f; duty += 0.002f) {
        // Test left
        setDutyRaw(duty, 0.0f);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        VescStatus sl = getStatus(vesc_left_);
        if (std::abs(sl.erpm) > 50 && result.left.min_duty_start == 0.040f) {
            result.left.min_duty_start = duty;
        }

        // Test right
        setDutyRaw(0.0f, duty);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        VescStatus sr = getStatus(vesc_right_);
        if (std::abs(sr.erpm) > 50 && result.right.min_duty_start == 0.035f) {
            result.right.min_duty_start = duty;
        }

        stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        int progress = static_cast<int>((duty - 0.020f) / 0.040f * 50);
        report(progress, "Testing duty " + std::to_string(duty));
    }

    report(50, "Finding minimum keeping duty...");

    // Find minimum keeping duty (start moving, then reduce)
    for (int wheel = 0; wheel < 2; wheel++) {
        // Get wheel moving
        if (wheel == 0) {
            setDutyRaw(0.050f, 0.0f);
        } else {
            setDutyRaw(0.0f, 0.050f);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Reduce duty until it stops
        for (float duty = 0.040f; duty >= 0.010f; duty -= 0.002f) {
            if (wheel == 0) {
                setDutyRaw(duty, 0.0f);
            } else {
                setDutyRaw(0.0f, duty);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));

            VescStatus status = getStatus(wheel == 0 ? vesc_left_ : vesc_right_);
            if (std::abs(status.erpm) < 50) {
                // Stopped - the previous duty was the minimum
                if (wheel == 0) {
                    result.left.min_duty_keep = duty + 0.002f;
                } else {
                    result.right.min_duty_keep = duty + 0.002f;
                }
                break;
            }
        }

        stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        report(50 + (wheel + 1) * 25, wheel == 0 ? "Left wheel done" : "Right wheel done");
    }

    result.valid = true;
    report(100, "Minimum duty calibration complete");
    return result;
}

CalibrationResult VescDriver::runRotationCalibration(
    std::function<float()> imu_yaw_callback,
    std::function<void(int, const std::string&)> progress_callback) {

    CalibrationResult result;
    result.left = cal_left_;
    result.right = cal_right_;
    result.geometry = geometry_;

    if (!connected_) {
        result.error_message = "Not connected";
        return result;
    }

    if (!imu_yaw_callback) {
        result.error_message = "IMU callback required";
        return result;
    }

    auto report = [&](int pct, const std::string& msg) {
        if (progress_callback) progress_callback(pct, msg);
    };

    report(0, "Starting rotation calibration...");

    // Reset odometry
    resetOdometry();
    float start_yaw = imu_yaw_callback();
    int32_t start_tach_l = status_left_.tachometer;
    int32_t start_tach_r = status_right_.tachometer;

    // Rotate 360 degrees (left wheel forward, right wheel backward)
    float target_rotation = 2.0f * 3.14159f;  // 360 degrees
    float current_rotation = 0.0f;

    while (current_rotation < target_rotation) {
        // Turn left (left forward, right backward)
        float duty = 0.050f;
        setDutyRaw(duty, -duty * cal_right_.getScaleFactor(duty));

        std::this_thread::sleep_for(std::chrono::milliseconds(20));

        float current_yaw = imu_yaw_callback();
        current_rotation = std::abs(current_yaw - start_yaw);

        // Handle wrap-around
        if (current_rotation > 3.14159f) {
            current_rotation = 2.0f * 3.14159f - current_rotation;
        }

        int progress = static_cast<int>(current_rotation / target_rotation * 90);
        report(progress, "Rotating...");
    }

    stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Get final readings
    float end_yaw = imu_yaw_callback();
    int32_t end_tach_l = status_left_.tachometer;
    int32_t end_tach_r = status_right_.tachometer;

    float total_rotation = std::abs(end_yaw - start_yaw);
    int32_t ticks_left = std::abs(end_tach_l - start_tach_l);
    int32_t ticks_right = std::abs(end_tach_r - start_tach_r);

    // Calculate ticks per radian
    result.ticks_per_radian = (ticks_left + ticks_right) / 2.0f / total_rotation;

    // Calculate effective track width
    // For skid-steer: rotation = (ticks_left + ticks_right) / 2 / ticks_per_meter / (track/2)
    // So: track = 2 * (ticks_left + ticks_right) / 2 / ticks_per_meter / rotation
    float avg_ticks_per_meter = (cal_left_.ticks_per_meter + cal_right_.ticks_per_meter) / 2.0f;
    float arc_length = (ticks_left + ticks_right) / 2.0f / avg_ticks_per_meter;
    result.effective_track_mm = arc_length / total_rotation * 1000.0f * 2.0f;
    result.geometry.effective_track_mm = result.effective_track_mm;

    result.valid = true;
    report(100, "Rotation calibration complete");
    return result;
}

CalibrationResult VescDriver::runCalibrationWithLocalization(
    std::function<float()> imu_yaw_callback,
    std::function<void(const std::vector<std::array<float, 3>>&)> point_cloud_callback,
    std::function<void(int, const std::string&)> progress_callback) {

    CalibrationResult result;
    result.left = cal_left_;
    result.right = cal_right_;
    result.geometry = geometry_;

    if (!connected_) {
        result.error_message = "Not connected";
        return result;
    }

    if (!imu_yaw_callback) {
        result.error_message = "IMU callback required";
        return result;
    }

    auto report = [&](int pct, const std::string& msg) {
        if (progress_callback) progress_callback(pct, msg);
    };

    // Phase 1: Forward drive with duty calibration
    report(0, "Phase 1: Forward calibration drive");

    std::vector<float> duty_points = {0.035f, 0.040f, 0.050f, 0.060f};
    std::vector<std::pair<float, float>> right_scale_table;

    for (size_t i = 0; i < duty_points.size(); i++) {
        float duty = duty_points[i];
        report(static_cast<int>(i * 100 / 4 / 4), "Testing duty " + std::to_string(duty));

        // Ramp to duty
        rampTo(duty, 0.5f, false);  // No scaling - raw duty

        // Collect ERPM samples
        std::vector<int32_t> erpm_left, erpm_right;
        auto start = std::chrono::steady_clock::now();

        while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(600)) {
            setDutyRaw(duty, duty);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));

            VescStatus sl = getStatus(vesc_left_);
            VescStatus sr = getStatus(vesc_right_);
            if (sl.erpm != 0) erpm_left.push_back(sl.erpm);
            if (sr.erpm != 0) erpm_right.push_back(sr.erpm);
        }

        if (!erpm_left.empty() && !erpm_right.empty()) {
            float avg_l = 0, avg_r = 0;
            for (auto e : erpm_left) avg_l += e;
            for (auto e : erpm_right) avg_r += e;
            avg_l /= erpm_left.size();
            avg_r /= erpm_right.size();

            if (avg_l > 100) {
                float scale = (avg_r > 0) ? (avg_l / avg_r) : 1.0f;
                right_scale_table.push_back({duty, scale});
            }
        }
    }

    // Ramp down
    rampTo(0.0f, 1.0f, false);
    stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Update right wheel calibration
    if (!right_scale_table.empty()) {
        result.right.duty_scale_table = right_scale_table;
        cal_right_.duty_scale_table = right_scale_table;  // Apply immediately
    }

    // Phase 2: Turn 90 degrees left
    report(25, "Phase 2: Turn 90 degrees left");

    float start_yaw = imu_yaw_callback();
    float target_yaw = start_yaw + 1.5708f;  // 90 degrees

    while (true) {
        float current_yaw = imu_yaw_callback();
        float delta = current_yaw - start_yaw;

        // Normalize
        while (delta > 3.14159f) delta -= 2.0f * 3.14159f;
        while (delta < -3.14159f) delta += 2.0f * 3.14159f;

        if (delta >= 1.5708f) break;

        float duty = 0.045f;
        setDutyRaw(duty, -duty * cal_right_.getScaleFactor(duty));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Phase 3: Return to center
    report(50, "Phase 3: Return to center");

    while (true) {
        float current_yaw = imu_yaw_callback();
        float delta = current_yaw - start_yaw;

        while (delta > 3.14159f) delta -= 2.0f * 3.14159f;
        while (delta < -3.14159f) delta += 2.0f * 3.14159f;

        if (std::abs(delta) < 0.05f) break;  // Within ~3 degrees

        float duty = 0.045f;
        if (delta > 0) {
            // Turn right
            setDutyRaw(-duty, duty * cal_right_.getScaleFactor(duty));
        } else {
            // Turn left
            setDutyRaw(duty, -duty * cal_right_.getScaleFactor(duty));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Phase 4: Turn 90 degrees right
    report(75, "Phase 4: Turn 90 degrees right");

    while (true) {
        float current_yaw = imu_yaw_callback();
        float delta = current_yaw - start_yaw;

        while (delta > 3.14159f) delta -= 2.0f * 3.14159f;
        while (delta < -3.14159f) delta += 2.0f * 3.14159f;

        if (delta <= -1.5708f) break;

        float duty = 0.045f;
        setDutyRaw(-duty, duty * cal_right_.getScaleFactor(duty));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Provide point cloud for global localization
    // The SLAM engine should have been accumulating points during this sequence
    if (point_cloud_callback) {
        // Note: The actual point cloud would come from the SLAM engine
        // This is a placeholder - the integration will be done at a higher level
        std::vector<std::array<float, 3>> captured_points;
        point_cloud_callback(captured_points);
    }

    result.valid = true;
    report(100, "Calibration with localization complete");
    return result;
}

} // namespace slam
