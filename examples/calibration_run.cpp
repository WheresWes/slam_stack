/**
 * Combined Calibration + Localization Run
 *
 * Single application that:
 * 1. Calibrates motor duty scaling (forward drive)
 * 2. Calibrates rotation (90° turns using IMU yaw from SLAM)
 * 3. Builds local point cloud map for global localization
 *
 * Run this BEFORE any job to calibrate the robot.
 *
 * Movement pattern:
 *   Start -> Forward (duty cal) -> Turn 90° Left -> Center -> Turn 90° Right
 *
 * Outputs:
 *   - vesc_calibration.ini (motor calibration)
 *   - local_map.ply (point cloud for global localization)
 */

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <cmath>

#include "slam/slam_engine.hpp"
#include "slam/livox_mid360.hpp"
#include "slam/vesc_driver.hpp"

// Note: Visualization removed - using native LivoxMid360 driver without Rerun

using namespace slam;

// ============================================================================
// Configuration
// ============================================================================

struct CalibrationConfig {
    // LiDAR
    std::string host_ip = "192.168.1.50";
    std::string device_ip = "192.168.1.144";

    // CAN/VESC
    std::string can_port = "COM3";
    uint8_t vesc_left = 1;
    uint8_t vesc_right = 126;

    // Fast duty calibration
    std::vector<float> duty_points = {0.035f, 0.045f, 0.055f, 0.065f};
    float duty_hold_time = 0.15f;   // seconds (reduced from 0.8 → 0.2 → 0.15)
    float max_duty = 0.065f;        // Max test duty
    int erpm_threshold = 100;       // ERPM to consider wheel "moving"

    // Fast turn calibration
    float turn_duty = 0.12f;        // Fast turning (0.045 → 0.08 → 0.10 → 0.12)
    float target_angle_deg = 90.0f;

    // Output
    std::string calibration_file = "vesc_calibration.ini";
    std::string map_file = "local_map.ply";
    bool visualize = true;
};

// ============================================================================
// Calibration Results
// ============================================================================

struct DutyCalPoint {
    float duty;
    float erpm_left;
    float erpm_right;
    float scale_right;
};

struct TurnCalResult {
    float angle_rad;
    int32_t ticks_left;
    int32_t ticks_right;
    float ticks_per_radian;
};

struct MinDutyThresholds {
    float start_left = 0.0f;    // Duty to overcome static friction (from standstill)
    float start_right = 0.0f;
    float keep_left = 0.0f;     // Duty to maintain motion (kinetic friction)
    float keep_right = 0.0f;
};

struct CalibrationResults {
    std::vector<DutyCalPoint> forward_scaling;
    std::vector<DutyCalPoint> reverse_scaling;
    MinDutyThresholds forward_min_duty;
    MinDutyThresholds reverse_min_duty;
    MinDutyThresholds turning_forward_min_duty;  // Min duty during turning (higher friction)
    MinDutyThresholds turning_reverse_min_duty;
    bool turning_calibrated = false;
    TurnCalResult turn_left;
    TurnCalResult turn_right;
    float effective_track_mm;
    float min_duty_margin = 0.10f;  // 10% safety margin
    bool valid = false;
};

// ============================================================================
// Helper: Get yaw from SLAM pose
// ============================================================================

float getYawFromSlam(SlamEngine& slam) {
    Eigen::Matrix3d R = slam.getRotation();
    return static_cast<float>(std::atan2(R(1, 0), R(0, 0)));
}

// ============================================================================
// Fast Combined Calibration (~12-15 seconds total)
// ============================================================================

/**
 * Run complete calibration in a single efficient sequence:
 * 1. Forward duty sweep with min-duty detection during ramp
 * 2. Reverse duty sweep with min-duty detection during ramp
 * 3. Rotation sweep: left 90° → right 180° → left 90° to center
 */
bool runFastCalibration(
    SlamEngine& slam,
    VescDriver& vesc,
    const CalibrationConfig& config,
    CalibrationResults& results)
{
    auto cal_start = std::chrono::steady_clock::now();

    auto elapsed_ms = [&]() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - cal_start).count();
    };

    // ========================================
    // PHASE 1: Forward Duty Sweep (~2.5s)
    // ========================================
    std::cout << "\n[" << elapsed_ms() << "ms] === FORWARD SWEEP ===" << std::endl;

    results.forward_scaling.clear();
    bool fwd_left_started = false, fwd_right_started = false;
    bool fwd_left_stopped = false, fwd_right_stopped = false;

    // Ramp up 0 → 0.03: detect start thresholds
    std::cout << "  Detecting start thresholds..." << std::endl;
    for (float d = 0.0f; d <= 0.035f; d += 0.002f) {
        vesc.setDutyRaw(d, d);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        slam.process();

        VescStatus sl = vesc.getStatus(config.vesc_left);
        VescStatus sr = vesc.getStatus(config.vesc_right);

        if (!fwd_left_started && std::abs(sl.erpm) > config.erpm_threshold) {
            results.forward_min_duty.start_left = d;
            fwd_left_started = true;
        }
        if (!fwd_right_started && std::abs(sr.erpm) > config.erpm_threshold) {
            results.forward_min_duty.start_right = d;
            fwd_right_started = true;
        }
    }

    // Quick staircase for ERPM sampling
    std::cout << "  Sampling ERPM at duty points..." << std::endl;
    for (float duty : config.duty_points) {
        vesc.setDutyRaw(duty, duty);

        // Short hold for stabilization
        auto hold_start = std::chrono::steady_clock::now();
        std::vector<int32_t> erpm_l, erpm_r;

        while (std::chrono::steady_clock::now() - hold_start <
               std::chrono::milliseconds(static_cast<int>(config.duty_hold_time * 1000))) {
            VescStatus sl = vesc.getStatus(config.vesc_left);
            VescStatus sr = vesc.getStatus(config.vesc_right);
            if (sl.erpm != 0) erpm_l.push_back(std::abs(sl.erpm));
            if (sr.erpm != 0) erpm_r.push_back(std::abs(sr.erpm));
            slam.process();
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        }

        if (!erpm_l.empty() && !erpm_r.empty()) {
            float avg_l = 0, avg_r = 0;
            for (auto e : erpm_l) avg_l += e;
            for (auto e : erpm_r) avg_r += e;
            avg_l /= erpm_l.size();
            avg_r /= erpm_r.size();

            DutyCalPoint p;
            p.duty = duty;
            p.erpm_left = avg_l;
            p.erpm_right = avg_r;
            p.scale_right = (avg_r > 0) ? (avg_l / avg_r) : 1.0f;
            results.forward_scaling.push_back(p);

            std::cout << "    " << duty << ": L=" << (int)avg_l
                      << " R=" << (int)avg_r << " scale=" << p.scale_right << std::endl;
        }
    }

    // Ramp down: detect keep thresholds
    std::cout << "  Detecting keep thresholds..." << std::endl;
    for (float d = config.max_duty; d >= 0; d -= 0.003f) {
        vesc.setDutyRaw(d, d);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        slam.process();

        VescStatus sl = vesc.getStatus(config.vesc_left);
        VescStatus sr = vesc.getStatus(config.vesc_right);

        if (!fwd_left_stopped && std::abs(sl.erpm) < config.erpm_threshold) {
            results.forward_min_duty.keep_left = d + 0.003f;
            fwd_left_stopped = true;
        }
        if (!fwd_right_stopped && std::abs(sr.erpm) < config.erpm_threshold) {
            results.forward_min_duty.keep_right = d + 0.003f;
            fwd_right_stopped = true;
        }
    }
    vesc.stop();

    std::cout << "  Forward min duty: start L=" << results.forward_min_duty.start_left
              << " R=" << results.forward_min_duty.start_right
              << " keep L=" << results.forward_min_duty.keep_left
              << " R=" << results.forward_min_duty.keep_right << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // ========================================
    // PHASE 2: Reverse Duty Sweep (~2.5s)
    // ========================================
    std::cout << "\n[" << elapsed_ms() << "ms] === REVERSE SWEEP ===" << std::endl;

    results.reverse_scaling.clear();
    bool rev_left_started = false, rev_right_started = false;
    bool rev_left_stopped = false, rev_right_stopped = false;

    // Ramp up (negative direction)
    std::cout << "  Detecting start thresholds..." << std::endl;
    for (float d = 0.0f; d <= 0.035f; d += 0.002f) {
        vesc.setDutyRaw(-d, -d);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        slam.process();

        VescStatus sl = vesc.getStatus(config.vesc_left);
        VescStatus sr = vesc.getStatus(config.vesc_right);

        if (!rev_left_started && std::abs(sl.erpm) > config.erpm_threshold) {
            results.reverse_min_duty.start_left = d;
            rev_left_started = true;
        }
        if (!rev_right_started && std::abs(sr.erpm) > config.erpm_threshold) {
            results.reverse_min_duty.start_right = d;
            rev_right_started = true;
        }
    }

    // ERPM sampling (reverse)
    std::cout << "  Sampling ERPM at duty points..." << std::endl;
    for (float duty : config.duty_points) {
        vesc.setDutyRaw(-duty, -duty);

        auto hold_start = std::chrono::steady_clock::now();
        std::vector<int32_t> erpm_l, erpm_r;

        while (std::chrono::steady_clock::now() - hold_start <
               std::chrono::milliseconds(static_cast<int>(config.duty_hold_time * 1000))) {
            VescStatus sl = vesc.getStatus(config.vesc_left);
            VescStatus sr = vesc.getStatus(config.vesc_right);
            if (sl.erpm != 0) erpm_l.push_back(std::abs(sl.erpm));
            if (sr.erpm != 0) erpm_r.push_back(std::abs(sr.erpm));
            slam.process();
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        }

        if (!erpm_l.empty() && !erpm_r.empty()) {
            float avg_l = 0, avg_r = 0;
            for (auto e : erpm_l) avg_l += e;
            for (auto e : erpm_r) avg_r += e;
            avg_l /= erpm_l.size();
            avg_r /= erpm_r.size();

            DutyCalPoint p;
            p.duty = duty;
            p.erpm_left = avg_l;
            p.erpm_right = avg_r;
            p.scale_right = (avg_r > 0) ? (avg_l / avg_r) : 1.0f;
            results.reverse_scaling.push_back(p);

            std::cout << "    " << duty << ": L=" << (int)avg_l
                      << " R=" << (int)avg_r << " scale=" << p.scale_right << std::endl;
        }
    }

    // Ramp down (reverse)
    std::cout << "  Detecting keep thresholds..." << std::endl;
    for (float d = config.max_duty; d >= 0; d -= 0.003f) {
        vesc.setDutyRaw(-d, -d);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        slam.process();

        VescStatus sl = vesc.getStatus(config.vesc_left);
        VescStatus sr = vesc.getStatus(config.vesc_right);

        if (!rev_left_stopped && std::abs(sl.erpm) < config.erpm_threshold) {
            results.reverse_min_duty.keep_left = d + 0.003f;
            rev_left_stopped = true;
        }
        if (!rev_right_stopped && std::abs(sr.erpm) < config.erpm_threshold) {
            results.reverse_min_duty.keep_right = d + 0.003f;
            rev_right_stopped = true;
        }
    }
    vesc.stop();

    std::cout << "  Reverse min duty: start L=" << results.reverse_min_duty.start_left
              << " R=" << results.reverse_min_duty.start_right
              << " keep L=" << results.reverse_min_duty.keep_left
              << " R=" << results.reverse_min_duty.keep_right << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // ========================================
    // PHASE 3: Rotation Sweep (~8s)
    // Turn left 90° → right 180° → left 90° (back to center)
    // ========================================
    std::cout << "\n[" << elapsed_ms() << "ms] === ROTATION SWEEP ===" << std::endl;

    float start_yaw = getYawFromSlam(slam);
    float target_rad = config.target_angle_deg * M_PI / 180.0f;
    float duty = config.turn_duty;

    // --- Turn LEFT 90° ---
    std::cout << "  Turning LEFT 90°..." << std::endl;
    VescOdometry odom_start = vesc.getOdometry();
    auto turn_start = std::chrono::steady_clock::now();

    while (true) {
        float current_yaw = getYawFromSlam(slam);
        float delta = current_yaw - start_yaw;
        while (delta > M_PI) delta -= 2.0f * M_PI;
        while (delta < -M_PI) delta += 2.0f * M_PI;

        if (delta >= target_rad) break;
        if (std::chrono::steady_clock::now() - turn_start > std::chrono::seconds(5)) {
            std::cout << " (timeout)" << std::endl;
            break;
        }

        vesc.setDutyRaw(-duty, duty);  // Left turn
        slam.process();
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    vesc.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    slam.process();

    VescOdometry odom_left = vesc.getOdometry();
    float yaw_after_left = getYawFromSlam(slam);
    float angle_left = yaw_after_left - start_yaw;
    while (angle_left > M_PI) angle_left -= 2.0f * M_PI;
    while (angle_left < -M_PI) angle_left += 2.0f * M_PI;

    results.turn_left.angle_rad = std::abs(angle_left);
    results.turn_left.ticks_left = std::abs(odom_left.tach_left - odom_start.tach_left);
    results.turn_left.ticks_right = std::abs(odom_left.tach_right - odom_start.tach_right);
    int32_t avg_ticks_left = (results.turn_left.ticks_left + results.turn_left.ticks_right) / 2;
    results.turn_left.ticks_per_radian = (results.turn_left.angle_rad > 0.1f) ?
        (avg_ticks_left / results.turn_left.angle_rad) : 0.0f;

    std::cout << "    Angle: " << (results.turn_left.angle_rad * 180 / M_PI)
              << "° ticks/rad: " << results.turn_left.ticks_per_radian << std::endl;

    // --- Turn RIGHT 180° (through center to 90° right of start) ---
    std::cout << "  Turning RIGHT 180°..." << std::endl;
    VescOdometry odom_mid = vesc.getOdometry();
    turn_start = std::chrono::steady_clock::now();
    float right_target = start_yaw - target_rad;  // 90° right of original start

    while (true) {
        float current_yaw = getYawFromSlam(slam);
        float delta = start_yaw - current_yaw;  // How far right of start
        while (delta > M_PI) delta -= 2.0f * M_PI;
        while (delta < -M_PI) delta += 2.0f * M_PI;

        if (delta >= target_rad) break;  // Reached 90° right of start
        if (std::chrono::steady_clock::now() - turn_start > std::chrono::seconds(8)) {
            std::cout << " (timeout)" << std::endl;
            break;
        }

        vesc.setDutyRaw(duty, -duty);  // Right turn
        slam.process();
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    vesc.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    slam.process();

    VescOdometry odom_right = vesc.getOdometry();
    float yaw_after_right = getYawFromSlam(slam);
    float angle_right = yaw_after_left - yaw_after_right;  // Angle turned right
    while (angle_right > M_PI) angle_right -= 2.0f * M_PI;
    while (angle_right < -M_PI) angle_right += 2.0f * M_PI;

    // For right turn calibration, we only care about the 90° portion
    results.turn_right.angle_rad = std::abs(angle_right) / 2.0f;  // Half of 180°
    int32_t ticks_180 = (std::abs(odom_right.tach_left - odom_mid.tach_left) +
                         std::abs(odom_right.tach_right - odom_mid.tach_right)) / 2;
    results.turn_right.ticks_left = ticks_180 / 2;
    results.turn_right.ticks_right = ticks_180 / 2;
    results.turn_right.ticks_per_radian = (std::abs(angle_right) > 0.1f) ?
        (ticks_180 / std::abs(angle_right)) : 0.0f;

    std::cout << "    Angle: " << (std::abs(angle_right) * 180 / M_PI)
              << "° ticks/rad: " << results.turn_right.ticks_per_radian << std::endl;

    // --- Return to center (turn LEFT 90°) ---
    std::cout << "  Returning to CENTER..." << std::endl;
    turn_start = std::chrono::steady_clock::now();

    while (true) {
        float current_yaw = getYawFromSlam(slam);
        float error = current_yaw - start_yaw;
        while (error > M_PI) error -= 2.0f * M_PI;
        while (error < -M_PI) error += 2.0f * M_PI;

        if (std::abs(error) < 0.10f) break;  // ~6 degrees is good enough
        if (std::chrono::steady_clock::now() - turn_start > std::chrono::seconds(5)) {
            std::cout << " (timeout, error=" << (error * 180 / M_PI) << "°)" << std::endl;
            break;
        }

        int dir = (error > 0) ? -1 : 1;  // Turn toward center
        vesc.setDutyRaw(-duty * dir, duty * dir);
        slam.process();
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    vesc.stop();

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // ========================================
    // PHASE 4: Turning Min Duty Calibration (~3s)
    // Higher friction due to wheel scrubbing during skid-steer turns
    // ========================================
    std::cout << "\n[" << elapsed_ms() << "ms] === TURNING MIN DUTY ===" << std::endl;

    // Test RIGHT turn (left wheel forward, right wheel backward)
    // This measures "forward turning" for left wheel, "reverse turning" for right wheel
    std::cout << "  Detecting turning start thresholds (RIGHT turn)..." << std::endl;
    bool turn_fwd_left_started = false, turn_rev_right_started = false;
    bool turn_fwd_left_stopped = false, turn_rev_right_stopped = false;

    // Ramp up to detect start thresholds during turning
    for (float d = 0.0f; d <= 0.050f; d += 0.002f) {
        vesc.setDutyRaw(d, -d);  // Left forward, right backward (right turn)
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        slam.process();

        VescStatus sl = vesc.getStatus(config.vesc_left);
        VescStatus sr = vesc.getStatus(config.vesc_right);

        if (!turn_fwd_left_started && std::abs(sl.erpm) > config.erpm_threshold) {
            results.turning_forward_min_duty.start_left = d;
            turn_fwd_left_started = true;
        }
        if (!turn_rev_right_started && std::abs(sr.erpm) > config.erpm_threshold) {
            results.turning_reverse_min_duty.start_right = d;
            turn_rev_right_started = true;
        }
    }

    // Hold briefly at max turning duty
    float turn_max_duty = 0.06f;
    vesc.setDutyRaw(turn_max_duty, -turn_max_duty);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    slam.process();

    // Ramp down to detect keep thresholds
    std::cout << "  Detecting turning keep thresholds (RIGHT turn)..." << std::endl;
    for (float d = turn_max_duty; d >= 0; d -= 0.003f) {
        vesc.setDutyRaw(d, -d);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        slam.process();

        VescStatus sl = vesc.getStatus(config.vesc_left);
        VescStatus sr = vesc.getStatus(config.vesc_right);

        if (!turn_fwd_left_stopped && std::abs(sl.erpm) < config.erpm_threshold) {
            results.turning_forward_min_duty.keep_left = d + 0.003f;
            turn_fwd_left_stopped = true;
        }
        if (!turn_rev_right_stopped && std::abs(sr.erpm) < config.erpm_threshold) {
            results.turning_reverse_min_duty.keep_right = d + 0.003f;
            turn_rev_right_stopped = true;
        }
    }
    vesc.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // Test LEFT turn (left wheel backward, right wheel forward)
    // This measures "reverse turning" for left wheel, "forward turning" for right wheel
    std::cout << "  Detecting turning start thresholds (LEFT turn)..." << std::endl;
    bool turn_rev_left_started = false, turn_fwd_right_started = false;
    bool turn_rev_left_stopped = false, turn_fwd_right_stopped = false;

    for (float d = 0.0f; d <= 0.050f; d += 0.002f) {
        vesc.setDutyRaw(-d, d);  // Left backward, right forward (left turn)
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        slam.process();

        VescStatus sl = vesc.getStatus(config.vesc_left);
        VescStatus sr = vesc.getStatus(config.vesc_right);

        if (!turn_rev_left_started && std::abs(sl.erpm) > config.erpm_threshold) {
            results.turning_reverse_min_duty.start_left = d;
            turn_rev_left_started = true;
        }
        if (!turn_fwd_right_started && std::abs(sr.erpm) > config.erpm_threshold) {
            results.turning_forward_min_duty.start_right = d;
            turn_fwd_right_started = true;
        }
    }

    // Hold briefly
    vesc.setDutyRaw(-turn_max_duty, turn_max_duty);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    slam.process();

    // Ramp down
    std::cout << "  Detecting turning keep thresholds (LEFT turn)..." << std::endl;
    for (float d = turn_max_duty; d >= 0; d -= 0.003f) {
        vesc.setDutyRaw(-d, d);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        slam.process();

        VescStatus sl = vesc.getStatus(config.vesc_left);
        VescStatus sr = vesc.getStatus(config.vesc_right);

        if (!turn_rev_left_stopped && std::abs(sl.erpm) < config.erpm_threshold) {
            results.turning_reverse_min_duty.keep_left = d + 0.003f;
            turn_rev_left_stopped = true;
        }
        if (!turn_fwd_right_stopped && std::abs(sr.erpm) < config.erpm_threshold) {
            results.turning_forward_min_duty.keep_right = d + 0.003f;
            turn_fwd_right_stopped = true;
        }
    }
    vesc.stop();

    results.turning_calibrated = true;
    std::cout << "  Turning forward min duty: start L=" << results.turning_forward_min_duty.start_left
              << " R=" << results.turning_forward_min_duty.start_right
              << " keep L=" << results.turning_forward_min_duty.keep_left
              << " R=" << results.turning_forward_min_duty.keep_right << std::endl;
    std::cout << "  Turning reverse min duty: start L=" << results.turning_reverse_min_duty.start_left
              << " R=" << results.turning_reverse_min_duty.start_right
              << " keep L=" << results.turning_reverse_min_duty.keep_left
              << " R=" << results.turning_reverse_min_duty.keep_right << std::endl;

    // Calculate effective track width
    float avg_tpr = (results.turn_left.ticks_per_radian + results.turn_right.ticks_per_radian) / 2.0f;
    float avg_ticks_per_meter = 14093.0f;
    results.effective_track_mm = 2.0f * avg_tpr / avg_ticks_per_meter * 1000.0f;

    results.valid = true;

    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - cal_start).count();
    std::cout << "\n[" << total_time << "ms] === CALIBRATION COMPLETE ===" << std::endl;

    return true;
}

// ============================================================================
// Save Results
// ============================================================================

void saveCalibration(const CalibrationConfig& config, const CalibrationResults& results) {
    std::ofstream file(config.calibration_file);
    if (!file.is_open()) {
        std::cerr << "Failed to save calibration to " << config.calibration_file << std::endl;
        return;
    }

    file << "# VESC Calibration - Generated by calibration_run\n\n";

    file << "[geometry]\n";
    file << "effective_track_mm=" << results.effective_track_mm << "\n\n";

    file << "[rotation]\n";
    file << "ticks_per_radian_left=" << results.turn_left.ticks_per_radian << "\n";
    file << "ticks_per_radian_right=" << results.turn_right.ticks_per_radian << "\n\n";

    file << "[vesc]\n";
    file << "vesc_id_left=" << static_cast<int>(config.vesc_left) << "\n";
    file << "vesc_id_right=" << static_cast<int>(config.vesc_right) << "\n\n";

    // Forward duty scaling (left is always 1.0, right gets scale)
    file << "[forward_duty_scale]\n";
    file << "# duty:scale_right (left is always 1.0)\n";
    for (const auto& p : results.forward_scaling) {
        file << p.duty << ":" << p.scale_right << "\n";
    }
    file << "\n";

    // Reverse duty scaling
    file << "[reverse_duty_scale]\n";
    file << "# duty:scale_right (left is always 1.0)\n";
    for (const auto& p : results.reverse_scaling) {
        file << p.duty << ":" << p.scale_right << "\n";
    }
    file << "\n";

    // Forward minimum duty thresholds
    file << "[forward_min_duty]\n";
    file << "start_left=" << results.forward_min_duty.start_left << "\n";
    file << "start_right=" << results.forward_min_duty.start_right << "\n";
    file << "keep_left=" << results.forward_min_duty.keep_left << "\n";
    file << "keep_right=" << results.forward_min_duty.keep_right << "\n\n";

    // Reverse minimum duty thresholds
    file << "[reverse_min_duty]\n";
    file << "start_left=" << results.reverse_min_duty.start_left << "\n";
    file << "start_right=" << results.reverse_min_duty.start_right << "\n";
    file << "keep_left=" << results.reverse_min_duty.keep_left << "\n";
    file << "keep_right=" << results.reverse_min_duty.keep_right << "\n\n";

    // Turning minimum duty thresholds (higher friction due to wheel scrubbing)
    if (results.turning_calibrated) {
        file << "[turning_forward_min_duty]\n";
        file << "start_left=" << results.turning_forward_min_duty.start_left << "\n";
        file << "start_right=" << results.turning_forward_min_duty.start_right << "\n";
        file << "keep_left=" << results.turning_forward_min_duty.keep_left << "\n";
        file << "keep_right=" << results.turning_forward_min_duty.keep_right << "\n\n";

        file << "[turning_reverse_min_duty]\n";
        file << "start_left=" << results.turning_reverse_min_duty.start_left << "\n";
        file << "start_right=" << results.turning_reverse_min_duty.start_right << "\n";
        file << "keep_left=" << results.turning_reverse_min_duty.keep_left << "\n";
        file << "keep_right=" << results.turning_reverse_min_duty.keep_right << "\n\n";
    }

    // Safety margin
    file << "[safety]\n";
    file << "min_duty_margin=" << results.min_duty_margin << "\n";

    file.close();
    std::cout << "Calibration saved to " << config.calibration_file << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "============================================" << std::endl;
    std::cout << "  Combined Calibration + Localization Run" << std::endl;
    std::cout << "============================================" << std::endl;

    CalibrationConfig config;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) {
            config.host_ip = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            config.device_ip = argv[++i];
        } else if (arg == "--can" && i + 1 < argc) {
            config.can_port = argv[++i];
        } else if (arg == "--no-vis") {
            config.visualize = false;
        } else if (arg == "--help") {
            std::cout << "Usage: calibration_run [options]\n"
                      << "  --host IP      Host IP (default: 192.168.1.50)\n"
                      << "  --device IP    LiDAR IP (default: 192.168.1.144)\n"
                      << "  --can PORT     CAN port (default: COM3)\n"
                      << "  --no-vis       Disable visualization\n";
            return 0;
        }
    }

    // Initialize SLAM engine
    std::cout << "\nInitializing SLAM engine..." << std::endl;
    SlamConfig slam_config;
    slam_config.deskew_enabled = true;
    slam_config.filter_size_surf = 0.2;  // Indoor voxel size
    slam_config.gyr_cov = 0.1;

    SlamEngine slam;
    if (!slam.init(slam_config)) {
        std::cerr << "Failed to initialize SLAM engine" << std::endl;
        return 1;
    }

    // Initialize LiDAR
    std::cout << "Connecting to LiDAR at " << config.device_ip << "..." << std::endl;
    LivoxMid360 lidar;

    if (!lidar.connect(config.device_ip, config.host_ip)) {
        std::cerr << "Failed to connect to LiDAR" << std::endl;
        return 1;
    }

    // Scan accumulation state (matching live_slam pattern)
    std::mutex scan_mutex;
    std::vector<LidarPoint> accumulated_points;
    uint64_t scan_start_time = 0;
    constexpr double SCAN_PERIOD_MS = 100.0;  // 10Hz scan rate
    constexpr double BLIND_DIST = 0.5;        // Min distance (indoor)
    constexpr int POINT_FILTER = 3;           // Keep every 3rd point
    size_t valid_point_counter = 0;

    // Set up point cloud callback (accumulates into complete scans)
    lidar.setPointCloudCallback([&](const LivoxPointCloudFrame& frame) {
        if (frame.points.empty()) return;

        std::lock_guard<std::mutex> lock(scan_mutex);

        // Start new scan if empty
        if (accumulated_points.empty()) {
            scan_start_time = frame.timestamp_ns;
        }

        // Calculate frame time offset from scan start (ms)
        double frame_offset_ms = (frame.timestamp_ns - scan_start_time) / 1e6;

        // Convert and filter points (matching live_slam settings)
        for (size_t i = 0; i < frame.points.size(); i++) {
            const V3D& p = frame.points[i];

            // Tag filtering - only keep valid returns
            if (!frame.tags.empty()) {
                uint8_t return_type = frame.tags[i] & 0x30;
                if (return_type != 0x00 && return_type != 0x10) continue;
            }

            // Distance filtering
            double dist = p.norm();
            if (dist < BLIND_DIST || dist > 100.0) continue;

            // Skip zero points
            if (std::abs(p.x()) < 0.001 && std::abs(p.y()) < 0.001 && std::abs(p.z()) < 0.001) continue;

            // Point decimation
            valid_point_counter++;
            if (POINT_FILTER > 1 && valid_point_counter % POINT_FILTER != 0) continue;

            LidarPoint lp;
            lp.x = static_cast<float>(p.x());
            lp.y = static_cast<float>(p.y());
            lp.z = static_cast<float>(p.z());
            lp.intensity = static_cast<float>(frame.reflectivities[i]);
            float point_offset_ms = (i < frame.time_offsets_us.size()) ?
                (frame.time_offsets_us[i] / 1000.0f) : 0.0f;
            lp.time_offset_ms = static_cast<float>(frame_offset_ms) + point_offset_ms;
            lp.tag = frame.tags.empty() ? 0 : frame.tags[i];
            lp.line = 0;

            accumulated_points.push_back(lp);
        }

        // Check if we have a complete scan (~100ms)
        if (frame_offset_ms >= SCAN_PERIOD_MS && accumulated_points.size() > 100) {
            PointCloud cloud;
            cloud.timestamp_ns = scan_start_time;
            cloud.timestamp_end_ns = frame.timestamp_ns;
            cloud.points = std::move(accumulated_points);
            accumulated_points.clear();
            accumulated_points.reserve(20000);

            slam.addPointCloud(cloud);
        }
    });

    // Set up IMU callback (using correct LivoxIMUFrame type)
    lidar.setIMUCallback([&slam](const LivoxIMUFrame& frame) {
        ImuData imu;
        imu.timestamp_ns = frame.timestamp_ns;
        // Mid-360 outputs accel in g-units, convert to m/s²
        constexpr double G_TO_MS2 = 9.81;
        imu.acc = frame.accel * G_TO_MS2;
        imu.gyro = frame.gyro;  // Already in rad/s
        slam.addImuData(imu);
    });

    if (!lidar.startStreaming()) {
        std::cerr << "Failed to start LiDAR streaming" << std::endl;
        return 1;
    }

    // Wait for SLAM initialization
    std::cout << "Waiting for SLAM initialization (keep robot STILL)..." << std::endl;
    for (int i = 0; i < 50; i++) {  // 5 seconds max
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        slam.process();  // Process buffered sensor data
        if (slam.isInitialized()) {
            std::cout << "SLAM initialized after " << (i * 100) << "ms" << std::endl;
            break;
        }
    }

    if (!slam.isInitialized()) {
        std::cerr << "SLAM failed to initialize" << std::endl;
        lidar.stop();
        return 1;
    }
    std::cout << "SLAM initialized!" << std::endl;

    // Initialize VESC driver
    std::cout << "Connecting to VESC motors on " << config.can_port << "..." << std::endl;
    VescDriver vesc;
    if (!vesc.init(config.can_port, config.vesc_left, config.vesc_right)) {
        std::cerr << "Failed to connect to VESC motors" << std::endl;
        lidar.stop();
        return 1;
    }

    // Wait for tachometer readings
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    CalibrationResults results;

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Ready to run FAST calibration (~15s)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Press Enter to start (robot will move!)..." << std::endl;
    std::cin.get();

    vesc.resetOdometry();

    // Run fast combined calibration
    if (!runFastCalibration(slam, vesc, config, results)) {
        std::cerr << "Calibration failed!" << std::endl;
    }

    // Print results
    std::cout << "\n========================================" << std::endl;
    std::cout << "  CALIBRATION COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\nFORWARD Duty Scaling:" << std::endl;
    std::cout << "  Duty   | ERPM L  | ERPM R  | Scale R" << std::endl;
    std::cout << "  -------|---------|---------|--------" << std::endl;
    for (const auto& p : results.forward_scaling) {
        printf("  %.3f  | %7.0f | %7.0f | %.3f\n",
               p.duty, p.erpm_left, p.erpm_right, p.scale_right);
    }

    std::cout << "\nREVERSE Duty Scaling:" << std::endl;
    std::cout << "  Duty   | ERPM L  | ERPM R  | Scale R" << std::endl;
    std::cout << "  -------|---------|---------|--------" << std::endl;
    for (const auto& p : results.reverse_scaling) {
        printf("  %.3f  | %7.0f | %7.0f | %.3f\n",
               p.duty, p.erpm_left, p.erpm_right, p.scale_right);
    }

    std::cout << "\nMinimum Duty Thresholds (Straight-line):" << std::endl;
    std::cout << "  Direction | Start L | Start R | Keep L  | Keep R" << std::endl;
    std::cout << "  ----------|---------|---------|---------|-------" << std::endl;
    printf("  Forward   | %.3f   | %.3f   | %.3f   | %.3f\n",
           results.forward_min_duty.start_left, results.forward_min_duty.start_right,
           results.forward_min_duty.keep_left, results.forward_min_duty.keep_right);
    printf("  Reverse   | %.3f   | %.3f   | %.3f   | %.3f\n",
           results.reverse_min_duty.start_left, results.reverse_min_duty.start_right,
           results.reverse_min_duty.keep_left, results.reverse_min_duty.keep_right);

    if (results.turning_calibrated) {
        std::cout << "\nMinimum Duty Thresholds (Turning - higher friction):" << std::endl;
        std::cout << "  Direction | Start L | Start R | Keep L  | Keep R" << std::endl;
        std::cout << "  ----------|---------|---------|---------|-------" << std::endl;
        printf("  Turn Fwd  | %.3f   | %.3f   | %.3f   | %.3f\n",
               results.turning_forward_min_duty.start_left, results.turning_forward_min_duty.start_right,
               results.turning_forward_min_duty.keep_left, results.turning_forward_min_duty.keep_right);
        printf("  Turn Rev  | %.3f   | %.3f   | %.3f   | %.3f\n",
               results.turning_reverse_min_duty.start_left, results.turning_reverse_min_duty.start_right,
               results.turning_reverse_min_duty.keep_left, results.turning_reverse_min_duty.keep_right);
        std::cout << "\nSafety Margin: " << (results.min_duty_margin * 100) << "%" << std::endl;
    }

    std::cout << "\nRotation Calibration:" << std::endl;
    std::cout << "  Left turn:  " << (results.turn_left.angle_rad * 180 / M_PI) << "° -> "
              << results.turn_left.ticks_per_radian << " ticks/rad" << std::endl;
    std::cout << "  Right turn: " << (results.turn_right.angle_rad * 180 / M_PI) << "° -> "
              << results.turn_right.ticks_per_radian << " ticks/rad" << std::endl;
    std::cout << "  Effective track: " << results.effective_track_mm << " mm" << std::endl;

    // Save calibration
    saveCalibration(config, results);

    // Save map
    std::cout << "\nSaving local map to " << config.map_file << "..." << std::endl;
    slam.saveMap(config.map_file);

    // Cleanup
    vesc.shutdown();
    lidar.stop();

    std::cout << "\nCalibration run complete!" << std::endl;
    return 0;
}
