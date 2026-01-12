/**
 * Motion Sequence Test
 *
 * Tests MotionController with calibration:
 * 1. Forward 1m
 * 2. Turn left 90°
 * 3. Turn right 90° (back to original heading)
 * 4. Reverse 1m (back to start)
 *
 * Uses odometry feedback to track progress.
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
#include <chrono>
#include <thread>
#include <cmath>

#include "slam/vesc_driver.hpp"
#include "slam/motion_controller.hpp"

using namespace slam;

constexpr float PI = 3.14159265358979f;
constexpr float DEG_TO_RAD = PI / 180.0f;
constexpr float RAD_TO_DEG = 180.0f / PI;

// Motion parameters
constexpr float DRIVE_DUTY = 0.8f;      // 80% of max (for closed-loop speed)
constexpr float TURN_DUTY = 0.6f;       // 60% of max for turning
constexpr float MAX_DUTY = 0.12f;       // 12% actual duty (higher for speed)
constexpr float RAMP_RATE = 0.4f;       // Faster ramping
constexpr float UPDATE_RATE_HZ = 50.0f;
constexpr float DT = 1.0f / UPDATE_RATE_HZ;

// Targets
constexpr float TARGET_DISTANCE_M = 1.0f;
constexpr float TARGET_ANGLE_DEG = 90.0f;
constexpr float DISTANCE_TOLERANCE_M = 0.02f;  // 2cm
constexpr float ANGLE_TOLERANCE_DEG = 3.0f;    // 3 degrees

// Timeouts
constexpr float DRIVE_TIMEOUT_S = 15.0f;
constexpr float TURN_TIMEOUT_S = 10.0f;

float normalizeAngle(float angle) {
    while (angle > PI) angle -= 2 * PI;
    while (angle < -PI) angle += 2 * PI;
    return angle;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================" << std::endl;
    std::cout << "  Motion Sequence Test (with calibration)" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Sequence: Forward 1m → Left 90° → Right 90° → Reverse 1m" << std::endl;

    // Configuration
    std::string can_port = "COM3";
    std::string cal_file = "vesc_calibration.ini";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--can" && i + 1 < argc) can_port = argv[++i];
        else if (arg == "--cal" && i + 1 < argc) cal_file = argv[++i];
        else if (arg == "--help") {
            std::cout << "Usage: test_motion_sequence [--can PORT] [--cal FILE]\n";
            return 0;
        }
    }

    // Initialize VESC driver
    std::cout << "\nConnecting to VESCs on " << can_port << "..." << std::endl;
    VescDriver vesc;
    if (!vesc.init(can_port, 1, 126)) {
        std::cerr << "Failed to connect to VESCs" << std::endl;
        return 1;
    }

    // Initialize motion controller
    std::cout << "Loading calibration from " << cal_file << "..." << std::endl;
    MotionController motion;
    if (!motion.init(&vesc, cal_file)) {
        std::cerr << "Failed to load calibration (using defaults)" << std::endl;
    }

    // Configure for higher speed
    motion.setMaxDuty(MAX_DUTY);
    motion.setRampRate(RAMP_RATE);

    // Print config
    const auto& cal = motion.getCalibration();
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Max duty: " << MAX_DUTY << " (" << (MAX_DUTY * 100) << "%)" << std::endl;
    std::cout << "  Drive duty: " << (DRIVE_DUTY * MAX_DUTY) << " actual" << std::endl;
    std::cout << "  Effective track: " << (cal.effective_track_m * 1000) << " mm" << std::endl;
    std::cout << "  Ticks/meter: " << cal.ticks_per_meter << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Press Enter to start sequence" << std::endl;
    std::cout << "  (Robot will move!)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cin.get();

    motion.resetPose();
    auto last_time = std::chrono::steady_clock::now();
    auto phase_start = last_time;

    // ================================================================
    // PHASE 1: Forward 1m
    // ================================================================
    std::cout << "\n>>> PHASE 1: Forward " << TARGET_DISTANCE_M << "m <<<" << std::endl;

    Pose2D start_pose = motion.getPose();
    float start_x = start_pose.x;
    float start_y = start_pose.y;

    motion.setDuty(DRIVE_DUTY, 0.0f);  // Forward, no turn
    phase_start = std::chrono::steady_clock::now();

    while (true) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        float elapsed = std::chrono::duration<float>(now - phase_start).count();
        last_time = now;

        motion.update(dt);

        Pose2D pose = motion.getPose();
        float dx = pose.x - start_x;
        float dy = pose.y - start_y;
        float distance = std::sqrt(dx * dx + dy * dy);

        // Get ERPM for display
        VescStatus sl = vesc.getStatus(1);
        VescStatus sr = vesc.getStatus(126);

        std::cout << "\r  dist=" << distance << "m / " << TARGET_DISTANCE_M << "m"
                  << " | duty L=" << motion.getCurrentDutyLeft()
                  << " R=" << motion.getCurrentDutyRight()
                  << " | ERPM L=" << sl.erpm << " R=" << sr.erpm
                  << "        " << std::flush;

        if (distance >= TARGET_DISTANCE_M - DISTANCE_TOLERANCE_M) {
            std::cout << "\n  Target reached!" << std::endl;
            break;
        }

        if (elapsed > DRIVE_TIMEOUT_S) {
            std::cout << "\n  Timeout!" << std::endl;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // Stop and settle
    motion.stop(false);
    for (int i = 0; i < 25; i++) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        motion.update(dt);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    Pose2D after_forward = motion.getPose();
    std::cout << "  Position after forward: x=" << after_forward.x
              << " y=" << after_forward.y
              << " theta=" << (after_forward.theta * RAD_TO_DEG) << "°" << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // ================================================================
    // PHASE 2: Turn left 90°
    // ================================================================
    std::cout << "\n>>> PHASE 2: Turn LEFT " << TARGET_ANGLE_DEG << "° <<<" << std::endl;

    float start_theta = motion.getPose().theta;
    float target_theta = normalizeAngle(start_theta + TARGET_ANGLE_DEG * DEG_TO_RAD);

    motion.setDuty(0.0f, TURN_DUTY);  // No forward, turn left
    phase_start = std::chrono::steady_clock::now();

    while (true) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        float elapsed = std::chrono::duration<float>(now - phase_start).count();
        last_time = now;

        motion.update(dt);

        Pose2D pose = motion.getPose();
        float delta_theta = normalizeAngle(pose.theta - start_theta);
        float progress_deg = delta_theta * RAD_TO_DEG;

        std::cout << "\r  angle=" << progress_deg << "° / " << TARGET_ANGLE_DEG << "°"
                  << " | duty L=" << motion.getCurrentDutyLeft()
                  << " R=" << motion.getCurrentDutyRight()
                  << "        " << std::flush;

        if (delta_theta >= (TARGET_ANGLE_DEG - ANGLE_TOLERANCE_DEG) * DEG_TO_RAD) {
            std::cout << "\n  Target reached!" << std::endl;
            break;
        }

        if (elapsed > TURN_TIMEOUT_S) {
            std::cout << "\n  Timeout!" << std::endl;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // Stop and settle
    motion.stop(false);
    for (int i = 0; i < 25; i++) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        motion.update(dt);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    Pose2D after_left_turn = motion.getPose();
    std::cout << "  Heading after left turn: " << (after_left_turn.theta * RAD_TO_DEG) << "°" << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // ================================================================
    // PHASE 3: Turn right 90° (back to original heading)
    // ================================================================
    std::cout << "\n>>> PHASE 3: Turn RIGHT " << TARGET_ANGLE_DEG << "° <<<" << std::endl;

    start_theta = motion.getPose().theta;
    target_theta = normalizeAngle(start_theta - TARGET_ANGLE_DEG * DEG_TO_RAD);

    motion.setDuty(0.0f, -TURN_DUTY);  // No forward, turn right (negative angular)
    phase_start = std::chrono::steady_clock::now();

    while (true) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        float elapsed = std::chrono::duration<float>(now - phase_start).count();
        last_time = now;

        motion.update(dt);

        Pose2D pose = motion.getPose();
        float delta_theta = normalizeAngle(start_theta - pose.theta);  // Right turn = decreasing theta
        float progress_deg = delta_theta * RAD_TO_DEG;

        std::cout << "\r  angle=" << progress_deg << "° / " << TARGET_ANGLE_DEG << "°"
                  << " | duty L=" << motion.getCurrentDutyLeft()
                  << " R=" << motion.getCurrentDutyRight()
                  << "        " << std::flush;

        if (delta_theta >= (TARGET_ANGLE_DEG - ANGLE_TOLERANCE_DEG) * DEG_TO_RAD) {
            std::cout << "\n  Target reached!" << std::endl;
            break;
        }

        if (elapsed > TURN_TIMEOUT_S) {
            std::cout << "\n  Timeout!" << std::endl;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // Stop and settle
    motion.stop(false);
    for (int i = 0; i < 25; i++) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        motion.update(dt);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    Pose2D after_right_turn = motion.getPose();
    std::cout << "  Heading after right turn: " << (after_right_turn.theta * RAD_TO_DEG) << "°" << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // ================================================================
    // PHASE 4: Reverse 1m (back to start)
    // ================================================================
    std::cout << "\n>>> PHASE 4: Reverse " << TARGET_DISTANCE_M << "m <<<" << std::endl;

    start_pose = motion.getPose();
    start_x = start_pose.x;
    start_y = start_pose.y;

    motion.setDuty(-DRIVE_DUTY, 0.0f);  // Reverse, no turn
    phase_start = std::chrono::steady_clock::now();

    while (true) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        float elapsed = std::chrono::duration<float>(now - phase_start).count();
        last_time = now;

        motion.update(dt);

        Pose2D pose = motion.getPose();
        float dx = pose.x - start_x;
        float dy = pose.y - start_y;
        float distance = std::sqrt(dx * dx + dy * dy);

        // Get ERPM for display
        VescStatus sl = vesc.getStatus(1);
        VescStatus sr = vesc.getStatus(126);

        std::cout << "\r  dist=" << distance << "m / " << TARGET_DISTANCE_M << "m"
                  << " | duty L=" << motion.getCurrentDutyLeft()
                  << " R=" << motion.getCurrentDutyRight()
                  << " | ERPM L=" << sl.erpm << " R=" << sr.erpm
                  << "        " << std::flush;

        if (distance >= TARGET_DISTANCE_M - DISTANCE_TOLERANCE_M) {
            std::cout << "\n  Target reached!" << std::endl;
            break;
        }

        if (elapsed > DRIVE_TIMEOUT_S) {
            std::cout << "\n  Timeout!" << std::endl;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // Stop
    motion.stop(false);
    for (int i = 0; i < 25; i++) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_time).count();
        last_time = now;
        motion.update(dt);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // ================================================================
    // RESULTS
    // ================================================================
    Pose2D final_pose = motion.getPose();

    std::cout << "\n================================================" << std::endl;
    std::cout << "  SEQUENCE COMPLETE" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "\nFinal pose:" << std::endl;
    std::cout << "  x: " << final_pose.x << " m (expected: ~0)" << std::endl;
    std::cout << "  y: " << final_pose.y << " m (expected: ~0)" << std::endl;
    std::cout << "  theta: " << (final_pose.theta * RAD_TO_DEG) << "° (expected: ~0°)" << std::endl;

    float return_error = std::sqrt(final_pose.x * final_pose.x + final_pose.y * final_pose.y);
    std::cout << "\nReturn-to-start error: " << (return_error * 100) << " cm" << std::endl;

    // Cleanup
    vesc.shutdown();

    std::cout << "\nTest complete!" << std::endl;
    return 0;
}
