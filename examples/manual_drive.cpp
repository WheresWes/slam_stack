// Manual drive program - Control robot with PS5 DualSense gamepad
// Use this to precisely position the robot before/after recording sessions

#include "slam/gamepad.hpp"
#include "slam/vesc_driver.hpp"
#include "slam/motion_controller.hpp"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>

volatile bool g_running = true;

void signalHandler(int) {
    g_running = false;
}

void printUsage() {
    std::cout << R"(
================================================
  Manual Drive - PS5 DualSense Control
================================================

Connect your PS5 controller via USB or Bluetooth.

Controls:
  Left Stick Y     Forward / Backward
  Left Stick X     Turn Left / Right
  L1               Emergency stop
  D-Pad Up/Down    Increase/Decrease max speed
  Options          Exit program

  100% stick = ~0.48 m/s

LED Colors:
  Green            Idle / Ready
  Blue             Moving
  Yellow           E-Stop active

)";
}

int main(int argc, char** argv) {
    std::signal(SIGINT, signalHandler);

    printUsage();

    // Parse arguments
    std::string com_port = "COM3";  // Default
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            com_port = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: manual_drive.exe [--port COM3]\n";
            return 0;
        }
    }

    // Initialize gamepad
    std::cout << "Initializing gamepad...\n";
    slam::Gamepad gamepad;
    if (!gamepad.init()) {
        std::cerr << "ERROR: No gamepad found!\n";
        std::cerr << "Connect your PS5 controller and try again.\n";
        return 1;
    }
    std::cout << "Gamepad: " << gamepad.getControllerName() << "\n";
    gamepad.setLEDColor(0, 255, 0);  // Green = ready
    gamepad.rumble(0.3f, 0.3f, 200);

    // Initialize VESC driver
    std::cout << "Initializing VESC driver on " << com_port << "...\n";
    slam::VescDriver vesc;
    if (!vesc.init(com_port, 1, 126)) {  // VESC IDs 1 and 126
        std::cerr << "ERROR: Failed to connect to VESC on " << com_port << "\n";
        std::cerr << "Check CAN adapter connection and COM port.\n";
        gamepad.shutdown();
        return 1;
    }
    std::cout << "VESC connected!\n";

    // Initialize motion controller
    slam::MotionController motion;
    if (!motion.init(&vesc, "calibration.ini")) {
        std::cout << "No calibration file - using default calibration\n";
        slam::MotionCalibration cal;  // Use defaults
        motion.init(&vesc, cal);
    } else {
        std::cout << "Loaded calibration from calibration.ini\n";
    }

    // Allow higher speeds for manual driving (default 0.15 is conservative)
    motion.setMaxDuty(0.30f);  // ~0.5 m/s max

    // Drive configuration - single stick mode
    slam::GamepadDriveConfig drive_config;
    drive_config.max_linear_velocity = 0.6f;    // 0.6 m/s max
    drive_config.max_angular_velocity = 3.5f;   // rad/s
    drive_config.stick_deadzone = 0.12f;
    drive_config.max_speed_scale = 0.8f;        // 100% stick = 80% of max (~0.48 m/s)
    drive_config.use_trigger_boost = false;     // Direct stick control
    drive_config.right_stick_steer = false;     // Single stick: left stick X = steering

    // State tracking
    float current_max_speed = drive_config.max_linear_velocity;
    bool was_dpad_up = false;
    bool was_dpad_down = false;
    bool e_stop_active = false;
    const float MAX_SPEED_LIMIT = 0.5f;
    const float MIN_SPEED_LIMIT = 0.05f;

    std::cout << "\n--- READY TO DRIVE ---\n";
    std::cout << "Max speed: " << current_max_speed << " m/s\n\n";

    auto last_print = std::chrono::steady_clock::now();
    auto last_update = std::chrono::steady_clock::now();

    while (g_running && gamepad.isConnected()) {
        gamepad.update();
        auto state = gamepad.getState();

        // Exit on Options button
        if (state.button_start) {
            std::cout << "\nOptions pressed - exiting.\n";
            break;
        }

        // Emergency stop handling
        if (state.left_shoulder) {
            if (!e_stop_active) {
                motion.emergencyStop();
                gamepad.setLEDColor(255, 255, 0);  // Yellow
                gamepad.rumble(0.8f, 0.8f, 500);
                e_stop_active = true;
                std::cout << "\n!!! EMERGENCY STOP !!!\n";
            }
        } else {
            if (e_stop_active) {
                gamepad.setLEDColor(0, 255, 0);  // Green
                e_stop_active = false;
                std::cout << "E-stop released.\n";
            }
        }

        // D-pad speed adjustment
        if (state.dpad_up && !was_dpad_up) {
            current_max_speed = std::min(current_max_speed + 0.05f, MAX_SPEED_LIMIT);
            drive_config.max_linear_velocity = current_max_speed;
            gamepad.rumble(0.2f, 0.2f, 50);
            std::cout << "Max speed: " << current_max_speed << " m/s\n";
        }
        if (state.dpad_down && !was_dpad_down) {
            current_max_speed = std::max(current_max_speed - 0.05f, MIN_SPEED_LIMIT);
            drive_config.max_linear_velocity = current_max_speed;
            gamepad.rumble(0.2f, 0.2f, 50);
            std::cout << "Max speed: " << current_max_speed << " m/s\n";
        }
        was_dpad_up = state.dpad_up;
        was_dpad_down = state.dpad_down;

        // Get drive command from gamepad
        auto cmd = gamepad.getDriveCommand(drive_config);

        // Calculate time delta
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_update).count();
        last_update = now;

        // Update motion controller (unless e-stopped)
        if (!e_stop_active) {
            motion.setVelocity(cmd.linear_velocity, cmd.angular_velocity);
            motion.update(dt);

            // Update LED based on motion
            if (std::abs(cmd.linear_velocity) > 0.01f || std::abs(cmd.angular_velocity) > 0.1f) {
                gamepad.setLEDColor(0, 100, 255);  // Blue = moving
            } else {
                gamepad.setLEDColor(0, 255, 0);    // Green = idle
            }
        }

        // Print status periodically
        auto print_elapsed = std::chrono::duration<float>(now - last_print).count();
        if (print_elapsed >= 0.25f) {  // 4 Hz
            last_print = now;

            auto odom = vesc.getOdometry();

            std::cout << "\r";
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Lin:" << std::setw(6) << cmd.linear_velocity << " m/s  "
                      << "Ang:" << std::setw(5) << cmd.angular_velocity << " rad/s  "
                      << "Spd:" << std::setw(3) << int(state.right_trigger * 100) << "%  "
                      << "| Pos:(" << std::setw(6) << odom.distance_left_m << ","
                      << std::setw(6) << odom.distance_right_m << ") m  ";

            if (e_stop_active) std::cout << "[E-STOP]";
            std::cout << "      " << std::flush;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));  // 50 Hz
    }

    // Clean shutdown
    std::cout << "\n\nShutting down...\n";
    motion.emergencyStop();
    gamepad.setLEDColor(0, 0, 0);
    gamepad.shutdown();
    vesc.shutdown();

    std::cout << "Done.\n";
    return 0;
}
