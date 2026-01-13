// Test program for PS5 DualSense / gamepad controller
// Connect your controller via USB or Bluetooth before running

#include "slam/gamepad.hpp"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <csignal>

volatile bool g_running = true;

void signalHandler(int) {
    g_running = false;
}

int main() {
    std::signal(SIGINT, signalHandler);

    std::cout << "================================================\n";
    std::cout << "  Gamepad Test - PS5 DualSense / Xbox\n";
    std::cout << "================================================\n";
    std::cout << "\nConnect your controller via USB or Bluetooth.\n";
    std::cout << "Press Ctrl+C to exit.\n\n";

    slam::Gamepad gamepad;

    if (!gamepad.init()) {
        std::cerr << "Failed to initialize gamepad!\n";
        std::cerr << "Make sure your controller is connected.\n";
        return 1;
    }

    std::cout << "Controller: " << gamepad.getControllerName() << "\n\n";

    // Flash the LED to confirm connection
    gamepad.setLEDColor(0, 255, 0);  // Green
    gamepad.rumble(0.5f, 0.5f, 200);

    std::cout << "Controls:\n";
    std::cout << "  Left Stick Y  : Forward/Backward\n";
    std::cout << "  Right Stick X : Steering\n";
    std::cout << "  R2 Trigger    : Speed boost\n";
    std::cout << "  L1            : Emergency stop\n";
    std::cout << "  Cross (A)     : Toggle recording\n\n";

    slam::GamepadDriveConfig config;
    config.max_linear_velocity = 0.2f;   // m/s
    config.max_angular_velocity = 1.5f;  // rad/s
    config.min_speed_scale = 0.3f;

    bool was_recording = false;
    bool recording = false;

    while (g_running && gamepad.isConnected()) {
        gamepad.update();

        auto state = gamepad.getState();
        auto cmd = gamepad.getDriveCommand(config);

        // Handle recording toggle (edge detection)
        if (state.button_south && !was_recording) {
            recording = !recording;
            if (recording) {
                gamepad.setLEDColor(255, 0, 0);  // Red = recording
                gamepad.rumble(0.3f, 0.3f, 100);
                std::cout << "\n*** RECORDING STARTED ***\n";
            } else {
                gamepad.setLEDColor(0, 255, 0);  // Green = idle
                gamepad.rumble(0.3f, 0.3f, 100);
                std::cout << "\n*** RECORDING STOPPED ***\n";
            }
        }
        was_recording = state.button_south;

        // Clear line and print status
        std::cout << "\r";
        std::cout << std::fixed << std::setprecision(2);

        if (cmd.emergency_stop) {
            std::cout << ">>> EMERGENCY STOP (L1) <<<                              ";
            gamepad.setLEDColor(255, 255, 0);  // Yellow
        } else {
            std::cout << "Lin: " << std::setw(6) << cmd.linear_velocity << " m/s  "
                      << "Ang: " << std::setw(6) << cmd.angular_velocity << " rad/s  "
                      << "Speed: " << std::setw(3) << int(state.right_trigger * 100) << "%  ";

            // Show active buttons
            if (state.button_north) std::cout << "[Y] ";
            if (state.button_east) std::cout << "[B] ";
            if (state.button_west) std::cout << "[X] ";
            if (state.button_south) std::cout << "[A] ";
            if (state.dpad_up) std::cout << "[UP] ";
            if (state.dpad_down) std::cout << "[DN] ";

            std::cout << "     ";
        }
        std::cout << std::flush;

        std::this_thread::sleep_for(std::chrono::milliseconds(20));  // 50 Hz
    }

    std::cout << "\n\nShutting down...\n";
    gamepad.setLEDColor(0, 0, 0);  // Turn off LED
    gamepad.shutdown();

    return 0;
}
