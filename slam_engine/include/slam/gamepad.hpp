#pragma once
// Gamepad controller interface using SDL2
// Supports PS5 DualSense, Xbox, and other controllers

#include <string>
#include <functional>
#include <memory>

namespace slam {

// Normalized axis values: -1.0 to 1.0
// Trigger values: 0.0 to 1.0
struct GamepadState {
    // Left stick
    float left_stick_x = 0.0f;   // -1 = left, +1 = right
    float left_stick_y = 0.0f;   // -1 = up, +1 = down

    // Right stick
    float right_stick_x = 0.0f;  // -1 = left, +1 = right
    float right_stick_y = 0.0f;  // -1 = up, +1 = down

    // Triggers
    float left_trigger = 0.0f;   // 0 = released, 1 = fully pressed
    float right_trigger = 0.0f;  // 0 = released, 1 = fully pressed

    // D-pad
    bool dpad_up = false;
    bool dpad_down = false;
    bool dpad_left = false;
    bool dpad_right = false;

    // Face buttons (PS5: Cross/Circle/Square/Triangle, Xbox: A/B/X/Y)
    bool button_south = false;   // PS5: Cross, Xbox: A
    bool button_east = false;    // PS5: Circle, Xbox: B
    bool button_west = false;    // PS5: Square, Xbox: X
    bool button_north = false;   // PS5: Triangle, Xbox: Y

    // Shoulder buttons
    bool left_shoulder = false;  // L1 / LB
    bool right_shoulder = false; // R1 / RB

    // Stick buttons
    bool left_stick_button = false;  // L3
    bool right_stick_button = false; // R3

    // System buttons
    bool button_start = false;   // Options / Start
    bool button_select = false;  // Create / Back
    bool button_guide = false;   // PS / Xbox button

    // PS5-specific
    bool touchpad_button = false;
    bool mute_button = false;

    // Connection status
    bool connected = false;
};

// Robot drive commands derived from gamepad state
struct DriveCommand {
    float linear_velocity = 0.0f;   // m/s, positive = forward
    float angular_velocity = 0.0f;  // rad/s, positive = counter-clockwise
    bool emergency_stop = false;
    bool recording_toggle = false;  // True when button just pressed
};

// Configuration for mapping gamepad to drive commands
struct GamepadDriveConfig {
    // Maximum velocities
    float max_linear_velocity = 0.15f;   // m/s (conservative default)
    float max_angular_velocity = 1.0f;   // rad/s

    // Deadzone for sticks (0.0 to 1.0)
    float stick_deadzone = 0.15f;

    // Speed scaling
    float max_speed_scale = 0.8f;   // 100% stick = this fraction of max velocity
    bool use_trigger_boost = false; // If true, trigger modulates speed (old behavior)
    float min_speed_scale = 0.3f;   // Only used if use_trigger_boost = true

    // Which stick controls what
    bool left_stick_drive = true;   // If true, left stick Y = linear velocity
    bool right_stick_steer = true;  // If true, right stick X = angular velocity
                                    // If false, left stick X = angular velocity (single-stick mode)

    // Invert axes if needed
    bool invert_linear = false;     // Invert forward/backward
    bool invert_angular = false;    // Invert left/right steering
};

class Gamepad {
public:
    Gamepad();
    ~Gamepad();

    // Initialize SDL and find controllers
    bool init();

    // Shutdown SDL
    void shutdown();

    // Poll for events and update state
    // Call this regularly (e.g., at 50Hz)
    void update();

    // Get current state
    GamepadState getState() const;

    // Convert gamepad state to drive command
    DriveCommand getDriveCommand(const GamepadDriveConfig& config = GamepadDriveConfig()) const;

    // Get controller info
    std::string getControllerName() const;
    bool isConnected() const;

    // Rumble feedback (0.0 to 1.0, duration in ms)
    void rumble(float low_freq, float high_freq, uint32_t duration_ms);

    // LED color (PS5 DualSense only, 0-255 each)
    void setLEDColor(uint8_t r, uint8_t g, uint8_t b);

    // Battery level (-1 = unknown, 0-100 = percentage)
    // Note: Not all controllers report battery level
    int getBatteryLevel() const;

    // Battery level enum (from SDL)
    enum class BatteryLevel {
        UNKNOWN = -1,
        EMPTY = 0,
        LOW = 25,
        MEDIUM = 50,
        FULL = 100,
        WIRED = 100,  // Wired = effectively full
        MAX = 100
    };

    // Get battery level as enum (more reliable than percentage)
    BatteryLevel getBatteryLevelEnum() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    GamepadState state_;
    bool prev_button_south_ = false;  // For edge detection

    float applyDeadzone(float value, float deadzone) const;
};

} // namespace slam
