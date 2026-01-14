#include "slam/gamepad.hpp"
#include <SDL.h>
#include <iostream>
#include <cmath>

namespace slam {

struct Gamepad::Impl {
    SDL_GameController* controller = nullptr;
    int controller_index = -1;
    bool sdl_initialized = false;
};

Gamepad::Gamepad() : impl_(std::make_unique<Impl>()) {}

Gamepad::~Gamepad() {
    shutdown();
}

bool Gamepad::init() {
    // Initialize SDL with gamepad subsystem
    if (SDL_Init(SDL_INIT_GAMECONTROLLER | SDL_INIT_HAPTIC) < 0) {
        std::cerr << "[Gamepad] SDL_Init failed: " << SDL_GetError() << "\n";
        return false;
    }
    impl_->sdl_initialized = true;

    // Enable PS5 DualSense support via HIDAPI
    SDL_SetHint(SDL_HINT_JOYSTICK_HIDAPI_PS5, "1");
    SDL_SetHint(SDL_HINT_JOYSTICK_HIDAPI_PS5_RUMBLE, "1");

    // Try to find and open a game controller
    int num_joysticks = SDL_NumJoysticks();
    std::cout << "[Gamepad] Found " << num_joysticks << " joystick(s)\n";

    for (int i = 0; i < num_joysticks; i++) {
        if (SDL_IsGameController(i)) {
            impl_->controller = SDL_GameControllerOpen(i);
            if (impl_->controller) {
                impl_->controller_index = i;
                const char* name = SDL_GameControllerName(impl_->controller);
                std::cout << "[Gamepad] Opened controller: " << (name ? name : "Unknown") << "\n";

                // Check if it's a PS5 controller
                SDL_GameControllerType type = SDL_GameControllerGetType(impl_->controller);
                if (type == SDL_CONTROLLER_TYPE_PS5) {
                    std::cout << "[Gamepad] PS5 DualSense detected!\n";
                    // Set a nice color
                    setLEDColor(0, 100, 255);  // Blue
                }
                state_.connected = true;
                return true;
            }
        }
    }

    std::cout << "[Gamepad] No game controller found. Plug in controller and restart.\n";
    return false;
}

void Gamepad::shutdown() {
    if (impl_->controller) {
        SDL_GameControllerClose(impl_->controller);
        impl_->controller = nullptr;
    }
    if (impl_->sdl_initialized) {
        SDL_Quit();
        impl_->sdl_initialized = false;
    }
    state_.connected = false;
}

void Gamepad::update() {
    if (!impl_->controller) {
        state_.connected = false;
        return;
    }

    // Process SDL events (required for controller state updates)
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_CONTROLLERDEVICEREMOVED:
                if (event.cdevice.which == impl_->controller_index) {
                    std::cout << "[Gamepad] Controller disconnected\n";
                    SDL_GameControllerClose(impl_->controller);
                    impl_->controller = nullptr;
                    state_.connected = false;
                    return;
                }
                break;
            case SDL_CONTROLLERDEVICEADDED:
                // Try to reconnect if we lost our controller
                if (!impl_->controller) {
                    impl_->controller = SDL_GameControllerOpen(event.cdevice.which);
                    if (impl_->controller) {
                        impl_->controller_index = event.cdevice.which;
                        std::cout << "[Gamepad] Controller reconnected: "
                                  << SDL_GameControllerName(impl_->controller) << "\n";
                        state_.connected = true;
                    }
                }
                break;
        }
    }

    if (!impl_->controller) return;

    // Read axis values (SDL returns -32768 to 32767)
    auto readAxis = [this](SDL_GameControllerAxis axis) -> float {
        int16_t value = SDL_GameControllerGetAxis(impl_->controller, axis);
        return value / 32767.0f;
    };

    // Left stick
    state_.left_stick_x = readAxis(SDL_CONTROLLER_AXIS_LEFTX);
    state_.left_stick_y = readAxis(SDL_CONTROLLER_AXIS_LEFTY);

    // Right stick
    state_.right_stick_x = readAxis(SDL_CONTROLLER_AXIS_RIGHTX);
    state_.right_stick_y = readAxis(SDL_CONTROLLER_AXIS_RIGHTY);

    // Triggers (0 to 32767, normalize to 0 to 1)
    state_.left_trigger = SDL_GameControllerGetAxis(impl_->controller, SDL_CONTROLLER_AXIS_TRIGGERLEFT) / 32767.0f;
    state_.right_trigger = SDL_GameControllerGetAxis(impl_->controller, SDL_CONTROLLER_AXIS_TRIGGERRIGHT) / 32767.0f;

    // Read button states
    auto readButton = [this](SDL_GameControllerButton button) -> bool {
        return SDL_GameControllerGetButton(impl_->controller, button) != 0;
    };

    // Face buttons
    state_.button_south = readButton(SDL_CONTROLLER_BUTTON_A);      // Cross
    state_.button_east = readButton(SDL_CONTROLLER_BUTTON_B);       // Circle
    state_.button_west = readButton(SDL_CONTROLLER_BUTTON_X);       // Square
    state_.button_north = readButton(SDL_CONTROLLER_BUTTON_Y);      // Triangle

    // D-pad
    state_.dpad_up = readButton(SDL_CONTROLLER_BUTTON_DPAD_UP);
    state_.dpad_down = readButton(SDL_CONTROLLER_BUTTON_DPAD_DOWN);
    state_.dpad_left = readButton(SDL_CONTROLLER_BUTTON_DPAD_LEFT);
    state_.dpad_right = readButton(SDL_CONTROLLER_BUTTON_DPAD_RIGHT);

    // Shoulder buttons
    state_.left_shoulder = readButton(SDL_CONTROLLER_BUTTON_LEFTSHOULDER);
    state_.right_shoulder = readButton(SDL_CONTROLLER_BUTTON_RIGHTSHOULDER);

    // Stick buttons
    state_.left_stick_button = readButton(SDL_CONTROLLER_BUTTON_LEFTSTICK);
    state_.right_stick_button = readButton(SDL_CONTROLLER_BUTTON_RIGHTSTICK);

    // System buttons
    state_.button_start = readButton(SDL_CONTROLLER_BUTTON_START);
    state_.button_select = readButton(SDL_CONTROLLER_BUTTON_BACK);
    state_.button_guide = readButton(SDL_CONTROLLER_BUTTON_GUIDE);

    // PS5-specific (mapped to misc buttons in SDL)
    state_.touchpad_button = readButton(SDL_CONTROLLER_BUTTON_TOUCHPAD);
    state_.mute_button = readButton(SDL_CONTROLLER_BUTTON_MISC1);

    state_.connected = true;
}

GamepadState Gamepad::getState() const {
    return state_;
}

float Gamepad::applyDeadzone(float value, float deadzone) const {
    if (std::abs(value) < deadzone) {
        return 0.0f;
    }
    // Scale remaining range to 0-1
    float sign = value > 0 ? 1.0f : -1.0f;
    return sign * (std::abs(value) - deadzone) / (1.0f - deadzone);
}

DriveCommand Gamepad::getDriveCommand(const GamepadDriveConfig& config) const {
    DriveCommand cmd;

    if (!state_.connected) {
        cmd.emergency_stop = true;
        return cmd;
    }

    // Emergency stop: L1 (left shoulder)
    cmd.emergency_stop = state_.left_shoulder;

    // Recording toggle: Cross button (button_south) - edge detection
    cmd.recording_toggle = state_.button_south && !prev_button_south_;
    // Note: prev_button_south_ is updated elsewhere or we need to make this non-const

    // Get raw input values
    float linear_input = 0.0f;
    float angular_input = 0.0f;

    if (config.left_stick_drive) {
        linear_input = -state_.left_stick_y;  // Negate because Y-up on stick = negative
    } else {
        linear_input = -state_.right_stick_y;
    }

    if (config.right_stick_steer) {
        angular_input = -state_.right_stick_x;  // Negate for natural steering
    } else {
        angular_input = -state_.left_stick_x;
    }

    // Apply deadzone
    linear_input = applyDeadzone(linear_input, config.stick_deadzone);
    angular_input = applyDeadzone(angular_input, config.stick_deadzone);

    // Apply inversion if configured
    if (config.invert_linear) linear_input = -linear_input;
    if (config.invert_angular) angular_input = -angular_input;

    // Calculate speed scale
    float speed_scale;
    if (config.use_trigger_boost) {
        // Old behavior: trigger modulates speed from min to max
        speed_scale = config.min_speed_scale +
                      state_.right_trigger * (1.0f - config.min_speed_scale);
    } else {
        // New behavior: direct mapping, 100% stick = max_speed_scale
        speed_scale = config.max_speed_scale;
    }

    // Calculate final velocities
    cmd.linear_velocity = linear_input * config.max_linear_velocity * speed_scale;
    cmd.angular_velocity = angular_input * config.max_angular_velocity * speed_scale;

    return cmd;
}

std::string Gamepad::getControllerName() const {
    if (impl_->controller) {
        const char* name = SDL_GameControllerName(impl_->controller);
        return name ? name : "Unknown";
    }
    return "Not connected";
}

bool Gamepad::isConnected() const {
    return state_.connected;
}

void Gamepad::rumble(float low_freq, float high_freq, uint32_t duration_ms) {
    if (!impl_->controller) return;

    // SDL rumble values are 0-65535
    uint16_t low = static_cast<uint16_t>(low_freq * 65535);
    uint16_t high = static_cast<uint16_t>(high_freq * 65535);

    SDL_GameControllerRumble(impl_->controller, low, high, duration_ms);
}

void Gamepad::setLEDColor(uint8_t r, uint8_t g, uint8_t b) {
    if (!impl_->controller) return;

    // This only works on PS5 DualSense (and some other controllers)
    SDL_GameControllerSetLED(impl_->controller, r, g, b);
}

int Gamepad::getBatteryLevel() const {
    if (!impl_->controller) return -1;

    // Get the underlying joystick
    SDL_Joystick* joystick = SDL_GameControllerGetJoystick(impl_->controller);
    if (!joystick) return -1;

    SDL_JoystickPowerLevel power = SDL_JoystickCurrentPowerLevel(joystick);

    switch (power) {
        case SDL_JOYSTICK_POWER_EMPTY:  return 5;
        case SDL_JOYSTICK_POWER_LOW:    return 20;
        case SDL_JOYSTICK_POWER_MEDIUM: return 50;
        case SDL_JOYSTICK_POWER_FULL:   return 100;
        case SDL_JOYSTICK_POWER_WIRED:  return 100;
        case SDL_JOYSTICK_POWER_MAX:    return 100;
        default:                        return -1;  // Unknown
    }
}

Gamepad::BatteryLevel Gamepad::getBatteryLevelEnum() const {
    if (!impl_->controller) return BatteryLevel::UNKNOWN;

    SDL_Joystick* joystick = SDL_GameControllerGetJoystick(impl_->controller);
    if (!joystick) return BatteryLevel::UNKNOWN;

    SDL_JoystickPowerLevel power = SDL_JoystickCurrentPowerLevel(joystick);

    switch (power) {
        case SDL_JOYSTICK_POWER_EMPTY:  return BatteryLevel::EMPTY;
        case SDL_JOYSTICK_POWER_LOW:    return BatteryLevel::LOW;
        case SDL_JOYSTICK_POWER_MEDIUM: return BatteryLevel::MEDIUM;
        case SDL_JOYSTICK_POWER_FULL:   return BatteryLevel::FULL;
        case SDL_JOYSTICK_POWER_WIRED:  return BatteryLevel::WIRED;
        case SDL_JOYSTICK_POWER_MAX:    return BatteryLevel::MAX;
        default:                        return BatteryLevel::UNKNOWN;
    }
}

} // namespace slam
