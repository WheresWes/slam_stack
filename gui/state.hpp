#pragma once
/**
 * @file state.hpp
 * @brief Application state machine for SLAM Control GUI
 */

#include <string>
#include <atomic>

namespace slam_gui {

// Application operational states
enum class AppState {
    STARTUP,           // Initializing hardware
    IDLE,              // Ready, no operation in progress
    OPERATING,         // Manual drive only, no SLAM
    CALIBRATING,       // Running calibration sequence
    MAPPING,           // Building map (SLAM mode)
    SAVE_MAP,          // Saving map to file
    MAP_LOADED,        // Pre-built map loaded, ready for localization
    RELOCALIZING,      // Running global relocalization
    RELOCALIZE_FAILED, // Relocalization failed, can retry
    LOCALIZED,         // Successfully localized, tracking pose
    E_STOPPED          // Emergency stop activated
};

// Get human-readable state name
inline const char* getStateName(AppState state) {
    switch (state) {
        case AppState::STARTUP:           return "Starting Up";
        case AppState::IDLE:              return "Idle";
        case AppState::OPERATING:         return "Operating";
        case AppState::CALIBRATING:       return "Calibrating";
        case AppState::MAPPING:           return "Mapping";
        case AppState::SAVE_MAP:          return "Saving Map";
        case AppState::MAP_LOADED:        return "Map Loaded";
        case AppState::RELOCALIZING:      return "Relocalizing";
        case AppState::RELOCALIZE_FAILED: return "Relocalization Failed";
        case AppState::LOCALIZED:         return "Localized";
        case AppState::E_STOPPED:         return "E-STOP";
        default:                          return "Unknown";
    }
}

// Get state color for UI (R, G, B, A)
inline void getStateColor(AppState state, float& r, float& g, float& b, float& a) {
    a = 1.0f;
    switch (state) {
        case AppState::STARTUP:
            r = 0.5f; g = 0.5f; b = 0.5f; break;
        case AppState::IDLE:
            r = 0.4f; g = 0.4f; b = 0.4f; break;
        case AppState::OPERATING:
            r = 0.4f; g = 0.8f; b = 1.0f; break;  // Cyan = operating (drive only)
        case AppState::CALIBRATING:
            r = 1.0f; g = 0.7f; b = 0.2f; break;  // Orange
        case AppState::MAPPING:
            r = 0.2f; g = 0.8f; b = 0.2f; break;  // Green
        case AppState::SAVE_MAP:
            r = 0.2f; g = 0.6f; b = 1.0f; break;  // Blue
        case AppState::MAP_LOADED:
            r = 0.4f; g = 0.6f; b = 0.8f; break;  // Light blue
        case AppState::RELOCALIZING:
            r = 1.0f; g = 1.0f; b = 0.2f; break;  // Yellow
        case AppState::RELOCALIZE_FAILED:
            r = 1.0f; g = 0.4f; b = 0.1f; break;  // Orange-red
        case AppState::LOCALIZED:
            r = 0.2f; g = 1.0f; b = 0.5f; break;  // Bright green
        case AppState::E_STOPPED:
            r = 1.0f; g = 0.1f; b = 0.1f; break;  // Red
        default:
            r = 0.5f; g = 0.5f; b = 0.5f; break;
    }
}

// Motion state from sensor fusion
enum class MotionState {
    STATIONARY,
    STRAIGHT_LINE,
    TURNING
};

inline const char* getMotionStateName(MotionState state) {
    switch (state) {
        case MotionState::STATIONARY:   return "STATIONARY";
        case MotionState::STRAIGHT_LINE: return "STRAIGHT";
        case MotionState::TURNING:      return "TURNING";
        default:                        return "UNKNOWN";
    }
}

// Camera modes for 3D viewer
enum class CameraMode {
    FREE,           // Orbit/pan/zoom freely
    FOLLOW,         // Chase camera behind robot
    TOP_DOWN,       // Orthographic from above, follows robot
    TOP_DOWN_FREE   // Orthographic from above, free pan
};

inline const char* getCameraModeName(CameraMode mode) {
    switch (mode) {
        case CameraMode::FREE:          return "Free";
        case CameraMode::FOLLOW:        return "Follow";
        case CameraMode::TOP_DOWN:      return "Top-Down";
        case CameraMode::TOP_DOWN_FREE: return "Top-Down Free";
        default:                        return "Unknown";
    }
}

// Connection status
enum class ConnectionStatus {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    CONNECTION_ERROR  // Note: ERROR is a Windows macro, can't use it
};

inline const char* getConnectionStatusName(ConnectionStatus status) {
    switch (status) {
        case ConnectionStatus::DISCONNECTED:      return "Disconnected";
        case ConnectionStatus::CONNECTING:        return "Connecting";
        case ConnectionStatus::CONNECTED:         return "Connected";
        case ConnectionStatus::CONNECTION_ERROR:  return "Error";
        default:                                  return "Unknown";
    }
}

// Calibration phase
enum class CalibrationPhase {
    IDLE,
    FORWARD_SWEEP,
    REVERSE_SWEEP,
    ROTATION_SWEEP,
    TURNING_MIN_DUTY,
    COMPLETE
};

inline const char* getCalibrationPhaseName(CalibrationPhase phase) {
    switch (phase) {
        case CalibrationPhase::IDLE:            return "Idle";
        case CalibrationPhase::FORWARD_SWEEP:   return "Forward Sweep";
        case CalibrationPhase::REVERSE_SWEEP:   return "Reverse Sweep";
        case CalibrationPhase::ROTATION_SWEEP:  return "Rotation Sweep";
        case CalibrationPhase::TURNING_MIN_DUTY: return "Turning Min Duty";
        case CalibrationPhase::COMPLETE:        return "Complete";
        default:                                return "Unknown";
    }
}

} // namespace slam_gui
