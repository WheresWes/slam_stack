#pragma once
/**
 * @file command_queue.hpp
 * @brief Thread-safe command queue for GUI → Worker thread communication
 */

#include <queue>
#include <mutex>
#include <string>
#include <variant>
#include <optional>

namespace slam_gui {

// Command types from GUI to worker thread
enum class CommandType {
    // System
    CONNECT_HARDWARE,
    DISCONNECT_HARDWARE,

    // Calibration
    RUN_CALIBRATION,
    CANCEL_CALIBRATION,
    LOAD_CALIBRATION,
    SAVE_CALIBRATION,

    // Mapping
    START_MAPPING,
    STOP_MAPPING,
    SAVE_MAP,
    CLEAR_MAP,

    // Localization
    LOAD_MAP,
    START_LOCALIZATION,
    STOP_LOCALIZATION,
    RELOCALIZE,
    RUN_GLOBAL_LOCALIZATION,  // Run ICP when coverage is sufficient
    CANCEL_GLOBAL_LOCALIZATION,
    SET_POSE_HINT,

    // Hull mesh
    LOAD_HULL_MESH,
    CLEAR_COVERAGE,
    EXPORT_COVERAGE,

    // Manual control
    SET_VELOCITY,
    START_OPERATING,   // Drive-only mode (no SLAM)
    STOP_OPERATING,
    SET_MANUAL_MODE,
    SET_AUTO_MODE,

    // Safety
    E_STOP,
    RESET_E_STOP,

    // Recording
    START_RECORDING,
    STOP_RECORDING
};

// Command payload structures
struct VelocityCommand {
    float linear_mps = 0.0f;   // m/s forward
    float angular_radps = 0.0f; // rad/s (positive = left)
};

struct PoseHintCommand {
    float x = 0.0f;
    float y = 0.0f;
    float theta_deg = 0.0f;
};

struct FileCommand {
    std::string path;
};

// Command with payload
struct Command {
    CommandType type;
    std::variant<
        std::monostate,      // No payload
        VelocityCommand,
        PoseHintCommand,
        FileCommand
    > payload;

    // Convenience constructors
    static Command simple(CommandType type) {
        return Command{type, std::monostate{}};
    }

    static Command velocity(float linear, float angular) {
        return Command{CommandType::SET_VELOCITY, VelocityCommand{linear, angular}};
    }

    static Command poseHint(float x, float y, float theta_deg) {
        return Command{CommandType::SET_POSE_HINT, PoseHintCommand{x, y, theta_deg}};
    }

    static Command file(CommandType type, const std::string& path) {
        return Command{type, FileCommand{path}};
    }
};

/**
 * Thread-safe command queue (MPSC - multiple producer, single consumer)
 */
class CommandQueue {
public:
    void push(const Command& cmd) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(cmd);
    }

    std::optional<Command> pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        Command cmd = queue_.front();
        queue_.pop();
        return cmd;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) queue_.pop();
    }

private:
    mutable std::mutex mutex_;
    std::queue<Command> queue_;
};

} // namespace slam_gui
