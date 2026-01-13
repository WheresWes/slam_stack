#pragma once
/**
 * @file shared_state.hpp
 * @brief Thread-safe shared state between worker thread and GUI
 */

#include <mutex>
#include <vector>
#include <string>
#include <atomic>
#include <chrono>

#include "state.hpp"

namespace slam_gui {

// 3D pose
struct Pose3D {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float roll = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
};

// 2D velocity
struct Velocity2D {
    float linear = 0.0f;   // m/s
    float angular = 0.0f;  // rad/s
};

// Point for rendering (position + color)
struct RenderPoint {
    float x, y, z;
    uint8_t r, g, b, a;
};

// VESC motor status
struct MotorStatus {
    int32_t erpm = 0;
    float duty = 0.0f;
    float current_motor = 0.0f;
    float current_input = 0.0f;
    float temp_fet = 0.0f;
    float temp_motor = 0.0f;
    float voltage_in = 0.0f;
    int32_t tachometer = 0;
    bool connected = false;
};

// LiDAR status
struct LidarStatus {
    ConnectionStatus connection = ConnectionStatus::DISCONNECTED;
    std::string ip_address;
    int point_rate = 0;      // points/sec
    int imu_rate = 0;        // Hz
    float gravity_mag = 0.0f; // m/s²
    bool imu_initialized = false;
};

// Calibration progress
struct CalibrationProgress {
    CalibrationPhase phase = CalibrationPhase::IDLE;
    float progress = 0.0f;   // 0.0 to 1.0
    std::string status_text;
    bool running = false;
    bool success = false;
};

// Relocalization progress
struct RelocalizationProgress {
    bool running = false;
    float progress = 0.0f;
    float confidence = 0.0f;
    std::string status_text;
    bool success = false;
};

// PAUT probe configuration
struct ProbeConfig {
    float offset_x = -0.15f;    // Behind robot center (meters)
    float offset_y = 0.0f;      // Lateral offset (meters)
    float width = 0.08f;        // Probe width (meters)
    float length = 0.02f;       // Probe length in travel direction (meters)
    bool enabled = true;
};

// Hull mesh info
struct HullMeshInfo {
    bool loaded = false;
    std::string filename;
    int vertex_count = 0;
    int triangle_count = 0;
    float coverage_percent = 0.0f;
    float covered_area_m2 = 0.0f;
    float total_area_m2 = 0.0f;
};

/**
 * Thread-safe shared state
 *
 * Worker thread writes, GUI thread reads.
 * All reads/writes go through mutex-protected accessors.
 */
class SharedState {
public:
    // ========== Atomic flags (no mutex needed) ==========

    std::atomic<bool> e_stop{false};
    std::atomic<bool> hardware_connected{false};
    std::atomic<bool> gamepad_connected{false};

    // ========== Application state ==========

    void setAppState(AppState state) {
        std::lock_guard<std::mutex> lock(mutex_);
        app_state_ = state;
    }

    AppState getAppState() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return app_state_;
    }

    // ========== Poses ==========

    void setSlamPose(const Pose3D& pose) {
        std::lock_guard<std::mutex> lock(mutex_);
        slam_pose_ = pose;
    }

    Pose3D getSlamPose() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slam_pose_;
    }

    void setFusedPose(const Pose3D& pose) {
        std::lock_guard<std::mutex> lock(mutex_);
        fused_pose_ = pose;
    }

    Pose3D getFusedPose() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return fused_pose_;
    }

    void setWheelOdomPose(const Pose3D& pose) {
        std::lock_guard<std::mutex> lock(mutex_);
        wheel_odom_pose_ = pose;
    }

    Pose3D getWheelOdomPose() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return wheel_odom_pose_;
    }

    // ========== Velocity ==========

    void setVelocity(const Velocity2D& vel) {
        std::lock_guard<std::mutex> lock(mutex_);
        velocity_ = vel;
    }

    Velocity2D getVelocity() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return velocity_;
    }

    // ========== Motion state ==========

    void setMotionState(MotionState state) {
        std::lock_guard<std::mutex> lock(mutex_);
        motion_state_ = state;
    }

    MotionState getMotionState() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return motion_state_;
    }

    // ========== Motor status ==========

    void setMotorStatus(int index, const MotorStatus& status) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= 0 && index < 2) {
            motor_status_[index] = status;
        }
    }

    MotorStatus getMotorStatus(int index) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= 0 && index < 2) {
            return motor_status_[index];
        }
        return MotorStatus{};
    }

    // ========== LiDAR status ==========

    void setLidarStatus(const LidarStatus& status) {
        std::lock_guard<std::mutex> lock(mutex_);
        lidar_status_ = status;
    }

    LidarStatus getLidarStatus() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return lidar_status_;
    }

    // ========== Map points (for 3D viewer) ==========

    void setMapPoints(const std::vector<RenderPoint>& points) {
        std::lock_guard<std::mutex> lock(mutex_);
        map_points_ = points;
        map_points_updated_ = true;
    }

    bool getMapPointsIfUpdated(std::vector<RenderPoint>& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (map_points_updated_) {
            out = map_points_;
            map_points_updated_ = false;
            return true;
        }
        return false;
    }

    size_t getMapPointCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return map_points_.size();
    }

    // ========== Current scan (for 3D viewer) ==========

    void setCurrentScan(const std::vector<RenderPoint>& points) {
        std::lock_guard<std::mutex> lock(mutex_);
        current_scan_ = points;
        current_scan_updated_ = true;
    }

    bool getCurrentScanIfUpdated(std::vector<RenderPoint>& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_scan_updated_) {
            out = current_scan_;
            current_scan_updated_ = false;
            return true;
        }
        return false;
    }

    // ========== Trajectory ==========

    void addTrajectoryPoint(const Pose3D& pose) {
        std::lock_guard<std::mutex> lock(mutex_);
        trajectory_.push_back(pose);
        // Keep last 1000 points
        if (trajectory_.size() > 1000) {
            trajectory_.erase(trajectory_.begin());
        }
    }

    std::vector<Pose3D> getTrajectory() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return trajectory_;
    }

    void clearTrajectory() {
        std::lock_guard<std::mutex> lock(mutex_);
        trajectory_.clear();
    }

    // ========== Calibration ==========

    void setCalibrationProgress(const CalibrationProgress& progress) {
        std::lock_guard<std::mutex> lock(mutex_);
        calibration_progress_ = progress;
    }

    CalibrationProgress getCalibrationProgress() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return calibration_progress_;
    }

    // ========== Relocalization ==========

    void setRelocalizationProgress(const RelocalizationProgress& progress) {
        std::lock_guard<std::mutex> lock(mutex_);
        relocalization_progress_ = progress;
    }

    RelocalizationProgress getRelocalizationProgress() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return relocalization_progress_;
    }

    // ========== Hull mesh ==========

    void setHullMeshInfo(const HullMeshInfo& info) {
        std::lock_guard<std::mutex> lock(mutex_);
        hull_mesh_info_ = info;
    }

    HullMeshInfo getHullMeshInfo() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return hull_mesh_info_;
    }

    // ========== Probe config ==========

    void setProbeConfig(const ProbeConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_);
        probe_config_ = config;
    }

    ProbeConfig getProbeConfig() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return probe_config_;
    }

    // ========== Recording ==========

    void setRecordingStatus(bool recording, const std::string& filename = "", float duration = 0.0f) {
        std::lock_guard<std::mutex> lock(mutex_);
        recording_ = recording;
        recording_filename_ = filename;
        recording_duration_ = duration;
    }

    bool isRecording() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return recording_;
    }

    std::string getRecordingFilename() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return recording_filename_;
    }

    float getRecordingDuration() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return recording_duration_;
    }

    // ========== Status messages ==========

    void setStatusMessage(const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        status_message_ = msg;
        status_time_ = std::chrono::steady_clock::now();
    }

    std::string getStatusMessage() const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto elapsed = std::chrono::steady_clock::now() - status_time_;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 5) {
            return "";  // Message expired
        }
        return status_message_;
    }

    // ========== Error messages ==========

    void setErrorMessage(const std::string& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        error_message_ = msg;
    }

    std::string getErrorMessage() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_message_;
    }

    void clearErrorMessage() {
        std::lock_guard<std::mutex> lock(mutex_);
        error_message_.clear();
    }

private:
    mutable std::mutex mutex_;

    // Application state
    AppState app_state_ = AppState::STARTUP;

    // Poses
    Pose3D slam_pose_;
    Pose3D fused_pose_;
    Pose3D wheel_odom_pose_;

    // Velocity
    Velocity2D velocity_;

    // Motion state
    MotionState motion_state_ = MotionState::STATIONARY;

    // Motor status
    MotorStatus motor_status_[2];

    // LiDAR status
    LidarStatus lidar_status_;

    // Point clouds for rendering
    std::vector<RenderPoint> map_points_;
    std::vector<RenderPoint> current_scan_;
    bool map_points_updated_ = false;
    bool current_scan_updated_ = false;

    // Trajectory
    std::vector<Pose3D> trajectory_;

    // Calibration
    CalibrationProgress calibration_progress_;

    // Relocalization
    RelocalizationProgress relocalization_progress_;

    // Hull mesh
    HullMeshInfo hull_mesh_info_;

    // Probe
    ProbeConfig probe_config_;

    // Recording
    bool recording_ = false;
    std::string recording_filename_;
    float recording_duration_ = 0.0f;

    // Status/error messages
    std::string status_message_;
    std::string error_message_;
    std::chrono::steady_clock::time_point status_time_;
};

} // namespace slam_gui
