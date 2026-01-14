#pragma once
/**
 * @file diagnostic_logger.hpp
 * @brief Comprehensive diagnostic logging for debugging SLAM performance and flyaways
 *
 * Captures:
 * - Raw hardware data (LiDAR points, IMU, VESC telemetry)
 * - SLAM internals (pose, timing, IKF state, map stats)
 * - System metrics (buffer sizes, thread timing, memory)
 * - Events (state changes, commands, errors)
 *
 * Binary format for high-rate data, with companion text event log.
 */

#include <fstream>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <queue>
#include <condition_variable>
#include <functional>
#include <iomanip>

namespace slam_gui {

//==============================================================================
// Log Message Types
//==============================================================================
enum class LogMsgType : uint8_t {
    // Hardware (0x01-0x0F)
    IMU_SAMPLE      = 0x01,  // Raw IMU data
    POINT_CLOUD     = 0x02,  // LiDAR point cloud (downsampled)
    VESC_STATUS     = 0x03,  // Motor controller status
    GAMEPAD_INPUT   = 0x04,  // Gamepad axis/button state
    LIDAR_STATUS    = 0x05,  // LiDAR connection/rate status

    // SLAM (0x10-0x1F)
    SLAM_POSE       = 0x10,  // Raw SLAM pose output
    SLAM_TIMING     = 0x11,  // SLAM processing breakdown
    SLAM_IKF_STATE  = 0x12,  // IKF internal state (covariance, iterations)
    SLAM_MAP_STATS  = 0x13,  // Map statistics (size, tree depth, etc.)
    FUSED_POSE      = 0x14,  // Sensor-fused pose
    WHEEL_ODOM      = 0x15,  // Wheel odometry pose

    // Performance (0x20-0x2F)
    BUFFER_STATUS   = 0x20,  // Buffer sizes (IMU, scans, commands)
    THREAD_TIMING   = 0x21,  // Thread loop timing
    MEMORY_STATUS   = 0x22,  // Memory usage
    FRAME_DROP      = 0x23,  // Dropped frame notification

    // Events (0x30-0x3F)
    STATE_CHANGE    = 0x30,  // Application state transition
    COMMAND_RECV    = 0x31,  // Command received
    COMMAND_EXEC    = 0x32,  // Command executed
    ERROR_EVENT     = 0x33,  // Error occurred
    WARNING_EVENT   = 0x34,  // Warning issued
    INFO_EVENT      = 0x35,  // Info message
    MARKER          = 0x36,  // User-placed marker for analysis

    // Calibration (0x40-0x4F)
    CALIB_PROGRESS  = 0x40,  // Calibration progress
    CALIB_RESULT    = 0x41,  // Calibration result

    // Localization (0x50-0x5F)
    RELOC_PROGRESS  = 0x50,  // Relocalization progress
    RELOC_ICP       = 0x51,  // ICP matching details
    RELOC_RESULT    = 0x52,  // Relocalization result
};

//==============================================================================
// Log Data Structures (packed for binary efficiency)
//==============================================================================
#pragma pack(push, 1)

struct LogHeader {
    char magic[4] = {'S', 'L', 'O', 'G'};  // "SLOG" = SLAM LOG
    uint32_t version = 1;
    uint64_t start_time_ns;      // System time at log start
    uint32_t flags;              // Logging options enabled
    char session_id[32];         // Unique session identifier
};

struct LogMessageHeader {
    LogMsgType type;
    uint64_t timestamp_ns;       // Nanoseconds since log start
    uint32_t payload_size;
};

// Hardware messages
struct LogImuSample {
    float acc_x, acc_y, acc_z;   // m/s² (already converted)
    float gyr_x, gyr_y, gyr_z;   // rad/s
    float gravity_mag;           // Computed gravity magnitude
    uint8_t initialized;         // IMU initialization status
};

struct LogPointCloudHeader {
    uint32_t point_count;
    uint32_t original_count;     // Before downsampling
    float downsample_ratio;      // For reference
};

struct LogPoint {
    float x, y, z;
    uint8_t intensity;
    float time_offset_us;        // Time offset within scan
};

struct LogVescStatus {
    uint8_t motor_id;            // 0=left, 1=right
    int32_t erpm;
    float duty;
    float current_motor;
    float current_input;
    float temp_fet;
    float temp_motor;
    float voltage_in;
    int32_t tachometer;
    uint8_t connected;
};

struct LogGamepadInput {
    float left_x, left_y;        // Left stick
    float right_x, right_y;      // Right stick
    float left_trigger, right_trigger;
    uint32_t buttons;            // Button bitmask
    uint8_t connected;
};

struct LogLidarStatus {
    uint8_t connection_status;   // ConnectionStatus enum
    int32_t point_rate;          // Points per second
    int32_t imu_rate;            // IMU Hz
    float gravity_mag;
    uint8_t imu_initialized;
};

// SLAM messages
struct LogSlamPose {
    float x, y, z;
    float roll, pitch, yaw;
    float vx, vy, vz;            // Linear velocity
    float wx, wy, wz;            // Angular velocity
    uint8_t valid;               // Pose validity flag
};

struct LogSlamTiming {
    float total_ms;              // Total process() time
    float imu_process_ms;        // IMU processing time
    float undistort_ms;          // Point cloud undistortion
    float downsample_ms;         // Voxel downsampling
    float ikf_update_ms;         // IKF update (main matching)
    float map_update_ms;         // Map incremental update
    uint32_t points_input;       // Points received
    uint32_t points_after_filter; // After point filter
    uint32_t points_after_ds;    // After downsampling
    uint32_t points_matched;     // Points used in IKF
};

struct LogSlamIkfState {
    // Covariance diagonal (position, rotation, velocity, bias)
    float cov_pos[3];            // Position uncertainty
    float cov_rot[3];            // Rotation uncertainty
    float cov_vel[3];            // Velocity uncertainty
    float cov_bg[3];             // Gyro bias uncertainty
    float cov_ba[3];             // Accel bias uncertainty
    uint32_t iterations;         // IKF iterations this update
    uint8_t converged;           // Convergence flag
    float residual_mean;         // Mean point-to-plane residual
    float residual_max;          // Max residual
};

struct LogSlamMapStats {
    uint32_t map_points;         // Total points in map
    uint32_t tree_nodes;         // ikd-tree nodes
    uint32_t tree_depth;         // ikd-tree max depth
    float map_extent[6];         // min/max x,y,z
    uint32_t points_added;       // Points added this update
    uint32_t points_deleted;     // Points deleted (box removal)
};

struct LogFusedPose {
    float x, y, z;
    float roll, pitch, yaw;
    float linear_vel;            // Fused linear velocity
    float angular_vel;           // Fused angular velocity
    uint8_t motion_state;        // MotionState enum
    float slam_weight;           // SLAM contribution (0-1)
    float odom_weight;           // Odometry contribution (0-1)
};

struct LogWheelOdom {
    float x, y, theta;           // Odometry pose
    float linear_vel;            // m/s
    float angular_vel;           // rad/s
    int32_t tach_left;           // Left tachometer
    int32_t tach_right;          // Right tachometer
    float distance_left;         // Left wheel distance
    float distance_right;        // Right wheel distance
};

// Performance messages
struct LogBufferStatus {
    uint32_t imu_buffer;         // IMU samples waiting
    uint32_t scan_buffer;        // Scans waiting
    uint32_t command_buffer;     // Commands waiting
    uint32_t log_buffer;         // Log messages waiting to write
};

struct LogThreadTiming {
    uint8_t thread_id;           // 0=worker, 1=render, 2=logger
    float loop_time_ms;          // Time for one loop iteration
    float sleep_time_ms;         // Time spent sleeping
    float busy_ratio;            // busy_time / total_time
};

struct LogMemoryStatus {
    uint64_t process_memory_mb;  // Process memory usage
    uint64_t map_memory_mb;      // Estimated map memory
    uint64_t peak_memory_mb;     // Peak memory usage
};

struct LogFrameDrop {
    uint8_t source;              // 0=LiDAR, 1=IMU, 2=SLAM
    uint32_t expected_seq;
    uint32_t received_seq;
    uint32_t frames_dropped;
};

// Event messages (variable length - use strings)
struct LogEventHeader {
    uint8_t severity;            // 0=info, 1=warning, 2=error
    uint16_t source_len;
    uint16_t message_len;
    // Followed by: source string, message string
};

struct LogStateChange {
    uint8_t old_state;
    uint8_t new_state;
    uint16_t reason_len;
    // Followed by: reason string
};

struct LogCommand {
    uint8_t command_type;        // CommandType enum
    uint8_t executed;            // 0=received, 1=executed, 2=failed
    uint16_t detail_len;
    // Followed by: detail string
};

struct LogMarker {
    uint32_t marker_id;
    uint16_t label_len;
    // Followed by: label string
};

// Relocalization messages
struct LogRelocProgress {
    uint8_t status;              // LocalizationStatus enum
    float progress;
    int32_t local_map_voxels;
    int32_t local_map_points;
    float rotation_deg;
    float distance_m;
    uint8_t attempt_number;
};

struct LogRelocIcp {
    uint8_t stage;               // 0=coarse, 1=medium, 2=fine
    uint32_t iteration;
    uint32_t hypotheses_total;
    uint32_t hypotheses_kept;
    float current_fitness;
    float best_fitness;
    float transform[16];         // 4x4 transform matrix
};

#pragma pack(pop)

//==============================================================================
// Logging Configuration
//==============================================================================
struct LoggingConfig {
    // What to log (flags)
    bool log_imu = true;
    bool log_points = true;
    bool log_vesc = true;
    bool log_gamepad = true;
    bool log_slam_pose = true;
    bool log_slam_timing = true;
    bool log_slam_ikf = true;
    bool log_slam_map = true;
    bool log_fused_pose = true;
    bool log_wheel_odom = true;
    bool log_buffers = true;
    bool log_threads = true;
    bool log_memory = true;
    bool log_events = true;
    bool log_reloc = true;

    // Downsampling (to reduce file size)
    int point_downsample = 10;   // Log every Nth point (1=all)
    int imu_downsample = 1;      // Log every Nth IMU sample
    int buffer_log_hz = 10;      // Buffer status log rate

    // File options
    std::string output_dir = "logs";
    bool compress = false;        // Future: gzip compression
    size_t max_file_size_mb = 500; // Split files if exceeded
    bool write_text_events = true; // Also write events.log text file
};

//==============================================================================
// Diagnostic Logger Class
//==============================================================================
class DiagnosticLogger {
public:
    DiagnosticLogger() = default;
    ~DiagnosticLogger() { stop(); }

    // Start/stop logging
    bool start(const std::string& session_name, const LoggingConfig& config = {});
    void stop();
    bool isRunning() const { return running_.load(); }

    // Get session info
    std::string getSessionName() const { return session_name_; }
    std::string getFilePath() const { return file_path_; }
    float getElapsedSeconds() const;
    size_t getFileSizeMB() const;
    size_t getMessageCount() const { return message_count_.load(); }
    size_t getDroppedCount() const { return dropped_count_.load(); }

    // Place a marker for analysis
    void placeMarker(const std::string& label);

    //==========================================================================
    // Logging Functions (thread-safe, non-blocking)
    //==========================================================================

    // Hardware
    void logImu(float ax, float ay, float az, float gx, float gy, float gz,
                float gravity_mag, bool initialized);

    void logPointCloud(const std::vector<float>& points_xyzit, size_t original_count);

    void logVescStatus(int motor_id, int32_t erpm, float duty,
                       float current_motor, float current_input,
                       float temp_fet, float temp_motor, float voltage_in,
                       int32_t tachometer, bool connected);

    void logGamepad(float lx, float ly, float rx, float ry,
                    float lt, float rt, uint32_t buttons, bool connected);

    void logLidarStatus(int connection, int point_rate, int imu_rate,
                        float gravity_mag, bool imu_init);

    // SLAM
    void logSlamPose(float x, float y, float z,
                     float roll, float pitch, float yaw,
                     float vx, float vy, float vz,
                     float wx, float wy, float wz, bool valid);

    void logSlamTiming(float total_ms, float imu_ms, float undistort_ms,
                       float downsample_ms, float ikf_ms, float map_ms,
                       uint32_t pts_in, uint32_t pts_filter, uint32_t pts_ds, uint32_t pts_match);

    void logSlamIkfState(const float* cov_pos, const float* cov_rot,
                         const float* cov_vel, const float* cov_bg, const float* cov_ba,
                         uint32_t iterations, bool converged,
                         float residual_mean, float residual_max);

    void logSlamMapStats(uint32_t map_points, uint32_t tree_nodes, uint32_t tree_depth,
                         const float* extent, uint32_t added, uint32_t deleted);

    // Fusion
    void logFusedPose(float x, float y, float z,
                      float roll, float pitch, float yaw,
                      float linear_vel, float angular_vel,
                      int motion_state, float slam_weight, float odom_weight);

    void logWheelOdom(float x, float y, float theta,
                      float linear_vel, float angular_vel,
                      int32_t tach_l, int32_t tach_r,
                      float dist_l, float dist_r);

    // Performance
    void logBufferStatus(uint32_t imu, uint32_t scans, uint32_t commands);
    void logThreadTiming(int thread_id, float loop_ms, float sleep_ms, float busy_ratio);
    void logMemoryStatus(uint64_t process_mb, uint64_t map_mb, uint64_t peak_mb);
    void logFrameDrop(int source, uint32_t expected, uint32_t received, uint32_t dropped);

    // Events
    void logEvent(int severity, const std::string& source, const std::string& message);
    void logStateChange(int old_state, int new_state, const std::string& reason);
    void logCommand(int command_type, int executed, const std::string& detail);

    // Relocalization
    void logRelocProgress(int status, float progress, int voxels, int points,
                          float rotation, float distance, int attempt);
    void logRelocIcp(int stage, uint32_t iteration, uint32_t hyp_total, uint32_t hyp_kept,
                     float fitness, float best_fitness, const float* transform);

private:
    // Get timestamp in nanoseconds since log start
    uint64_t getTimestampNs() const;

    // Queue a message for async writing
    void queueMessage(LogMsgType type, const void* data, size_t size);

    // Writer thread function
    void writerThread();

    // Write to text event log
    void writeTextEvent(const std::string& severity, const std::string& source,
                        const std::string& message);

    std::atomic<bool> running_{false};
    std::string session_name_;
    std::string file_path_;
    std::string text_log_path_;
    LoggingConfig config_;

    // Timing
    std::chrono::steady_clock::time_point start_time_;

    // File output
    std::ofstream binary_file_;
    std::ofstream text_file_;
    std::mutex file_mutex_;

    // Async write queue
    struct QueuedMessage {
        std::vector<uint8_t> data;
    };
    std::queue<QueuedMessage> write_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread writer_thread_;

    // Statistics
    std::atomic<size_t> message_count_{0};
    std::atomic<size_t> dropped_count_{0};
    std::atomic<size_t> bytes_written_{0};

    // Downsampling counters
    std::atomic<int> imu_counter_{0};
    std::atomic<int> point_counter_{0};

    static constexpr size_t MAX_QUEUE_SIZE = 10000;
};

//==============================================================================
// Implementation
//==============================================================================

inline bool DiagnosticLogger::start(const std::string& session_name, const LoggingConfig& config) {
    if (running_.load()) {
        return false;  // Already running
    }

    config_ = config;
    session_name_ = session_name;

    // Create output directory if needed
    std::string dir = config_.output_dir;
    // CreateDirectoryA(dir.c_str(), nullptr);  // Windows-specific

    // Generate filename with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    char time_str[32];
    std::strftime(time_str, sizeof(time_str), "%Y%m%d_%H%M%S", std::localtime(&time_t));

    file_path_ = dir + "/" + session_name + "_" + time_str + ".slog";
    text_log_path_ = dir + "/" + session_name + "_" + time_str + "_events.log";

    // Open binary file
    binary_file_.open(file_path_, std::ios::binary);
    if (!binary_file_.is_open()) {
        return false;
    }

    // Write header
    LogHeader header;
    header.start_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
    header.flags = 0;  // Reserved for future use
    std::strncpy(header.session_id, session_name.c_str(), sizeof(header.session_id) - 1);
    binary_file_.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Open text event log
    if (config_.write_text_events) {
        text_file_.open(text_log_path_);
        if (text_file_.is_open()) {
            text_file_ << "# SLAM Diagnostic Event Log\n";
            text_file_ << "# Session: " << session_name << "\n";
            text_file_ << "# Started: " << time_str << "\n";
            text_file_ << "# Format: [timestamp_s] [severity] [source] message\n";
            text_file_ << "#\n";
            text_file_.flush();
        }
    }

    start_time_ = std::chrono::steady_clock::now();
    message_count_.store(0);
    dropped_count_.store(0);
    bytes_written_.store(sizeof(header));
    imu_counter_.store(0);
    point_counter_.store(0);

    running_.store(true);

    // Start writer thread
    writer_thread_ = std::thread(&DiagnosticLogger::writerThread, this);

    logEvent(0, "Logger", "Diagnostic logging started: " + file_path_);
    return true;
}

inline void DiagnosticLogger::stop() {
    if (!running_.load()) {
        return;
    }

    logEvent(0, "Logger", "Diagnostic logging stopped. Messages: " +
             std::to_string(message_count_.load()) + ", Dropped: " +
             std::to_string(dropped_count_.load()));

    running_.store(false);

    // Wake up writer thread
    queue_cv_.notify_one();

    // Wait for writer to finish
    if (writer_thread_.joinable()) {
        writer_thread_.join();
    }

    // Close files
    if (binary_file_.is_open()) {
        binary_file_.close();
    }
    if (text_file_.is_open()) {
        text_file_.close();
    }
}

inline uint64_t DiagnosticLogger::getTimestampNs() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time_).count();
}

inline float DiagnosticLogger::getElapsedSeconds() const {
    return getTimestampNs() / 1e9f;
}

inline size_t DiagnosticLogger::getFileSizeMB() const {
    return bytes_written_.load() / (1024 * 1024);
}

inline void DiagnosticLogger::queueMessage(LogMsgType type, const void* data, size_t size) {
    if (!running_.load()) return;

    // Build message with header
    LogMessageHeader hdr;
    hdr.type = type;
    hdr.timestamp_ns = getTimestampNs();
    hdr.payload_size = static_cast<uint32_t>(size);

    QueuedMessage msg;
    msg.data.resize(sizeof(hdr) + size);
    std::memcpy(msg.data.data(), &hdr, sizeof(hdr));
    if (size > 0 && data) {
        std::memcpy(msg.data.data() + sizeof(hdr), data, size);
    }

    // Try to queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (write_queue_.size() < MAX_QUEUE_SIZE) {
            write_queue_.push(std::move(msg));
            message_count_++;
        } else {
            dropped_count_++;
        }
    }
    queue_cv_.notify_one();
}

inline void DiagnosticLogger::writerThread() {
    std::vector<QueuedMessage> batch;
    batch.reserve(100);

    while (running_.load() || !write_queue_.empty()) {
        // Wait for messages
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return !write_queue_.empty() || !running_.load();
            });

            // Grab batch of messages
            while (!write_queue_.empty() && batch.size() < 100) {
                batch.push_back(std::move(write_queue_.front()));
                write_queue_.pop();
            }
        }

        // Write batch to file
        if (!batch.empty()) {
            std::lock_guard<std::mutex> lock(file_mutex_);
            for (const auto& msg : batch) {
                binary_file_.write(reinterpret_cast<const char*>(msg.data.data()),
                                   msg.data.size());
                bytes_written_ += msg.data.size();
            }
            binary_file_.flush();
            batch.clear();
        }
    }
}

inline void DiagnosticLogger::writeTextEvent(const std::string& severity,
                                              const std::string& source,
                                              const std::string& message) {
    if (!config_.write_text_events || !text_file_.is_open()) return;

    float elapsed = getElapsedSeconds();
    std::lock_guard<std::mutex> lock(file_mutex_);
    text_file_ << "[" << std::fixed << std::setprecision(3) << elapsed << "] "
               << "[" << severity << "] [" << source << "] " << message << "\n";
    text_file_.flush();
}

inline void DiagnosticLogger::placeMarker(const std::string& label) {
    static uint32_t marker_id = 0;

    std::vector<uint8_t> data;
    LogMarker marker;
    marker.marker_id = marker_id++;
    marker.label_len = static_cast<uint16_t>(label.size());

    data.resize(sizeof(marker) + label.size());
    std::memcpy(data.data(), &marker, sizeof(marker));
    std::memcpy(data.data() + sizeof(marker), label.data(), label.size());

    queueMessage(LogMsgType::MARKER, data.data(), data.size());
    writeTextEvent("MARKER", "User", label);
}

//==============================================================================
// Hardware Logging
//==============================================================================

inline void DiagnosticLogger::logImu(float ax, float ay, float az,
                                      float gx, float gy, float gz,
                                      float gravity_mag, bool initialized) {
    if (!config_.log_imu) return;

    // Downsampling
    int count = imu_counter_.fetch_add(1);
    if (count % config_.imu_downsample != 0) return;

    LogImuSample sample;
    sample.acc_x = ax; sample.acc_y = ay; sample.acc_z = az;
    sample.gyr_x = gx; sample.gyr_y = gy; sample.gyr_z = gz;
    sample.gravity_mag = gravity_mag;
    sample.initialized = initialized ? 1 : 0;

    queueMessage(LogMsgType::IMU_SAMPLE, &sample, sizeof(sample));
}

inline void DiagnosticLogger::logPointCloud(const std::vector<float>& points_xyzit,
                                             size_t original_count) {
    if (!config_.log_points) return;

    // Points are stored as: x, y, z, intensity, time_offset (5 floats per point)
    size_t num_points = points_xyzit.size() / 5;
    size_t logged_count = (num_points + config_.point_downsample - 1) / config_.point_downsample;

    std::vector<uint8_t> data;
    LogPointCloudHeader hdr;
    hdr.point_count = static_cast<uint32_t>(logged_count);
    hdr.original_count = static_cast<uint32_t>(original_count);
    hdr.downsample_ratio = static_cast<float>(config_.point_downsample);

    data.resize(sizeof(hdr) + logged_count * sizeof(LogPoint));
    std::memcpy(data.data(), &hdr, sizeof(hdr));

    LogPoint* pts = reinterpret_cast<LogPoint*>(data.data() + sizeof(hdr));
    size_t out_idx = 0;
    for (size_t i = 0; i < num_points && out_idx < logged_count; i += config_.point_downsample) {
        size_t base = i * 5;
        pts[out_idx].x = points_xyzit[base];
        pts[out_idx].y = points_xyzit[base + 1];
        pts[out_idx].z = points_xyzit[base + 2];
        pts[out_idx].intensity = static_cast<uint8_t>(points_xyzit[base + 3]);
        pts[out_idx].time_offset_us = points_xyzit[base + 4];
        out_idx++;
    }

    queueMessage(LogMsgType::POINT_CLOUD, data.data(), data.size());
}

inline void DiagnosticLogger::logVescStatus(int motor_id, int32_t erpm, float duty,
                                             float current_motor, float current_input,
                                             float temp_fet, float temp_motor, float voltage_in,
                                             int32_t tachometer, bool connected) {
    if (!config_.log_vesc) return;

    LogVescStatus status;
    status.motor_id = static_cast<uint8_t>(motor_id);
    status.erpm = erpm;
    status.duty = duty;
    status.current_motor = current_motor;
    status.current_input = current_input;
    status.temp_fet = temp_fet;
    status.temp_motor = temp_motor;
    status.voltage_in = voltage_in;
    status.tachometer = tachometer;
    status.connected = connected ? 1 : 0;

    queueMessage(LogMsgType::VESC_STATUS, &status, sizeof(status));
}

inline void DiagnosticLogger::logGamepad(float lx, float ly, float rx, float ry,
                                          float lt, float rt, uint32_t buttons, bool connected) {
    if (!config_.log_gamepad) return;

    LogGamepadInput input;
    input.left_x = lx; input.left_y = ly;
    input.right_x = rx; input.right_y = ry;
    input.left_trigger = lt; input.right_trigger = rt;
    input.buttons = buttons;
    input.connected = connected ? 1 : 0;

    queueMessage(LogMsgType::GAMEPAD_INPUT, &input, sizeof(input));
}

inline void DiagnosticLogger::logLidarStatus(int connection, int point_rate, int imu_rate,
                                              float gravity_mag, bool imu_init) {
    LogLidarStatus status;
    status.connection_status = static_cast<uint8_t>(connection);
    status.point_rate = point_rate;
    status.imu_rate = imu_rate;
    status.gravity_mag = gravity_mag;
    status.imu_initialized = imu_init ? 1 : 0;

    queueMessage(LogMsgType::LIDAR_STATUS, &status, sizeof(status));
}

//==============================================================================
// SLAM Logging
//==============================================================================

inline void DiagnosticLogger::logSlamPose(float x, float y, float z,
                                           float roll, float pitch, float yaw,
                                           float vx, float vy, float vz,
                                           float wx, float wy, float wz, bool valid) {
    if (!config_.log_slam_pose) return;

    LogSlamPose pose;
    pose.x = x; pose.y = y; pose.z = z;
    pose.roll = roll; pose.pitch = pitch; pose.yaw = yaw;
    pose.vx = vx; pose.vy = vy; pose.vz = vz;
    pose.wx = wx; pose.wy = wy; pose.wz = wz;
    pose.valid = valid ? 1 : 0;

    queueMessage(LogMsgType::SLAM_POSE, &pose, sizeof(pose));
}

inline void DiagnosticLogger::logSlamTiming(float total_ms, float imu_ms, float undistort_ms,
                                             float downsample_ms, float ikf_ms, float map_ms,
                                             uint32_t pts_in, uint32_t pts_filter,
                                             uint32_t pts_ds, uint32_t pts_match) {
    if (!config_.log_slam_timing) return;

    LogSlamTiming timing;
    timing.total_ms = total_ms;
    timing.imu_process_ms = imu_ms;
    timing.undistort_ms = undistort_ms;
    timing.downsample_ms = downsample_ms;
    timing.ikf_update_ms = ikf_ms;
    timing.map_update_ms = map_ms;
    timing.points_input = pts_in;
    timing.points_after_filter = pts_filter;
    timing.points_after_ds = pts_ds;
    timing.points_matched = pts_match;

    queueMessage(LogMsgType::SLAM_TIMING, &timing, sizeof(timing));
}

inline void DiagnosticLogger::logSlamIkfState(const float* cov_pos, const float* cov_rot,
                                               const float* cov_vel, const float* cov_bg,
                                               const float* cov_ba, uint32_t iterations,
                                               bool converged, float residual_mean,
                                               float residual_max) {
    if (!config_.log_slam_ikf) return;

    LogSlamIkfState state;
    std::memcpy(state.cov_pos, cov_pos, sizeof(state.cov_pos));
    std::memcpy(state.cov_rot, cov_rot, sizeof(state.cov_rot));
    std::memcpy(state.cov_vel, cov_vel, sizeof(state.cov_vel));
    std::memcpy(state.cov_bg, cov_bg, sizeof(state.cov_bg));
    std::memcpy(state.cov_ba, cov_ba, sizeof(state.cov_ba));
    state.iterations = iterations;
    state.converged = converged ? 1 : 0;
    state.residual_mean = residual_mean;
    state.residual_max = residual_max;

    queueMessage(LogMsgType::SLAM_IKF_STATE, &state, sizeof(state));
}

inline void DiagnosticLogger::logSlamMapStats(uint32_t map_points, uint32_t tree_nodes,
                                               uint32_t tree_depth, const float* extent,
                                               uint32_t added, uint32_t deleted) {
    if (!config_.log_slam_map) return;

    LogSlamMapStats stats;
    stats.map_points = map_points;
    stats.tree_nodes = tree_nodes;
    stats.tree_depth = tree_depth;
    std::memcpy(stats.map_extent, extent, sizeof(stats.map_extent));
    stats.points_added = added;
    stats.points_deleted = deleted;

    queueMessage(LogMsgType::SLAM_MAP_STATS, &stats, sizeof(stats));
}

//==============================================================================
// Fusion Logging
//==============================================================================

inline void DiagnosticLogger::logFusedPose(float x, float y, float z,
                                            float roll, float pitch, float yaw,
                                            float linear_vel, float angular_vel,
                                            int motion_state, float slam_weight,
                                            float odom_weight) {
    if (!config_.log_fused_pose) return;

    LogFusedPose pose;
    pose.x = x; pose.y = y; pose.z = z;
    pose.roll = roll; pose.pitch = pitch; pose.yaw = yaw;
    pose.linear_vel = linear_vel;
    pose.angular_vel = angular_vel;
    pose.motion_state = static_cast<uint8_t>(motion_state);
    pose.slam_weight = slam_weight;
    pose.odom_weight = odom_weight;

    queueMessage(LogMsgType::FUSED_POSE, &pose, sizeof(pose));
}

inline void DiagnosticLogger::logWheelOdom(float x, float y, float theta,
                                            float linear_vel, float angular_vel,
                                            int32_t tach_l, int32_t tach_r,
                                            float dist_l, float dist_r) {
    if (!config_.log_wheel_odom) return;

    LogWheelOdom odom;
    odom.x = x; odom.y = y; odom.theta = theta;
    odom.linear_vel = linear_vel;
    odom.angular_vel = angular_vel;
    odom.tach_left = tach_l;
    odom.tach_right = tach_r;
    odom.distance_left = dist_l;
    odom.distance_right = dist_r;

    queueMessage(LogMsgType::WHEEL_ODOM, &odom, sizeof(odom));
}

//==============================================================================
// Performance Logging
//==============================================================================

inline void DiagnosticLogger::logBufferStatus(uint32_t imu, uint32_t scans, uint32_t commands) {
    if (!config_.log_buffers) return;

    LogBufferStatus status;
    status.imu_buffer = imu;
    status.scan_buffer = scans;
    status.command_buffer = commands;

    // Add log queue size
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        status.log_buffer = static_cast<uint32_t>(write_queue_.size());
    }

    queueMessage(LogMsgType::BUFFER_STATUS, &status, sizeof(status));
}

inline void DiagnosticLogger::logThreadTiming(int thread_id, float loop_ms,
                                               float sleep_ms, float busy_ratio) {
    if (!config_.log_threads) return;

    LogThreadTiming timing;
    timing.thread_id = static_cast<uint8_t>(thread_id);
    timing.loop_time_ms = loop_ms;
    timing.sleep_time_ms = sleep_ms;
    timing.busy_ratio = busy_ratio;

    queueMessage(LogMsgType::THREAD_TIMING, &timing, sizeof(timing));
}

inline void DiagnosticLogger::logMemoryStatus(uint64_t process_mb, uint64_t map_mb,
                                               uint64_t peak_mb) {
    if (!config_.log_memory) return;

    LogMemoryStatus status;
    status.process_memory_mb = process_mb;
    status.map_memory_mb = map_mb;
    status.peak_memory_mb = peak_mb;

    queueMessage(LogMsgType::MEMORY_STATUS, &status, sizeof(status));
}

inline void DiagnosticLogger::logFrameDrop(int source, uint32_t expected,
                                            uint32_t received, uint32_t dropped) {
    LogFrameDrop drop;
    drop.source = static_cast<uint8_t>(source);
    drop.expected_seq = expected;
    drop.received_seq = received;
    drop.frames_dropped = dropped;

    queueMessage(LogMsgType::FRAME_DROP, &drop, sizeof(drop));

    // Also log as warning event
    std::string src_name = (source == 0) ? "LiDAR" : (source == 1) ? "IMU" : "SLAM";
    writeTextEvent("WARNING", src_name, "Dropped " + std::to_string(dropped) +
                   " frames (expected " + std::to_string(expected) +
                   ", got " + std::to_string(received) + ")");
}

//==============================================================================
// Event Logging
//==============================================================================

inline void DiagnosticLogger::logEvent(int severity, const std::string& source,
                                        const std::string& message) {
    if (!config_.log_events) return;

    std::vector<uint8_t> data;
    LogEventHeader hdr;
    hdr.severity = static_cast<uint8_t>(severity);
    hdr.source_len = static_cast<uint16_t>(source.size());
    hdr.message_len = static_cast<uint16_t>(message.size());

    data.resize(sizeof(hdr) + source.size() + message.size());
    std::memcpy(data.data(), &hdr, sizeof(hdr));
    std::memcpy(data.data() + sizeof(hdr), source.data(), source.size());
    std::memcpy(data.data() + sizeof(hdr) + source.size(), message.data(), message.size());

    queueMessage(severity == 2 ? LogMsgType::ERROR_EVENT :
                 severity == 1 ? LogMsgType::WARNING_EVENT : LogMsgType::INFO_EVENT,
                 data.data(), data.size());

    // Write to text log
    std::string sev_str = (severity == 2) ? "ERROR" : (severity == 1) ? "WARNING" : "INFO";
    writeTextEvent(sev_str, source, message);
}

inline void DiagnosticLogger::logStateChange(int old_state, int new_state,
                                              const std::string& reason) {
    if (!config_.log_events) return;

    std::vector<uint8_t> data;
    LogStateChange change;
    change.old_state = static_cast<uint8_t>(old_state);
    change.new_state = static_cast<uint8_t>(new_state);
    change.reason_len = static_cast<uint16_t>(reason.size());

    data.resize(sizeof(change) + reason.size());
    std::memcpy(data.data(), &change, sizeof(change));
    std::memcpy(data.data() + sizeof(change), reason.data(), reason.size());

    queueMessage(LogMsgType::STATE_CHANGE, data.data(), data.size());

    writeTextEvent("INFO", "State", "Changed from " + std::to_string(old_state) +
                   " to " + std::to_string(new_state) + ": " + reason);
}

inline void DiagnosticLogger::logCommand(int command_type, int executed,
                                          const std::string& detail) {
    if (!config_.log_events) return;

    std::vector<uint8_t> data;
    LogCommand cmd;
    cmd.command_type = static_cast<uint8_t>(command_type);
    cmd.executed = static_cast<uint8_t>(executed);
    cmd.detail_len = static_cast<uint16_t>(detail.size());

    data.resize(sizeof(cmd) + detail.size());
    std::memcpy(data.data(), &cmd, sizeof(cmd));
    std::memcpy(data.data() + sizeof(cmd), detail.data(), detail.size());

    queueMessage(executed == 0 ? LogMsgType::COMMAND_RECV : LogMsgType::COMMAND_EXEC,
                 data.data(), data.size());
}

//==============================================================================
// Relocalization Logging
//==============================================================================

inline void DiagnosticLogger::logRelocProgress(int status, float progress,
                                                int voxels, int points,
                                                float rotation, float distance, int attempt) {
    if (!config_.log_reloc) return;

    LogRelocProgress prog;
    prog.status = static_cast<uint8_t>(status);
    prog.progress = progress;
    prog.local_map_voxels = voxels;
    prog.local_map_points = points;
    prog.rotation_deg = rotation;
    prog.distance_m = distance;
    prog.attempt_number = static_cast<uint8_t>(attempt);

    queueMessage(LogMsgType::RELOC_PROGRESS, &prog, sizeof(prog));
}

inline void DiagnosticLogger::logRelocIcp(int stage, uint32_t iteration,
                                           uint32_t hyp_total, uint32_t hyp_kept,
                                           float fitness, float best_fitness,
                                           const float* transform) {
    if (!config_.log_reloc) return;

    LogRelocIcp icp;
    icp.stage = static_cast<uint8_t>(stage);
    icp.iteration = iteration;
    icp.hypotheses_total = hyp_total;
    icp.hypotheses_kept = hyp_kept;
    icp.current_fitness = fitness;
    icp.best_fitness = best_fitness;
    if (transform) {
        std::memcpy(icp.transform, transform, sizeof(icp.transform));
    } else {
        std::memset(icp.transform, 0, sizeof(icp.transform));
    }

    queueMessage(LogMsgType::RELOC_ICP, &icp, sizeof(icp));
}

} // namespace slam_gui
