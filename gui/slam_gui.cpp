/**
 * @file slam_gui.cpp
 * @brief SLAM Control GUI - Main Application
 *
 * Comprehensive GUI for robot SLAM operations:
 * - Manual control with gamepad
 * - Map building and saving
 * - Localization with pre-built maps
 * - Calibration
 * - Hull mesh coverage tracking
 * - PAUT probe configuration
 */

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
#endif

#include <windows.h>
#include <d3d11.h>
#include <tchar.h>
#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <limits>
#include <algorithm>

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"

#include "state.hpp"
#include "shared_state.hpp"
#include "command_queue.hpp"
#include "probe_coverage.hpp"
#include "diagnostic_logger.hpp"

// Hardware drivers
#define ENABLE_HARDWARE 1  // Set to 0 for simulation mode

#if ENABLE_HARDWARE
#include "slam/slam_engine.hpp"
#include "slam/livox_mid360.hpp"
#include "slam/vesc_driver.hpp"
#include "slam/sensor_fusion.hpp"
#include "slam/motion_controller.hpp"
#endif

// GPU-accelerated viewer
#include "slam_viewer.hpp"

#pragma comment(lib, "d3d11.lib")

// Gamepad support via SDL2 (supports PS5 DualSense, Xbox, etc.)
#ifdef HAS_SDL2
#include "slam/gamepad.hpp"
static std::unique_ptr<slam::Gamepad> g_gamepad;
static slam::GamepadDriveConfig g_gamepad_config;
#endif

using namespace slam_gui;

// Forward declarations
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

//==============================================================================
// DirectX 11 Globals
//==============================================================================
static ID3D11Device*            g_pd3dDevice = nullptr;
static ID3D11DeviceContext*     g_pd3dDeviceContext = nullptr;
static IDXGISwapChain*          g_pSwapChain = nullptr;
static bool                     g_SwapChainOccluded = false;
static UINT                     g_ResizeWidth = 0, g_ResizeHeight = 0;
static ID3D11RenderTargetView*  g_mainRenderTargetView = nullptr;

// GPU-accelerated map viewers
static std::unique_ptr<slam::viz::SlamViewer> g_mapping_viewer;    // For Mapping tab
static std::unique_ptr<slam::viz::SlamViewer> g_localization_viewer;  // For Localization tab
static bool g_viewers_initialized = false;

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

//==============================================================================
// Application Configuration
//==============================================================================
struct AppConfig {
    // Connection settings
    char lidar_ip[64] = "192.168.1.144";
    char host_ip[64] = "192.168.1.50";
    char can_port[32] = "COM3";
    int vesc_left_id = 1;
    int vesc_right_id = 126;

    // SLAM parameters
    float voxel_size = 0.2f;
    float blind_distance = 0.5f;
    int max_iterations = 2;       // Reduced from 3 for real-time
    float gyr_cov = 0.1f;
    bool deskew_enabled = true;
    int point_filter = 3;         // Keep every Nth point (1=all, 3=keep 1/3)
    int max_points_icp = 2000;    // CRITICAL: Limit points in IEKF matching (0=unlimited)
    int max_map_points = 500000;  // Max map points to prevent ikd-tree slowdown (0=unlimited)

    // Flyaway protection
    float max_position_jump = 0.3f;     // Max allowed position jump (m) before rejection
    float max_rotation_jump = 20.0f;    // Max allowed rotation jump (deg) before rejection
    int min_effective_points = 50;      // Min matched points required for valid update
    float imu_lpf_alpha = 0.3f;         // IMU low-pass filter (0=disabled, 0.3-0.5=moderate)

    // Fusion parameters
    float slam_alpha_pos = 0.60f;
    float slam_alpha_hdg = 0.65f;
    float straight_correction = 4.0f;
    float turning_correction = 12.0f;
    float stationary_correction = 0.25f;
    float stationary_threshold = 0.02f;

    // Motion parameters
    float max_duty = 0.15f;
    float ramp_rate = 0.30f;
    float safety_margin = 0.10f;
    float max_speed = 0.5f;  // m/s limit for manual control

    // Paths
    char calibration_file[256] = "vesc_calibration.ini";
    char map_directory[256] = ".";

    // Probe configuration
    PAUTProbeConfig probe;

    // Camera settings
    CameraMode camera_mode = CameraMode::FREE;
    float camera_distance = 5.0f;
    float camera_pitch = 30.0f;
    float camera_yaw = 0.0f;

    // Viewer settings (SlamViewer)
    float viewer_point_size = 3.0f;
    int viewer_colormap = 1;  // 0=Grayscale, 1=TURBO, 2=Viridis, 3=Height
    bool viewer_enable_lod = false;   // Disabled by default to show all points
    float viewer_lod_distance = 20.0f;
    int viewer_max_points = 5000000;  // Max visible points
    float viewer_min_zoom = 0.5f;     // Minimum camera distance (close zoom)

    // Performance settings
    bool operate_show_map = false;        // Show map points in operate tab (CPU intensive!)
    bool operate_show_scan = false;       // Show current scan in operate tab (CPU intensive!)
    int operate_max_points = 10000;       // Max points to render in operate tab
    float viewer_max_zoom = 500.0f;   // Maximum camera distance (far zoom)
    bool disable_map_viewer = false;      // Disable map viewer updates entirely (for performance testing)
};

//==============================================================================
// Global Application State
//==============================================================================
static AppConfig g_config;
static SharedState g_shared;
static CommandQueue g_commands;
static HullCoverage g_hull;

static std::atomic<bool> g_running{true};
static std::atomic<bool> g_e_stop{false};
static std::atomic<bool> g_calibration_cancel{false};  // Direct cancel flag (bypasses command queue)
static std::thread g_worker_thread;
static std::thread g_control_thread;  // High-priority motor control thread
static HWND g_hwnd = nullptr;  // Main window handle for focus checking

// Atomic velocity state for control thread (bypasses command queue latency)
static std::atomic<float> g_target_linear{0.0f};
static std::atomic<float> g_target_angular{0.0f};
static std::atomic<bool> g_can_drive{false};

// Hardware drivers
static std::atomic<bool> g_hardware_initialized{false};

#if ENABLE_HARDWARE
static std::unique_ptr<slam::LivoxMid360> g_lidar;
static std::unique_ptr<slam::VescDriver> g_vesc;
static std::mutex g_vesc_mutex;  // Protects g_vesc lifecycle operations (init/shutdown)
static std::unique_ptr<slam::SlamEngine> g_slam;
static std::unique_ptr<slam::SensorFusion> g_fusion;
static std::unique_ptr<slam::MotionController> g_motion;

// IMU queue for thread-safe callback handling
// PERFORMANCE FIX: Use deque instead of vector for O(1) front removal
static std::mutex g_imu_mutex;
static std::deque<slam::ImuData> g_imu_queue;

// Point cloud accumulation (matches live_slam.cpp pattern)
static std::mutex g_scan_mutex;
static std::vector<slam::LidarPoint> g_accumulated_points;
static uint64_t g_scan_start_time = 0;
static constexpr double SCAN_PERIOD_MS = 100.0;  // 10 Hz scan rate
static size_t g_point_filter_counter = 0;

// Completed scans ready for SLAM processing
static std::mutex g_completed_scans_mutex;
static std::vector<slam::PointCloud> g_completed_scans;

// Gravity constant for IMU conversion (Mid-360 outputs in g-units)
static constexpr double G_M_S2 = 9.81;

// Debug counters for tracking data flow
static std::atomic<uint64_t> g_point_frame_count{0};
static std::atomic<uint64_t> g_imu_count{0};
static std::atomic<uint64_t> g_scan_count{0};
static std::atomic<uint64_t> g_slam_process_count{0};

// IMU CSV recording for vibration analysis
static std::atomic<bool> g_imu_recording{false};
static std::ofstream g_imu_csv_file;
static std::mutex g_imu_csv_mutex;
static uint64_t g_imu_record_start_ns{0};
static size_t g_imu_record_count{0};
#endif

// Gamepad state
static bool g_gamepad_connected = false;
static slam::DriveCommand g_drive_cmd;  // Computed by getDriveCommand()
static float g_gamepad_left_x = 0.0f;   // For display only
static float g_gamepad_left_y = 0.0f;   // For display only

// UI state
static int g_current_tab = 0;
static bool g_show_calibration_popup = false;
static char g_map_filename[256] = "slam_map.ply";
static char g_hull_filename[256] = "";
static std::vector<std::string> g_available_maps;

// Time tracking
static auto g_start_time = std::chrono::steady_clock::now();
static float g_current_time = 0.0f;

// Performance metrics
struct PerformanceMetrics {
    std::atomic<float> slam_rate_hz{0.0f};       // SLAM updates per second
    std::atomic<float> slam_time_ms{0.0f};       // Time per SLAM update
    std::atomic<float> imu_rate_hz{0.0f};        // IMU samples per second
    std::atomic<float> point_rate_hz{0.0f};      // Points per second
    std::atomic<int> map_points{0};              // Total map points
    std::atomic<int> scan_points{0};             // Points in current scan
    std::atomic<int> buffer_imu{0};              // IMU samples in buffer
    std::atomic<int> buffer_scans{0};            // Scans waiting to process
    std::atomic<float> cpu_usage{0.0f};          // Estimated CPU usage (0-100)

    // Rolling averages
    float slam_times[60] = {0};
    int slam_time_idx = 0;
    std::chrono::steady_clock::time_point last_slam_time;
    int slam_count_window = 0;
    std::chrono::steady_clock::time_point window_start;

    void recordSlamUpdate(float time_ms) {
        slam_time_ms.store(time_ms);
        slam_times[slam_time_idx++ % 60] = time_ms;
        slam_count_window++;

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<float>(now - window_start).count();
        if (elapsed >= 1.0f) {
            slam_rate_hz.store(slam_count_window / elapsed);
            slam_count_window = 0;
            window_start = now;
        }
    }

    float getAvgSlamTime() const {
        float sum = 0;
        for (int i = 0; i < 60; i++) sum += slam_times[i];
        return sum / 60.0f;
    }
};
static PerformanceMetrics g_perf;

//==============================================================================
// Diagnostic Logger - Comprehensive logging for debugging
//==============================================================================
static DiagnosticLogger g_diag_logger;
static bool g_logging_enabled = false;
static char g_log_session_name[64] = "slam_session";

//==============================================================================
// System Log - Captures errors, warnings, and info messages
//==============================================================================
enum class LogLevel { INFO, WARNING, ERROR_LEVEL };

struct LogEntry {
    std::chrono::system_clock::time_point timestamp;
    LogLevel level;
    std::string source;     // e.g., "LiDAR", "VESC", "SLAM"
    std::string message;
};

static std::mutex g_log_mutex;
static std::vector<LogEntry> g_system_log;
static const size_t MAX_LOG_ENTRIES = 500;  // Keep last 500 entries
static bool g_show_system_log = false;

void AddLogEntry(LogLevel level, const std::string& source, const std::string& message) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    LogEntry entry;
    entry.timestamp = std::chrono::system_clock::now();
    entry.level = level;
    entry.source = source;
    entry.message = message;
    g_system_log.push_back(entry);

    // Keep log size bounded
    if (g_system_log.size() > MAX_LOG_ENTRIES) {
        g_system_log.erase(g_system_log.begin());
    }

    // Also print to console
    const char* level_str = (level == LogLevel::ERROR_LEVEL) ? "ERROR" :
                            (level == LogLevel::WARNING) ? "WARN" : "INFO";
    std::cout << "[" << level_str << "][" << source << "] " << message << std::endl;
}

// Convenience macros
#define LOG_INFO(src, msg)  AddLogEntry(LogLevel::INFO, src, msg)
#define LOG_WARN(src, msg)  AddLogEntry(LogLevel::WARNING, src, msg)
#define LOG_ERROR(src, msg) AddLogEntry(LogLevel::ERROR_LEVEL, src, msg)

//==============================================================================
// Helper: Tooltip
//==============================================================================
void HelpMarker(const char* desc) {
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 25.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void SetTooltip(const char* desc) {
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
        ImGui::SetTooltip("%s", desc);
    }
}

//==============================================================================
// Helper: Settings persistence
//==============================================================================
static const char* SETTINGS_FILE = "slam_gui_settings.ini";

bool SaveSettings() {
    std::ofstream f(SETTINGS_FILE);
    if (!f) return false;

    f << "[Connection]\n";
    f << "lidar_ip=" << g_config.lidar_ip << "\n";
    f << "host_ip=" << g_config.host_ip << "\n";
    f << "can_port=" << g_config.can_port << "\n";
    f << "vesc_left_id=" << g_config.vesc_left_id << "\n";
    f << "vesc_right_id=" << g_config.vesc_right_id << "\n";

    f << "\n[SLAM]\n";
    f << "voxel_size=" << g_config.voxel_size << "\n";
    f << "blind_distance=" << g_config.blind_distance << "\n";
    f << "max_iterations=" << g_config.max_iterations << "\n";
    f << "max_points_icp=" << g_config.max_points_icp << "\n";
    f << "max_map_points=" << g_config.max_map_points << "\n";
    f << "gyr_cov=" << g_config.gyr_cov << "\n";
    f << "deskew_enabled=" << (g_config.deskew_enabled ? 1 : 0) << "\n";
    f << "point_filter=" << g_config.point_filter << "\n";
    f << "max_position_jump=" << g_config.max_position_jump << "\n";
    f << "max_rotation_jump=" << g_config.max_rotation_jump << "\n";
    f << "min_effective_points=" << g_config.min_effective_points << "\n";
    f << "imu_lpf_alpha=" << g_config.imu_lpf_alpha << "\n";

    f << "\n[Fusion]\n";
    f << "slam_alpha_pos=" << g_config.slam_alpha_pos << "\n";
    f << "slam_alpha_hdg=" << g_config.slam_alpha_hdg << "\n";
    f << "straight_correction=" << g_config.straight_correction << "\n";
    f << "turning_correction=" << g_config.turning_correction << "\n";

    f << "\n[Motion]\n";
    f << "max_duty=" << g_config.max_duty << "\n";
    f << "ramp_rate=" << g_config.ramp_rate << "\n";
    f << "max_speed=" << g_config.max_speed << "\n";

    f << "\n[Viewer]\n";
    f << "point_size=" << g_config.viewer_point_size << "\n";
    f << "colormap=" << g_config.viewer_colormap << "\n";
    f << "enable_lod=" << (g_config.viewer_enable_lod ? 1 : 0) << "\n";
    f << "lod_distance=" << g_config.viewer_lod_distance << "\n";
    f << "max_points=" << g_config.viewer_max_points << "\n";
    f << "min_zoom=" << g_config.viewer_min_zoom << "\n";
    f << "max_zoom=" << g_config.viewer_max_zoom << "\n";

    f << "\n[Paths]\n";
    f << "calibration_file=" << g_config.calibration_file << "\n";
    f << "map_directory=" << g_config.map_directory << "\n";

    return true;
}

bool LoadSettings() {
    std::ifstream f(SETTINGS_FILE);
    if (!f) return false;

    std::string line;
    while (std::getline(f, line)) {
        // Skip comments and section headers
        if (line.empty() || line[0] == '#' || line[0] == '[') continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string value = line.substr(eq + 1);

        // Connection
        if (key == "lidar_ip") strncpy(g_config.lidar_ip, value.c_str(), sizeof(g_config.lidar_ip) - 1);
        else if (key == "host_ip") strncpy(g_config.host_ip, value.c_str(), sizeof(g_config.host_ip) - 1);
        else if (key == "can_port") strncpy(g_config.can_port, value.c_str(), sizeof(g_config.can_port) - 1);
        else if (key == "vesc_left_id") g_config.vesc_left_id = std::stoi(value);
        else if (key == "vesc_right_id") g_config.vesc_right_id = std::stoi(value);
        // SLAM
        else if (key == "voxel_size") g_config.voxel_size = std::stof(value);
        else if (key == "blind_distance") g_config.blind_distance = std::stof(value);
        else if (key == "max_iterations") g_config.max_iterations = std::stoi(value);
        else if (key == "max_points_icp") g_config.max_points_icp = std::stoi(value);
        else if (key == "max_map_points") g_config.max_map_points = std::stoi(value);
        else if (key == "gyr_cov") g_config.gyr_cov = std::stof(value);
        else if (key == "deskew_enabled") g_config.deskew_enabled = (std::stoi(value) != 0);
        else if (key == "point_filter") g_config.point_filter = std::stoi(value);
        else if (key == "max_position_jump") g_config.max_position_jump = std::stof(value);
        else if (key == "max_rotation_jump") g_config.max_rotation_jump = std::stof(value);
        else if (key == "min_effective_points") g_config.min_effective_points = std::stoi(value);
        else if (key == "imu_lpf_alpha") g_config.imu_lpf_alpha = std::stof(value);
        // Fusion
        else if (key == "slam_alpha_pos") g_config.slam_alpha_pos = std::stof(value);
        else if (key == "slam_alpha_hdg") g_config.slam_alpha_hdg = std::stof(value);
        else if (key == "straight_correction") g_config.straight_correction = std::stof(value);
        else if (key == "turning_correction") g_config.turning_correction = std::stof(value);
        // Motion
        else if (key == "max_duty") g_config.max_duty = std::stof(value);
        else if (key == "ramp_rate") g_config.ramp_rate = std::stof(value);
        else if (key == "max_speed") g_config.max_speed = std::stof(value);
        // Viewer
        else if (key == "point_size") g_config.viewer_point_size = std::stof(value);
        else if (key == "colormap") g_config.viewer_colormap = std::stoi(value);
        else if (key == "enable_lod") g_config.viewer_enable_lod = (std::stoi(value) != 0);
        else if (key == "lod_distance") g_config.viewer_lod_distance = std::stof(value);
        else if (key == "max_points") g_config.viewer_max_points = std::stoi(value);
        else if (key == "min_zoom") g_config.viewer_min_zoom = std::stof(value);
        else if (key == "max_zoom") g_config.viewer_max_zoom = std::stof(value);
        // Paths
        else if (key == "calibration_file") strncpy(g_config.calibration_file, value.c_str(), sizeof(g_config.calibration_file) - 1);
        else if (key == "map_directory") strncpy(g_config.map_directory, value.c_str(), sizeof(g_config.map_directory) - 1);
    }
    return true;
}

//==============================================================================
// Helper: Scan for PLY files
//==============================================================================
void ScanForMaps() {
    g_available_maps.clear();
    try {
        for (const auto& entry : std::filesystem::directory_iterator(g_config.map_directory)) {
            if (entry.path().extension() == ".ply") {
                g_available_maps.push_back(entry.path().filename().string());
            }
        }
    } catch (...) {}
}

//==============================================================================
// Gamepad Input (SDL2 - supports PS5 DualSense, Xbox, etc.)
// Uses getDriveCommand() matching manual_drive.cpp for identical behavior
//==============================================================================
void UpdateGamepad() {
#ifdef HAS_SDL2
    if (!g_gamepad) return;

    // Poll SDL events and update gamepad state
    g_gamepad->update();

    // Try to reconnect if disconnected (check every ~2 seconds)
    static auto last_reconnect_attempt = std::chrono::steady_clock::now();
    if (!g_gamepad->isConnected()) {
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - last_reconnect_attempt).count();
        if (elapsed >= 2.0f) {
            last_reconnect_attempt = now;
            // Try to re-initialize (will scan for new controllers)
            if (g_gamepad->init()) {
                g_shared.setStatusMessage("Gamepad reconnected: " + g_gamepad->getControllerName());
                g_gamepad->setLEDColor(255, 255, 255);
                g_gamepad->rumble(0.3f, 0.3f, 200);
            }
        }
    }

    if (g_gamepad->isConnected()) {
        g_gamepad_connected = true;
        g_shared.gamepad_connected.store(true);

        // Get raw state for display and E-STOP detection
        slam::GamepadState state = g_gamepad->getState();

        // Store raw values for UI display (with deadzone for display)
        const float deadzone = 0.12f;
        auto applyDeadzone = [deadzone](float val) -> float {
            if (std::abs(val) < deadzone) return 0.0f;
            float sign = val > 0 ? 1.0f : -1.0f;
            return sign * (std::abs(val) - deadzone) / (1.0f - deadzone);
        };
        g_gamepad_left_y = applyDeadzone(-state.left_stick_y);  // For display
        g_gamepad_left_x = applyDeadzone(-state.left_stick_x);  // For display (steering)

        // Use getDriveCommand() for actual control - matches manual_drive.cpp exactly
        // Config: single-stick mode (left stick for both drive and steer)
        g_gamepad_config.max_linear_velocity = g_config.max_speed;
        g_gamepad_config.max_angular_velocity = 3.5f;
        g_gamepad_config.stick_deadzone = 0.12f;
        g_gamepad_config.max_speed_scale = 0.8f;
        g_gamepad_config.use_trigger_boost = false;
        g_gamepad_config.right_stick_steer = false;  // CRITICAL: Left stick X = steering
        g_drive_cmd = g_gamepad->getDriveCommand(g_gamepad_config);

        // E-STOP: L1 (left shoulder) or Circle (east button)
        static bool prev_estop = false;
        bool estop_pressed = state.left_shoulder || state.button_east;
        if (estop_pressed && !prev_estop) {
            g_e_stop.store(true);
            g_shared.e_stop.store(true);
            g_commands.push(Command::simple(CommandType::E_STOP));
            g_gamepad->rumble(0.8f, 0.8f, 500);  // Haptic feedback
            g_gamepad->setLEDColor(255, 0, 0);   // Red LED
        }
        prev_estop = estop_pressed;

        // Update LED color based on state (user-requested scheme)
        // Blue = connected/idle, Green = scanning modes, Red = E-STOP, Orange flash = error
        if (g_e_stop.load()) {
            g_gamepad->setLEDColor(255, 0, 0);  // Red = E-STOP active
        } else {
            AppState app_state = g_shared.getAppState();
            bool has_error = !g_shared.getErrorMessage().empty();

            // Check for connection issues
            bool lidar_ok = (g_shared.getLidarStatus().connection == ConnectionStatus::CONNECTED);
            MotorStatus motor_l = g_shared.getMotorStatus(0);
            MotorStatus motor_r = g_shared.getMotorStatus(1);
            bool vesc_ok = motor_l.connected || motor_r.connected;
            bool connection_issue = !lidar_ok || !vesc_ok;

            if (has_error || connection_issue) {
                // Orange flashing for errors/connection issues
                static auto last_flash = std::chrono::steady_clock::now();
                static bool flash_on = true;
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<float>(now - last_flash).count() > 0.3f) {
                    flash_on = !flash_on;
                    last_flash = now;
                }
                if (flash_on) {
                    g_gamepad->setLEDColor(255, 140, 0);  // Orange
                } else {
                    g_gamepad->setLEDColor(50, 30, 0);    // Dim orange
                }
            } else if (app_state == AppState::MAPPING ||
                       app_state == AppState::LOCALIZED ||
                       app_state == AppState::RELOCALIZING ||
                       app_state == AppState::OPERATING) {
                g_gamepad->setLEDColor(0, 255, 0);  // Green = scanning/operating modes
            } else {
                g_gamepad->setLEDColor(0, 100, 255);  // Blue = connected/idle
            }
        }
    } else {
        g_gamepad_connected = false;
        g_shared.gamepad_connected.store(false);
        g_gamepad_left_x = 0.0f;
        g_gamepad_left_y = 0.0f;
        g_drive_cmd = slam::DriveCommand{};  // Zero velocity
    }
#else
    // No gamepad support without SDL2
    g_gamepad_connected = false;
    g_shared.gamepad_connected.store(false);
#endif
}

//==============================================================================
// Worker Thread
//==============================================================================

#if ENABLE_HARDWARE

// LiDAR callback - accumulate points into scans (matches live_slam.cpp pattern)
void OnPointCloud(const slam::LivoxPointCloudFrame& frame) {
    g_point_frame_count++;  // Debug counter
    std::lock_guard<std::mutex> lock(g_scan_mutex);

    // Start new scan if empty
    if (g_accumulated_points.empty()) {
        g_scan_start_time = frame.timestamp_ns;
        g_accumulated_points.reserve(20000);
    }

    // Calculate frame time offset from scan start
    double frame_offset_ms = (frame.timestamp_ns - g_scan_start_time) / 1e6;

    // Process each point with proper filtering (matches live_slam.cpp)
    for (size_t i = 0; i < frame.points.size(); i++) {
        const slam::V3D& p = frame.points[i];

        // Tag filtering - only keep valid returns (matching original FAST-LIO)
        // Tag bits 4-5: 0x00=single return, 0x10=strongest
        if (!frame.tags.empty()) {
            uint8_t return_type = frame.tags[i] & 0x30;
            if (return_type != 0x00 && return_type != 0x10) continue;
        }

        // Range filtering
        double dist = p.norm();
        if (dist < g_config.blind_distance || dist > 100.0) continue;

        // Skip zero points
        if (std::abs(p.x()) < 0.001 && std::abs(p.y()) < 0.001 && std::abs(p.z()) < 0.001) continue;

        // Point decimation - keep every Nth valid point (reduces noise)
        g_point_filter_counter++;
        if (g_config.point_filter > 1 && (g_point_filter_counter % g_config.point_filter) != 0) continue;

        slam::LidarPoint lp;
        lp.x = static_cast<float>(p.x());
        lp.y = static_cast<float>(p.y());
        lp.z = static_cast<float>(p.z());
        lp.intensity = (i < frame.reflectivities.size()) ? static_cast<float>(frame.reflectivities[i]) : 100.0f;

        // Calculate time offset: frame offset + per-point offset within packet
        float point_offset_ms = (i < frame.time_offsets_us.size()) ?
            (frame.time_offsets_us[i] / 1000.0f) : 0.0f;
        lp.time_offset_ms = static_cast<float>(frame_offset_ms) + point_offset_ms;

        lp.tag = frame.tags.empty() ? 0 : frame.tags[i];
        lp.line = 0;

        g_accumulated_points.push_back(lp);
    }

    // Check if we've accumulated enough for a complete scan (~100ms)
    if (frame_offset_ms >= SCAN_PERIOD_MS && g_accumulated_points.size() > 100) {
        // Create completed scan
        slam::PointCloud cloud;
        cloud.timestamp_ns = g_scan_start_time;
        cloud.points = std::move(g_accumulated_points);

        // Queue for SLAM processing
        {
            std::lock_guard<std::mutex> scans_lock(g_completed_scans_mutex);
            g_completed_scans.push_back(std::move(cloud));
            g_scan_count++;  // Debug counter
            // Keep only recent scans if processing is slow
            if (g_completed_scans.size() > 5) {
                g_completed_scans.erase(g_completed_scans.begin());
            }
        }

        // Reset accumulator
        g_accumulated_points.clear();
        g_accumulated_points.reserve(20000);
    }
}

void OnIMU(const slam::LivoxIMUFrame& frame) {
    g_imu_count++;  // Debug counter
    std::lock_guard<std::mutex> lock(g_imu_mutex);
    // Convert to ImuData with proper units
    slam::ImuData imu;
    imu.timestamp_ns = frame.timestamp_ns;
    imu.gyro = frame.gyro;                       // Already in rad/s
    imu.acc = frame.accel * G_M_S2;              // Convert g-units to m/s²
    g_imu_queue.push_back(imu);

    // Log IMU sample (downsampling handled by logger)
    if (g_diag_logger.isRunning()) {
        float gravity_mag = static_cast<float>(imu.acc.norm());
        bool imu_init = g_slam ? g_slam->isInitialized() : false;
        g_diag_logger.logImu(
            static_cast<float>(imu.acc.x()), static_cast<float>(imu.acc.y()), static_cast<float>(imu.acc.z()),
            static_cast<float>(imu.gyro.x()), static_cast<float>(imu.gyro.y()), static_cast<float>(imu.gyro.z()),
            gravity_mag, imu_init);
    }

    // CSV recording for vibration analysis
    if (g_imu_recording.load()) {
        std::lock_guard<std::mutex> csv_lock(g_imu_csv_mutex);
        if (g_imu_csv_file.is_open()) {
            if (g_imu_record_start_ns == 0) {
                g_imu_record_start_ns = frame.timestamp_ns;
            }
            double time_s = (frame.timestamp_ns - g_imu_record_start_ns) / 1e9;
            g_imu_csv_file << std::fixed << std::setprecision(6)
                << time_s << ","
                << frame.gyro.x() << "," << frame.gyro.y() << "," << frame.gyro.z() << ","
                << frame.accel.x() << "," << frame.accel.y() << "," << frame.accel.z() << "\n";
            g_imu_record_count++;
        }
    }

    // Keep buffer bounded
    // PERFORMANCE FIX: Use pop_front (O(1) for deque) instead of erase (O(n) for vector)
    while (g_imu_queue.size() > 100) {
        g_imu_queue.pop_front();
    }
}

// Helper: Convert SlamEngine state to GUI types
void UpdateSlamState() {
    if (!g_slam || !g_slam->isInitialized() || !g_fusion) return;

    // Get pose from SLAM (4x4 transformation matrix)
    Eigen::Matrix4d T = g_slam->getPose();
    Eigen::Vector3d pos = T.block<3,1>(0,3);
    Eigen::Matrix3d rot = T.block<3,3>(0,0);

    // Extract Euler angles properly for ground robot
    // Yaw (rotation about Z): atan2(R21, R11)
    // Pitch (rotation about Y): -asin(R31)
    // Roll (rotation about X): atan2(R32, R33)
    float yaw = static_cast<float>(std::atan2(rot(1,0), rot(0,0)));
    float pitch = static_cast<float>(-std::asin(std::clamp(rot(2,0), -1.0, 1.0)));
    float roll = static_cast<float>(std::atan2(rot(2,1), rot(2,2)));

    // Update sensor fusion with SLAM pose
    slam::Pose3D slam_pose;
    slam_pose.x = static_cast<float>(pos.x());
    slam_pose.y = static_cast<float>(pos.y());
    slam_pose.z = static_cast<float>(pos.z());
    slam_pose.roll = roll;
    slam_pose.pitch = pitch;
    slam_pose.yaw = yaw;

    // Update fusion
    g_fusion->updateSlamPose(slam_pose);

    // Update shared state with SLAM pose
    Pose3D gui_slam_pose;
    gui_slam_pose.x = slam_pose.x;
    gui_slam_pose.y = slam_pose.y;
    gui_slam_pose.z = slam_pose.z;
    gui_slam_pose.roll = slam_pose.roll;
    gui_slam_pose.pitch = slam_pose.pitch;
    gui_slam_pose.yaw = slam_pose.yaw;
    g_shared.setSlamPose(gui_slam_pose);

    // Log SLAM pose for diagnostics
    if (g_diag_logger.isRunning()) {
        g_diag_logger.logSlamPose(
            slam_pose.x, slam_pose.y, slam_pose.z,
            slam_pose.roll, slam_pose.pitch, slam_pose.yaw,
            0.0f, 0.0f, 0.0f,  // Velocity (not available here)
            0.0f, 0.0f, 0.0f,  // Angular velocity
            true);  // Valid
    }
}

// Helper: Update map points for 3D visualization
// Called periodically (not every frame) to avoid performance issues
void UpdateMapPointsForVisualization() {
    if (!g_slam || !g_slam->isInitialized()) return;

    // Skip entirely if viewer is disabled (for performance testing)
    if (g_config.disable_map_viewer) return;

    static auto last_update = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(now - last_update).count();

    // Update map points at 2Hz (every 0.5 seconds) to reduce overhead
    if (elapsed < 0.5f) return;
    last_update = now;

    // Get map points from SLAM engine
    auto world_points = g_slam->getMapPoints();

    // Convert to RenderPoints with height-based coloring
    std::vector<RenderPoint> render_points;
    render_points.reserve(world_points.size());

    // Find Z range for color mapping
    float z_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::lowest();
    for (const auto& wp : world_points) {
        if (wp.z < z_min) z_min = wp.z;
        if (wp.z > z_max) z_max = wp.z;
    }
    float z_range = std::max(z_max - z_min, 0.1f);  // Avoid division by zero

    for (const auto& wp : world_points) {
        RenderPoint rp;
        rp.x = wp.x;
        rp.y = wp.y;
        rp.z = wp.z;

        // Color by height (rainbow: blue->cyan->green->yellow->red)
        float t = (wp.z - z_min) / z_range;  // 0 to 1
        t = std::clamp(t, 0.0f, 1.0f);

        // HSV to RGB (hue = 240-0 degrees, blue to red)
        float hue = (1.0f - t) * 240.0f;  // 240 = blue, 0 = red
        float s = 1.0f, v = 1.0f;

        // HSV to RGB conversion
        float c = v * s;
        float x = c * (1.0f - std::abs(std::fmod(hue / 60.0f, 2.0f) - 1.0f));
        float m = v - c;
        float r, g, b;

        if (hue < 60) { r = c; g = x; b = 0; }
        else if (hue < 120) { r = x; g = c; b = 0; }
        else if (hue < 180) { r = 0; g = c; b = x; }
        else if (hue < 240) { r = 0; g = x; b = c; }
        else if (hue < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }

        rp.r = static_cast<uint8_t>((r + m) * 255);
        rp.g = static_cast<uint8_t>((g + m) * 255);
        rp.b = static_cast<uint8_t>((b + m) * 255);
        rp.a = 255;

        render_points.push_back(rp);
    }

    g_shared.setMapPoints(render_points);

    // Also update GPU viewer with PointData format
    if (g_viewers_initialized && g_mapping_viewer) {
        std::vector<slam::viz::PointData> gpu_points;
        gpu_points.reserve(world_points.size());
        for (const auto& wp : world_points) {
            slam::viz::PointData pd;
            pd.x = wp.x;
            pd.y = wp.y;
            pd.z = wp.z;
            // Map intensity to 0-255, or use height-based intensity
            pd.intensity = static_cast<uint8_t>(std::clamp(wp.intensity, 0.0f, 255.0f));
            pd.padding[0] = pd.padding[1] = pd.padding[2] = 0;
            gpu_points.push_back(pd);
        }
        g_mapping_viewer->updatePointCloud(gpu_points.data(), gpu_points.size());
    }

    // Also update current scan (more frequently visible)
    auto scan_points = g_slam->getCurrentScan();
    std::vector<RenderPoint> scan_render;
    scan_render.reserve(scan_points.size());

    for (const auto& wp : scan_points) {
        RenderPoint rp;
        rp.x = wp.x;
        rp.y = wp.y;
        rp.z = wp.z;
        // Current scan in bright white/cyan
        rp.r = 200;
        rp.g = 255;
        rp.b = 255;
        rp.a = 255;
        scan_render.push_back(rp);
    }

    g_shared.setCurrentScan(scan_render);

    // During localization, update the localization viewer overlay with the local map
    // This shows the local map being built (in green) over the pre-built map (in turbo colormap)
    AppState state = g_shared.getAppState();
    if (g_viewers_initialized && g_localization_viewer &&
        (state == AppState::RELOCALIZING || state == AppState::LOCALIZED)) {
        std::vector<slam::viz::PointData> overlay_points;
        overlay_points.reserve(world_points.size());
        for (const auto& wp : world_points) {
            slam::viz::PointData pd;
            pd.x = wp.x;
            pd.y = wp.y;
            pd.z = wp.z;
            pd.intensity = static_cast<uint8_t>(std::clamp(wp.intensity, 0.0f, 255.0f));
            pd.padding[0] = pd.padding[1] = pd.padding[2] = 0;
            overlay_points.push_back(pd);
        }
        g_localization_viewer->updateOverlayPointCloud(overlay_points.data(), overlay_points.size());
        g_localization_viewer->setOverlayColorTint(0.2f, 1.0f, 0.4f);  // Bright green for local map
    }
}

// Helper: Update motor status in shared state
void UpdateMotorStatus() {
    if (!g_vesc) return;

    // Get status for both motors
    slam::VescStatus status_l = g_vesc->getStatus(g_config.vesc_left_id);
    slam::VescStatus status_r = g_vesc->getStatus(g_config.vesc_right_id);

    // Convert to GUI MotorStatus
    MotorStatus gui_l, gui_r;

    gui_l.erpm = status_l.erpm;
    gui_l.duty = status_l.duty;
    gui_l.current_motor = status_l.current;
    gui_l.temp_fet = status_l.temp_fet;
    gui_l.temp_motor = status_l.temp_motor;
    gui_l.voltage_in = status_l.voltage;
    gui_l.tachometer = status_l.tachometer;
    gui_l.connected = (status_l.last_update_us > 0);

    gui_r.erpm = status_r.erpm;
    gui_r.duty = status_r.duty;
    gui_r.current_motor = status_r.current;
    gui_r.temp_fet = status_r.temp_fet;
    gui_r.temp_motor = status_r.temp_motor;
    gui_r.voltage_in = status_r.voltage;
    gui_r.tachometer = status_r.tachometer;
    gui_r.connected = (status_r.last_update_us > 0);

    g_shared.setMotorStatus(0, gui_l);
    g_shared.setMotorStatus(1, gui_r);

    // Log VESC status for diagnostics
    if (g_diag_logger.isRunning()) {
        g_diag_logger.logVescStatus(0, gui_l.erpm, gui_l.duty,
            gui_l.current_motor, gui_l.current_input,
            gui_l.temp_fet, gui_l.temp_motor, gui_l.voltage_in,
            gui_l.tachometer, gui_l.connected);
        g_diag_logger.logVescStatus(1, gui_r.erpm, gui_r.duty,
            gui_r.current_motor, gui_r.current_input,
            gui_r.temp_fet, gui_r.temp_motor, gui_r.voltage_in,
            gui_r.tachometer, gui_r.connected);
    }
}

// Helper: Check and attempt VESC reconnection
static std::chrono::steady_clock::time_point g_last_vesc_reconnect_attempt;
static bool g_vesc_was_connected = false;
static bool g_show_vesc_diagnostics = false;
static std::string g_vesc_diag_log;
static bool g_vesc_diag_running = false;

// VESC single-thread architecture: ControlThread owns all VESC operations
// Other threads request operations via flags and read cached results
static std::atomic<bool> g_vesc_diag_requested{false};  // Main thread sets, ControlThread processes
static std::mutex g_vesc_cache_mutex;  // Protects cached odometry/status
static slam::VescOdometry g_vesc_odom_cache;  // Updated by ControlThread
static slam::VescStatus g_vesc_status_left_cache;
static slam::VescStatus g_vesc_status_right_cache;
static std::chrono::steady_clock::time_point g_vesc_cache_time;

// LiDAR reconnect and diagnostics
static std::chrono::steady_clock::time_point g_last_lidar_reconnect_attempt;
static bool g_lidar_was_connected = false;
static bool g_show_lidar_diagnostics = false;
static std::string g_lidar_diag_log;
static bool g_lidar_diag_running = false;

void CheckVescReconnect() {
    if (!g_vesc) return;

    bool connected = g_vesc->isConnected();

    // If we just lost connection, log it
    if (g_vesc_was_connected && !connected) {
        g_shared.setStatusMessage("VESC connection lost - will retry...");
        LOG_WARN("VESC", "Connection lost - will attempt reconnection");
    }

    // If disconnected, try to reconnect periodically (every 3 seconds)
    if (!connected) {
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - g_last_vesc_reconnect_attempt).count();

        if (elapsed >= 3.0f) {
            g_last_vesc_reconnect_attempt = now;

            // Try to reinitialize - use mutex to prevent race with diagnostics
            std::unique_lock<std::mutex> lock(g_vesc_mutex, std::try_to_lock);
            if (!lock.owns_lock()) {
                // Another thread is doing VESC operations, skip this attempt
                return;
            }

            g_vesc->shutdown();
            if (g_vesc->init(g_config.can_port, g_config.vesc_left_id, g_config.vesc_right_id)) {
                g_shared.setStatusMessage("VESC reconnected!");
                LOG_INFO("VESC", "Reconnected on " + std::string(g_config.can_port));

                // Also reinit motion controller with new VESC instance
                if (g_motion) {
                    g_motion->init(g_vesc.get(), g_config.calibration_file);
                }
            }
        }
    }

    g_vesc_was_connected = connected;
}

// Helper: Update cached VESC odometry/status for other threads to read
// Called from ControlThread at 50Hz - avoids other threads accessing g_vesc directly
void UpdateVescCache() {
    if (!g_vesc || !g_vesc->isConnected()) return;

    slam::VescOdometry odom = g_vesc->getOdometry();
    slam::VescStatus left = g_vesc->getStatus(g_config.vesc_left_id);
    slam::VescStatus right = g_vesc->getStatus(g_config.vesc_right_id);

    std::lock_guard<std::mutex> lock(g_vesc_cache_mutex);
    g_vesc_odom_cache = odom;
    g_vesc_status_left_cache = left;
    g_vesc_status_right_cache = right;
    g_vesc_cache_time = std::chrono::steady_clock::now();
}

// Helper: Process VESC diagnostics request from main thread
// Called from ControlThread - all VESC lifecycle operations happen in this thread
void ProcessVescDiagnostics() {
    if (!g_vesc_diag_requested.load()) return;

    g_vesc_diag_running = true;
    g_vesc_diag_log.clear();
    g_vesc_diag_log += "=== VESC Diagnostics ===\n\n";

    // Check if COM port exists
    std::string full_port = "\\\\.\\" + std::string(g_config.can_port);
    HANDLE test_handle = CreateFileA(full_port.c_str(), GENERIC_READ | GENERIC_WRITE,
                                      0, NULL, OPEN_EXISTING, 0, NULL);
    if (test_handle == INVALID_HANDLE_VALUE) {
        DWORD err = GetLastError();
        g_vesc_diag_log += "[FAIL] Cannot open " + std::string(g_config.can_port);
        if (err == 2) {
            g_vesc_diag_log += " (port does not exist)\n";
            g_vesc_diag_log += "  -> Check if CANable is plugged in\n";
            g_vesc_diag_log += "  -> Check Device Manager for correct COM port\n";
        } else if (err == 5) {
            g_vesc_diag_log += " (access denied)\n";
            g_vesc_diag_log += "  -> Another program may be using the port\n";
            g_vesc_diag_log += "  -> Close VESC Tool, serial monitors, etc.\n";
        } else {
            g_vesc_diag_log += " (error " + std::to_string(err) + ")\n";
            g_vesc_diag_log += "  -> Try unplugging and replugging CANable\n";
        }
    } else {
        CloseHandle(test_handle);
        g_vesc_diag_log += "[OK] " + std::string(g_config.can_port) + " is accessible\n\n";

        // Try to initialize VESC
        g_vesc_diag_log += "Attempting VESC initialization...\n";
        if (g_vesc) {
            g_vesc->shutdown();
        } else {
            g_vesc = std::make_unique<slam::VescDriver>();
        }

        if (g_vesc->init(g_config.can_port, g_config.vesc_left_id, g_config.vesc_right_id)) {
            g_vesc_diag_log += "[OK] VESC initialized successfully!\n";
            g_vesc_diag_log += "  -> Waiting for status messages...\n\n";

            // Wait a bit for status messages
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            auto left_status = g_vesc->getStatus(g_config.vesc_left_id);
            auto right_status = g_vesc->getStatus(g_config.vesc_right_id);
            bool got_left = left_status.erpm != 0 || left_status.voltage > 0;
            bool got_right = right_status.erpm != 0 || right_status.voltage > 0;

            if (got_left) {
                char buf[128];
                snprintf(buf, sizeof(buf), "[OK] Left VESC (ID %d): %.1fV\n",
                         g_config.vesc_left_id, left_status.voltage);
                g_vesc_diag_log += buf;
            } else {
                char buf[128];
                snprintf(buf, sizeof(buf), "[WARN] No response from Left VESC (ID %d)\n",
                         g_config.vesc_left_id);
                g_vesc_diag_log += buf;
                g_vesc_diag_log += "  -> Check CAN wiring to left motor\n";
                g_vesc_diag_log += "  -> Verify VESC ID in VESC Tool\n";
            }

            if (got_right) {
                char buf[128];
                snprintf(buf, sizeof(buf), "[OK] Right VESC (ID %d): %.1fV\n",
                         g_config.vesc_right_id, right_status.voltage);
                g_vesc_diag_log += buf;
            } else {
                char buf[128];
                snprintf(buf, sizeof(buf), "[WARN] No response from Right VESC (ID %d)\n",
                         g_config.vesc_right_id);
                g_vesc_diag_log += buf;
                g_vesc_diag_log += "  -> Check CAN wiring to right motor\n";
                g_vesc_diag_log += "  -> Verify VESC ID in VESC Tool\n";
            }

            if (!got_left && !got_right) {
                g_vesc_diag_log += "\n[FAIL] No VESC responses received\n";
                g_vesc_diag_log += "  -> Check battery power to VESCs\n";
                g_vesc_diag_log += "  -> Enable CAN Status Messages in VESC Tool:\n";
                g_vesc_diag_log += "     App Settings > General > CAN Status Rate = 50Hz\n";
            } else if (got_left && got_right) {
                g_vesc_diag_log += "\n[SUCCESS] Both VESCs responding!\n";
                // Re-init motion controller
                if (g_motion) {
                    g_motion->init(g_vesc.get(), g_config.calibration_file);
                }
            }
        } else {
            g_vesc_diag_log += "[FAIL] VESC initialization failed\n";
            g_vesc_diag_log += "  -> SLCAN handshake failed\n";
            g_vesc_diag_log += "  -> Try unplugging and replugging CANable\n";
        }
    }

    g_vesc_diag_requested.store(false);
    g_vesc_diag_running = false;
}

// Helper: Check and attempt LiDAR reconnection
void CheckLidarReconnect() {
    // If LiDAR is already connected and streaming, nothing to do
    if (g_lidar && g_lidar->isStreaming()) {
        g_lidar_was_connected = true;
        return;
    }

    // If we were previously connected and now we're not, log it
    if (g_lidar_was_connected && (!g_lidar || !g_lidar->isStreaming())) {
        g_shared.setStatusMessage("LiDAR connection lost - will retry...");
        LOG_WARN("LiDAR", "Connection lost - will attempt reconnection");
        g_lidar_was_connected = false;
    }

    // Try to reconnect periodically (every 5 seconds)
    auto now = std::chrono::steady_clock::now();
    float elapsed = std::chrono::duration<float>(now - g_last_lidar_reconnect_attempt).count();

    if (elapsed >= 5.0f) {
        g_last_lidar_reconnect_attempt = now;

        // If we don't have a LiDAR instance, create one
        if (!g_lidar) {
            g_lidar = std::make_unique<slam::LivoxMid360>();
            g_lidar->setPointCloudCallback(OnPointCloud);
            g_lidar->setIMUCallback(OnIMU);
        }

        // Try to connect
        if (g_lidar->connect(g_config.lidar_ip, g_config.host_ip)) {
            g_shared.setStatusMessage("LiDAR reconnected!");
            LOG_INFO("LiDAR", "Reconnected to " + std::string(g_config.lidar_ip));
            g_lidar_was_connected = true;

            // Update LiDAR status
            LidarStatus lidar_status;
            lidar_status.connection = ConnectionStatus::CONNECTED;
            lidar_status.ip_address = g_config.lidar_ip;
            g_shared.setLidarStatus(lidar_status);

            // Create SLAM engine if needed
            if (!g_slam) {
                g_slam = std::make_unique<slam::SlamEngine>();
                slam::SlamConfig slam_config;
                slam_config.filter_size_surf = g_config.voxel_size;
                slam_config.filter_size_map = g_config.voxel_size;
                slam_config.gyr_cov = g_config.gyr_cov;
                slam_config.acc_cov = g_config.gyr_cov;
                slam_config.b_gyr_cov = 0.0001;
                slam_config.b_acc_cov = 0.0001;
                g_slam->init(slam_config);
            }

            g_shared.hardware_connected.store(true);
        }
    }
}

// Forward declaration for cleanup
void ShutdownHardware();

// Helper: Initialize hardware
bool InitializeHardware() {
    g_shared.setStatusMessage("Initializing LiDAR...");

    // Create LiDAR driver
    g_lidar = std::make_unique<slam::LivoxMid360>();
    g_lidar->setPointCloudCallback(OnPointCloud);
    g_lidar->setIMUCallback(OnIMU);

    if (!g_lidar->connect(g_config.lidar_ip, g_config.host_ip)) {
        std::string err_msg = "Failed to connect to LiDAR at " + std::string(g_config.lidar_ip);
        g_shared.setErrorMessage(err_msg);
        LOG_ERROR("LiDAR", err_msg);
        ShutdownHardware();  // Cleanup partial init
        return false;
    }
    LOG_INFO("LiDAR", "Connected to " + std::string(g_config.lidar_ip));

    // Update LiDAR status
    LidarStatus lidar_status;
    lidar_status.connection = ConnectionStatus::CONNECTED;
    lidar_status.ip_address = g_config.lidar_ip;
    g_shared.setLidarStatus(lidar_status);

    g_shared.setStatusMessage("Initializing VESC...");

    // Create VESC driver (with mutex to prevent race with diagnostics)
    {
        std::lock_guard<std::mutex> lock(g_vesc_mutex);
        g_vesc = std::make_unique<slam::VescDriver>();
        if (!g_vesc->init(g_config.can_port, g_config.vesc_left_id, g_config.vesc_right_id)) {
            std::string err_msg = "Failed to initialize VESC on " + std::string(g_config.can_port);
            g_shared.setErrorMessage(err_msg);
            LOG_ERROR("VESC", err_msg);
            ShutdownHardware();  // Cleanup partial init
            return false;
        }
    }
    LOG_INFO("VESC", "Initialized on " + std::string(g_config.can_port));

    g_shared.setStatusMessage("Initializing SLAM engine...");

    // Create SLAM engine with config (matching live_slam.cpp)
    g_slam = std::make_unique<slam::SlamEngine>();
    slam::SlamConfig slam_config;
    // Voxel filter sizes - CRITICAL for proper map building
    slam_config.filter_size_surf = g_config.voxel_size;
    slam_config.filter_size_map = g_config.voxel_size;
    // IMU noise covariances - must match live_slam.cpp defaults
    slam_config.gyr_cov = g_config.gyr_cov;
    slam_config.acc_cov = g_config.gyr_cov;   // Use same value
    slam_config.b_gyr_cov = 0.0001;
    slam_config.b_acc_cov = 0.0001;
    // Processing parameters
    slam_config.max_iterations = g_config.max_iterations;
    slam_config.max_points_icp = g_config.max_points_icp;  // CRITICAL for real-time performance
    slam_config.max_map_points = g_config.max_map_points;  // Limit map growth to prevent slowdown
    slam_config.deskew_enabled = g_config.deskew_enabled;
    // Flyaway protection
    slam_config.max_position_jump = g_config.max_position_jump;
    slam_config.max_rotation_jump_deg = g_config.max_rotation_jump;
    slam_config.min_effective_points = g_config.min_effective_points;
    slam_config.imu_lpf_alpha = g_config.imu_lpf_alpha;  // Vibration compensation
    slam_config.save_map = true;
    if (!g_slam->init(slam_config)) {
        g_shared.setErrorMessage("Failed to initialize SLAM engine");
        LOG_ERROR("SLAM", "Failed to initialize SLAM engine");
        ShutdownHardware();  // Cleanup partial init
        return false;
    }
    LOG_INFO("SLAM", "Engine initialized");

    g_shared.setStatusMessage("Initializing sensor fusion...");

    // Create sensor fusion
    g_fusion = std::make_unique<slam::SensorFusion>();
    slam::FusionConfig fusion_config;
    fusion_config.slam_position_alpha = g_config.slam_alpha_pos;
    fusion_config.slam_heading_alpha = g_config.slam_alpha_hdg;
    fusion_config.straight_position_correction = g_config.straight_correction;
    fusion_config.straight_heading_correction = g_config.straight_correction;
    fusion_config.turning_position_correction = g_config.turning_correction;
    fusion_config.stationary_position_correction = g_config.stationary_correction;
    fusion_config.stationary_heading_correction = g_config.stationary_correction;
    fusion_config.stationary_correction_threshold = g_config.stationary_threshold;
    g_fusion->init(fusion_config);

    g_shared.setStatusMessage("Initializing motion controller...");

    // Create motion controller
    g_motion = std::make_unique<slam::MotionController>();
    if (!g_motion->init(g_vesc.get(), g_config.calibration_file)) {
        // Non-fatal - can still operate with default calibration
        g_shared.setStatusMessage("Warning: Using default motor calibration");
    }
    g_motion->setMaxDuty(g_config.max_duty);
    g_motion->setRampRate(g_config.ramp_rate);

    // Start LiDAR streaming
    if (!g_lidar->startStreaming()) {
        g_shared.setErrorMessage("Failed to start LiDAR streaming");
        ShutdownHardware();  // Cleanup partial init
        return false;
    }

    g_shared.hardware_connected.store(true);
    g_shared.setStatusMessage("Hardware initialized");
    return true;
}

// Helper: Shutdown hardware
void ShutdownHardware() {
    if (g_motion) {
        g_motion->stop(true);  // Immediate stop
    }
    if (g_lidar) {
        g_lidar->stop();
    }

    // Shutdown VESC with mutex to prevent race conditions
    {
        std::lock_guard<std::mutex> lock(g_vesc_mutex);
        if (g_vesc) {
            g_vesc->stop();
            g_vesc->shutdown();
        }
        g_vesc.reset();
    }

    g_lidar.reset();
    g_slam.reset();
    g_fusion.reset();
    g_motion.reset();

    g_shared.hardware_connected.store(false);
}

//==============================================================================
// High-Priority Motor Control Thread
// Runs at 50Hz guaranteed, never blocked by SLAM processing
//==============================================================================
void ControlThread() {
    // Set thread priority to highest for real-time motor control
    // This ensures motor commands are sent consistently even during heavy SLAM processing
    HANDLE hThread = GetCurrentThread();
    SetThreadPriority(hThread, THREAD_PRIORITY_HIGHEST);

    auto last_update = std::chrono::steady_clock::now();

    while (g_running.load()) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_update).count();
        last_update = now;

        // Clamp dt to reasonable range
        if (dt > 0.1f) dt = 0.02f;
        if (dt < 0.001f) dt = 0.001f;

        // Handle E-STOP
        if (g_e_stop.load()) {
            if (g_motion) {
                g_motion->emergencyStop();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Only control motors when hardware is ready and we can drive
        if (g_hardware_initialized.load() && g_motion && g_can_drive.load()) {
            // Read atomic velocity targets
            float linear = g_target_linear.load();
            float angular = g_target_angular.load();

            // Send motor commands
            g_motion->setVelocity(linear, angular);
            g_motion->update(dt);
        } else if (g_motion) {
            // Not in drivable state - ensure motors are stopped
            g_motion->setVelocity(0.0f, 0.0f);
            g_motion->update(dt);
        }

        // Only check for reconnection AFTER initial hardware init is complete
        // This prevents race conditions with WorkerThread's InitializeHardware()
        if (g_hardware_initialized.load()) {
            // Check for VESC disconnection and attempt reconnection
            CheckVescReconnect();

            // Check for LiDAR disconnection and attempt reconnection
            CheckLidarReconnect();

            // Update VESC cache for WorkerThread to read (single-thread arch)
            UpdateVescCache();

            // Update motor status for GUI (always, not just when driving)
            UpdateMotorStatus();

            // Process any pending VESC diagnostics request from main thread
            ProcessVescDiagnostics();
        }

        // Precise 50Hz timing (20ms period)
        auto elapsed = std::chrono::steady_clock::now() - now;
        auto sleep_time = std::chrono::milliseconds(20) - elapsed;
        if (sleep_time > std::chrono::milliseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        }
    }

    // Ensure motors stop on thread exit
    if (g_motion) {
        g_motion->stop(true);
    }
}

void WorkerThread() {
    g_shared.setAppState(AppState::STARTUP);
    g_shared.setStatusMessage("Initializing hardware...");

    // Initialize hardware
    if (!InitializeHardware()) {
        g_shared.setAppState(AppState::IDLE);
        g_shared.setStatusMessage("Hardware init failed - check connections");
        // Continue in degraded mode
    } else {
        g_shared.setAppState(AppState::IDLE);
    }

    g_hardware_initialized.store(true);

    auto last_update = std::chrono::steady_clock::now();
    // Note: Motor velocity targets are now in g_target_linear/g_target_angular atomics
    // Control is handled by the dedicated high-priority ControlThread

    while (g_running.load()) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_update).count();
        last_update = now;

        // Clamp dt to reasonable range (prevents issues after long stalls)
        dt = std::clamp(dt, 0.001f, 0.1f);

        // 1. Check E-STOP first
        if (g_e_stop.load()) {
            g_shared.setAppState(AppState::E_STOPPED);

            // Emergency stop motors
            if (g_motion) {
                g_motion->emergencyStop();
            }

            // Wait for E-STOP reset
            while (g_e_stop.load() && g_running.load()) {
                while (auto cmd = g_commands.pop()) {
                    if (cmd->type == CommandType::RESET_E_STOP) {
                        g_e_stop.store(false);
                        g_shared.e_stop.store(false);

                        // Check if hardware needs reconnection
                        bool lidar_ok = g_lidar && g_lidar->isConnected();
                        bool vesc_ok = g_vesc && g_vesc->isConnected();

                        if (!lidar_ok || !vesc_ok) {
                            g_shared.setStatusMessage("Reconnecting hardware after E-STOP...");
                            if (InitializeHardware()) {
                                g_shared.setStatusMessage("Hardware reconnected");
                            } else {
                                g_shared.setErrorMessage("Failed to reconnect hardware - check connections");
                            }
                        }

                        g_shared.setAppState(AppState::IDLE);
                        g_shared.setStatusMessage("E-STOP reset");
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            // Reset timing after E-STOP to avoid huge dt
            last_update = std::chrono::steady_clock::now();
            continue;
        }

        // 2. Process commands from GUI
        while (auto cmd = g_commands.pop()) {
            // Log command received for diagnostics
            if (g_diag_logger.isRunning()) {
                g_diag_logger.logCommand(static_cast<int>(cmd->type), 0, "");
            }

            switch (cmd->type) {
                case CommandType::E_STOP:
                    g_e_stop.store(true);
                    g_shared.e_stop.store(true);
                    if (g_motion) g_motion->emergencyStop();
                    LOG_WARN("System", "E-STOP activated - motors stopped");
                    break;

                case CommandType::RESET_E_STOP:
                    g_e_stop.store(false);
                    g_shared.e_stop.store(false);
                    g_shared.setAppState(AppState::IDLE);
                    LOG_INFO("System", "E-STOP reset - operation resumed");
                    break;

                case CommandType::CONNECT_HARDWARE:
                    if (!g_shared.hardware_connected.load()) {
                        if (InitializeHardware()) {
                            g_shared.setStatusMessage("Hardware connected");
                        }
                    }
                    break;

                case CommandType::DISCONNECT_HARDWARE:
                    ShutdownHardware();
                    g_shared.setStatusMessage("Hardware disconnected");
                    break;

                case CommandType::START_MAPPING:
                    // CRITICAL: Clear all stale sensor data first!
                    {
                        std::lock_guard<std::mutex> lock(g_completed_scans_mutex);
                        g_completed_scans.clear();
                    }
                    {
                        std::lock_guard<std::mutex> lock(g_imu_mutex);
                        g_imu_queue.clear();
                    }
                    g_accumulated_points.clear();

                    if (g_slam) {
                        g_slam->reset();
                        g_slam->setLocalizationMode(false);
                        // Apply current settings before mapping starts
                        g_slam->setVoxelSize(g_config.voxel_size);
                        g_slam->setGyrCov(g_config.gyr_cov);
                        g_slam->setDeskewEnabled(g_config.deskew_enabled);
                        // Note: max_iterations, blind_distance, point_filter require restart
                    }
                    if (g_fusion) {
                        g_fusion->reset();
                    }
                    g_shared.setAppState(AppState::MAPPING);
                    g_shared.setStatusMessage("Mapping started");
                    g_shared.clearTrajectory();
                    break;

                case CommandType::STOP_MAPPING:
                    g_target_linear.store(0.0f);
                    g_target_angular.store(0.0f);
                    if (g_motion) g_motion->stop();
                    g_shared.setAppState(AppState::IDLE);
                    g_shared.setStatusMessage("Mapping stopped");
                    break;

                case CommandType::START_OPERATING:
                    // Drive-only mode (no SLAM processing)
                    g_target_linear.store(0.0f);
                    g_target_angular.store(0.0f);
                    g_shared.setAppState(AppState::OPERATING);
                    g_shared.setStatusMessage("Operating mode - manual drive only");
                    break;

                case CommandType::STOP_OPERATING:
                    g_target_linear.store(0.0f);
                    g_target_angular.store(0.0f);
                    if (g_motion) g_motion->stop();
                    g_shared.setAppState(AppState::IDLE);
                    g_shared.setStatusMessage("Operating stopped");
                    break;

                case CommandType::SAVE_MAP: {
                    auto* file_cmd = std::get_if<FileCommand>(&cmd->payload);
                    if (file_cmd && g_slam) {
                        g_shared.setAppState(AppState::SAVE_MAP);
                        g_shared.setStatusMessage("Saving map...");
                        if (g_slam->saveMap(file_cmd->path)) {
                            g_shared.setStatusMessage("Map saved to " + file_cmd->path);
                        } else {
                            g_shared.setErrorMessage("Failed to save map");
                        }
                        g_shared.setAppState(AppState::IDLE);
                    }
                    break;
                }

                case CommandType::LOAD_MAP: {
                    auto* file_cmd = std::get_if<FileCommand>(&cmd->payload);
                    if (file_cmd) {
                        if (!g_slam) {
                            // SLAM engine not initialized - try to load for preview only
                            g_shared.setStatusMessage("Loading map for preview (LiDAR not connected)...");

                            // Load PLY directly into viewer for preview
                            if (g_viewers_initialized && g_localization_viewer) {
                                std::vector<slam::viz::PointData> gpu_points;
                                std::ifstream ply_file(file_cmd->path);
                                if (ply_file.is_open()) {
                                    std::string line;
                                    bool header_done = false;
                                    size_t vertex_count = 0;

                                    // Parse PLY header
                                    while (std::getline(ply_file, line)) {
                                        if (line.find("element vertex") != std::string::npos) {
                                            sscanf(line.c_str(), "element vertex %zu", &vertex_count);
                                        }
                                        if (line == "end_header") {
                                            header_done = true;
                                            break;
                                        }
                                    }

                                    if (header_done && vertex_count > 0) {
                                        gpu_points.reserve(vertex_count);
                                        float x, y, z;
                                        int intensity;
                                        while (ply_file >> x >> y >> z >> intensity) {
                                            slam::viz::PointData pd;
                                            pd.x = x;
                                            pd.y = y;
                                            pd.z = z;
                                            pd.intensity = static_cast<uint8_t>(std::clamp(intensity, 0, 255));
                                            pd.padding[0] = pd.padding[1] = pd.padding[2] = 0;
                                            gpu_points.push_back(pd);
                                        }

                                        g_localization_viewer->updatePointCloud(gpu_points.data(), gpu_points.size());
                                        g_localization_viewer->fitCameraToContent();
                                        g_shared.setStatusMessage("Map preview loaded (" + std::to_string(gpu_points.size()) + " points) - Connect LiDAR to localize");
                                        g_shared.setAppState(AppState::IDLE);  // Can't localize without LiDAR
                                    } else {
                                        g_shared.setErrorMessage("Failed to parse PLY file");
                                    }
                                } else {
                                    g_shared.setErrorMessage("Cannot open map file: " + file_cmd->path);
                                }
                            } else {
                                g_shared.setErrorMessage("Cannot load map: LiDAR not connected");
                            }
                        } else {
                            g_shared.setStatusMessage("Loading pre-built map...");
                            // Use loadPrebuiltMap() - stores in memory, doesn't replace ikd-tree
                            // This allows FAST-LIO to build a local map for progressive localization
                            if (g_slam->loadPrebuiltMap(file_cmd->path)) {
                                g_shared.setAppState(AppState::MAP_LOADED);
                                g_shared.setStatusMessage("Map loaded: " + file_cmd->path + " - Click Relocalize to start");

                                // Update localization viewer with pre-built map
                                if (g_viewers_initialized && g_localization_viewer) {
                                    auto prebuilt = g_slam->getPrebuiltMapPoints();
                                    std::vector<slam::viz::PointData> gpu_points;
                                    gpu_points.reserve(prebuilt.size());
                                    for (const auto& wp : prebuilt) {
                                        slam::viz::PointData pd;
                                        pd.x = wp.x;
                                        pd.y = wp.y;
                                        pd.z = wp.z;
                                        pd.intensity = static_cast<uint8_t>(std::clamp(wp.intensity, 0.0f, 255.0f));
                                        pd.padding[0] = pd.padding[1] = pd.padding[2] = 0;
                                        gpu_points.push_back(pd);
                                    }
                                    g_localization_viewer->updatePointCloud(gpu_points.data(), gpu_points.size());
                                    g_localization_viewer->fitCameraToContent();
                                }

                                // Reset relocalization progress
                                RelocalizationProgress prog;
                                prog.status_text = "Map loaded - ready to start localization";
                                g_shared.setRelocalizationProgress(prog);
                            } else {
                                g_shared.setErrorMessage("Failed to load map file");
                            }
                        }
                    }
                    break;
                }

                case CommandType::RELOCALIZE:
                    if ((g_shared.getAppState() == AppState::MAP_LOADED ||
                         g_shared.getAppState() == AppState::RELOCALIZE_FAILED) && g_slam) {
                        // CRITICAL FIX: Reset SLAM engine before starting progressive localization
                        // This clears the old ikd-tree and EKF state from previous mapping session.
                        // Without this reset, new LiDAR scans would match against stale map points,
                        // causing immediate flyaway.
                        g_slam->reset();

                        // Apply current settings (same as START_MAPPING)
                        g_slam->setVoxelSize(g_config.voxel_size);
                        g_slam->setGyrCov(g_config.gyr_cov);
                        g_slam->setDeskewEnabled(g_config.deskew_enabled);
                        g_slam->setImuLpfAlpha(g_config.imu_lpf_alpha);

                        if (g_fusion) {
                            g_fusion->reset();
                        }

                        // Start progressive localization
                        // FAST-LIO will build a local map, which we match against the pre-built map
                        g_shared.setAppState(AppState::RELOCALIZING);
                        g_shared.setStatusMessage("Starting progressive localization - rotate robot to build geometry...");

                        // Configure progressive localizer
                        // Coverage thresholds are managed internally by CoverageMonitor
                        slam::ProgressiveLocalizerConfig loc_config;
                        loc_config.max_attempts = 5;
                        loc_config.min_confidence = 0.45;
                        loc_config.high_confidence = 0.65;

                        // Start progressive localization (FAST-LIO continues building local map)
                        g_slam->startProgressiveLocalization(loc_config);

                        // Update progress to show accumulating
                        RelocalizationProgress prog;
                        prog.running = true;
                        prog.accumulating = true;
                        prog.progress = 0.0f;
                        prog.status_text = "Building local map - rotate robot slowly...";
                        prog.local_map_voxels = 0;
                        prog.rotation_deg = 0.0f;
                        prog.attempt_number = 0;
                        g_shared.setRelocalizationProgress(prog);
                    }
                    break;

                case CommandType::STOP_LOCALIZATION:
                    if (g_shared.getAppState() == AppState::LOCALIZED ||
                        g_shared.getAppState() == AppState::RELOCALIZING) {
                        // Reset to idle state
                        if (g_slam) {
                            g_slam->setLocalizationMode(false);
                        }
                        g_shared.setAppState(AppState::IDLE);
                        g_shared.setStatusMessage("Localization stopped");

                        // Clear progress
                        RelocalizationProgress prog;
                        g_shared.setRelocalizationProgress(prog);
                    }
                    break;

                case CommandType::RUN_GLOBAL_LOCALIZATION:
                    if (g_shared.getAppState() == AppState::RELOCALIZING && g_slam) {
                        if (!g_slam->isReadyForGlobalLocalization()) {
                            g_shared.setStatusMessage("Not ready yet - continue building map");
                            break;
                        }

                        // Stop robot before ICP
                        g_target_linear.store(0.0f);
                        g_target_angular.store(0.0f);
                        g_shared.setStatusMessage("Running global localization - please wait...");

                        // Update progress to show ICP is running
                        RelocalizationProgress prog = g_shared.getRelocalizationProgress();
                        prog.icp_running = true;
                        prog.ready_to_localize = false;
                        prog.status_text = "Starting ICP alignment...";
                        g_shared.setRelocalizationProgress(prog);

                        // Progress callback for ICP stages
                        auto progress_callback = [](const slam::LocalizationProgress& p) {
                            RelocalizationProgress prog = g_shared.getRelocalizationProgress();
                            prog.icp_stage = slam::toString(p.stage);
                            prog.icp_progress = p.progress;
                            prog.icp_fitness = static_cast<float>(p.current_fitness);
                            prog.icp_hypotheses = p.hypotheses_count;
                            prog.icp_kept = p.hypotheses_kept;
                            prog.icp_iteration = p.current_iteration;
                            prog.icp_max_iterations = p.max_iterations;
                            prog.status_text = p.message;
                            prog.progress = p.progress;
                            g_shared.setRelocalizationProgress(prog);
                        };

                        // Run global localization (this blocks but reports progress via callback)
                        slam::LocalizationResult result = g_slam->runGlobalLocalization(progress_callback);

                        // Update final state
                        prog = g_shared.getRelocalizationProgress();
                        prog.icp_running = false;

                        if (result.status == slam::LocalizationStatus::SUCCESS) {
                            prog.running = false;
                            prog.success = true;
                            prog.confidence = static_cast<float>(result.confidence);
                            prog.status_text = "Localized! (" + std::to_string(int(result.confidence * 100)) + "% confidence)";
                            g_shared.setAppState(AppState::LOCALIZED);
                            g_shared.setStatusMessage("Localization successful!");
                        } else if (result.status == slam::LocalizationStatus::FAILED) {
                            prog.running = false;
                            prog.success = false;
                            prog.status_text = result.message;
                            g_shared.setAppState(AppState::RELOCALIZE_FAILED);
                            g_shared.setStatusMessage("Localization failed");
                        } else {
                            // LOW_CONFIDENCE or cancelled - allow retry
                            prog.ready_to_localize = true;
                            prog.status_text = result.message;
                            g_shared.setStatusMessage(result.message);
                        }

                        g_shared.setRelocalizationProgress(prog);
                    }
                    break;

                case CommandType::CANCEL_GLOBAL_LOCALIZATION:
                    if (g_slam) {
                        g_slam->cancelGlobalLocalization();
                        g_shared.setStatusMessage("Cancelling localization...");
                    }
                    break;

                case CommandType::SET_VELOCITY: {
                    // Legacy command - velocity now set directly via atomics from main thread
                    // Keep handler for any remaining command-based callers
                    auto* vel_cmd = std::get_if<VelocityCommand>(&cmd->payload);
                    if (vel_cmd) {
                        g_target_linear.store(vel_cmd->linear_mps);
                        g_target_angular.store(vel_cmd->angular_radps);
                    }
                    break;
                }

                case CommandType::LOAD_CALIBRATION: {
                    auto* file_cmd = std::get_if<FileCommand>(&cmd->payload);
                    if (file_cmd && g_motion) {
                        if (g_motion->loadCalibration(file_cmd->path)) {
                            g_shared.setStatusMessage("Calibration loaded");
                        } else {
                            g_shared.setErrorMessage("Failed to load calibration");
                        }
                    }
                    break;
                }

                case CommandType::RUN_CALIBRATION: {
                    // Full calibration sequence using VESC driver methods
                    if (!g_vesc) {
                        g_shared.setErrorMessage("VESC not connected - cannot calibrate");
                        break;
                    }

                    g_shared.setAppState(AppState::CALIBRATING);
                    g_shared.setStatusMessage("Starting calibration...");

                    CalibrationProgress cal_prog;
                    cal_prog.running = true;
                    cal_prog.progress = 0.0f;
                    cal_prog.success = false;

                    // Progress callback for calibration phases
                    auto progress_callback = [&](int progress, const std::string& status) {
                        cal_prog.progress = progress / 100.0f;
                        cal_prog.status_text = status;
                        g_shared.setCalibrationProgress(cal_prog);
                        g_shared.setStatusMessage(status);
                    };

                    // Reset calibration cancel flag at start
                    g_calibration_cancel.store(false);

                    auto should_cancel = [&]() -> bool {
                        // Check e-stop and calibration cancel flags directly
                        // These bypass the command queue for immediate response
                        return g_e_stop.load() || g_calibration_cancel.load();
                    };

                    bool success = true;
                    slam::CalibrationResult cal_result;

                    // Phase 1: Forward/Reverse Duty Calibration
                    cal_prog.phase = CalibrationPhase::FORWARD_SWEEP;
                    g_shared.setCalibrationProgress(cal_prog);

                    if (!should_cancel()) {
                        progress_callback(0, "Phase 1: Duty calibration - forward sweep");
                        cal_result = g_vesc->runDutyCalibration([&](int p, const std::string& s) {
                            progress_callback(p / 4, "Forward: " + s);  // 0-25%
                        });

                        if (should_cancel()) {
                            success = false;
                        }
                    } else {
                        success = false;
                    }

                    // Phase 2: Reverse sweep (part of duty calibration)
                    if (success && !should_cancel()) {
                        cal_prog.phase = CalibrationPhase::REVERSE_SWEEP;
                        g_shared.setCalibrationProgress(cal_prog);
                        progress_callback(25, "Phase 2: Duty calibration - reverse sweep");
                        // Reverse is included in runDutyCalibration, just update progress
                        for (int i = 25; i <= 50 && !should_cancel(); i += 5) {
                            progress_callback(i, "Reverse sweep...");
                            std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        }
                    }

                    // Phase 3: Minimum duty calibration
                    if (success && !should_cancel()) {
                        cal_prog.phase = CalibrationPhase::TURNING_MIN_DUTY;
                        g_shared.setCalibrationProgress(cal_prog);
                        progress_callback(50, "Phase 3: Finding minimum duty thresholds");

                        auto min_result = g_vesc->runMinDutyCalibration([&](int p, const std::string& s) {
                            progress_callback(50 + p / 4, "Min duty: " + s);  // 50-75%
                        });

                        // Merge results
                        cal_result.left.min_duty_start = min_result.left.min_duty_start;
                        cal_result.right.min_duty_start = min_result.right.min_duty_start;
                    }

                    // Phase 4: Rotation calibration (if SLAM initialized for yaw reference)
                    if (success && !should_cancel()) {
                        cal_prog.phase = CalibrationPhase::ROTATION_SWEEP;
                        g_shared.setCalibrationProgress(cal_prog);
                        progress_callback(75, "Phase 4: Rotation calibration");

                        // Get yaw from SLAM if available, otherwise skip
                        if (g_slam && g_slam->isInitialized()) {
                            auto get_yaw = [&]() -> float {
                                auto pose = g_slam->getPose();
                                return static_cast<float>(std::atan2(pose(1,0), pose(0,0)));
                            };

                            auto rot_result = g_vesc->runRotationCalibration(get_yaw, [&](int p, const std::string& s) {
                                progress_callback(75 + p / 4, "Rotation: " + s);  // 75-100%
                            });

                            // Merge rotation results (stored at CalibrationResult level)
                            cal_result.ticks_per_radian = rot_result.ticks_per_radian;
                            cal_result.effective_track_mm = rot_result.effective_track_mm;
                            cal_result.geometry.effective_track_mm = rot_result.geometry.effective_track_mm;
                        } else {
                            progress_callback(90, "Skipping rotation cal (SLAM not initialized)");
                            std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        }
                    }

                    // Apply calibration if successful
                    if (success && !should_cancel()) {
                        g_vesc->applyCalibration(cal_result);

                        // Save to file
                        if (g_vesc->saveCalibration(g_config.calibration_file)) {
                            progress_callback(100, "Calibration saved to " + std::string(g_config.calibration_file));
                        } else {
                            progress_callback(100, "Calibration complete (save failed)");
                        }

                        // Reload into motion controller
                        if (g_motion) {
                            g_motion->loadCalibration(g_config.calibration_file);
                        }

                        cal_prog.success = true;
                        cal_prog.status_text = "Calibration complete!";
                    } else {
                        cal_prog.success = false;
                        cal_prog.status_text = "Calibration cancelled or failed";
                    }

                    cal_prog.running = false;
                    cal_prog.phase = CalibrationPhase::COMPLETE;
                    g_shared.setCalibrationProgress(cal_prog);
                    g_shared.setAppState(AppState::IDLE);

                    if (cal_prog.success) {
                        g_shared.setStatusMessage("Calibration completed successfully!");
                    } else {
                        g_shared.setStatusMessage("Calibration cancelled");
                    }
                    break;
                }

                case CommandType::CANCEL_CALIBRATION:
                    g_target_linear.store(0.0f);
                    g_target_angular.store(0.0f);
                    if (g_motion) g_motion->stop();
                    g_shared.setAppState(AppState::IDLE);
                    g_shared.setStatusMessage("Calibration cancelled");
                    break;

                case CommandType::CLEAR_MAP:
                    // Clear all stale sensor data
                    {
                        std::lock_guard<std::mutex> lock(g_completed_scans_mutex);
                        g_completed_scans.clear();
                    }
                    {
                        std::lock_guard<std::mutex> lock(g_imu_mutex);
                        g_imu_queue.clear();
                    }
                    g_accumulated_points.clear();

                    if (g_slam) {
                        g_slam->reset();
                    }
                    g_shared.clearTrajectory();
                    // Also clear the visible map points in SharedState
                    g_shared.setMapPoints({});
                    g_shared.setCurrentScan({});
                    g_shared.setStatusMessage("Map cleared");
                    break;

                // Note: STOP_LOCALIZATION is handled in the hardware-connected section above

                case CommandType::SET_POSE_HINT: {
                    auto* hint_cmd = std::get_if<PoseHintCommand>(&cmd->payload);
                    if (hint_cmd && g_slam) {
                        // Convert hint to 4x4 transform and set as initial guess
                        float theta = hint_cmd->theta_deg * 3.14159f / 180.0f;
                        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                        T(0,0) = std::cos(theta); T(0,1) = -std::sin(theta);
                        T(1,0) = std::sin(theta); T(1,1) = std::cos(theta);
                        T(0,3) = hint_cmd->x;
                        T(1,3) = hint_cmd->y;
                        g_slam->setInitialPose(T);
                    }
                    break;
                }

                case CommandType::START_RECORDING: {
                    auto* file_cmd = std::get_if<FileCommand>(&cmd->payload);
                    if (file_cmd) {
                        // TODO: Implement recording to file
                        g_shared.setRecordingStatus(true, file_cmd->path);
                        g_shared.setStatusMessage("Recording started: " + file_cmd->path);
                    }
                    break;
                }

                case CommandType::STOP_RECORDING:
                    // TODO: Stop actual recording
                    g_shared.setRecordingStatus(false);
                    g_shared.setStatusMessage("Recording stopped");
                    break;

                case CommandType::LOAD_HULL_MESH: {
                    auto* file_cmd = std::get_if<FileCommand>(&cmd->payload);
                    if (file_cmd) {
                        std::string path = file_cmd->path;
                        std::string ext = path.substr(path.find_last_of('.'));
                        bool loaded = false;
                        if (ext == ".obj" || ext == ".OBJ") {
                            loaded = g_hull.loadOBJ(path);
                        } else if (ext == ".stl" || ext == ".STL") {
                            loaded = g_hull.loadSTL(path);
                        }
                        if (loaded) {
                            g_shared.setStatusMessage("Hull mesh loaded");
                            HullMeshInfo info;
                            info.loaded = true;
                            info.filename = g_hull.getFilename();
                            info.vertex_count = static_cast<int>(g_hull.getVertexCount());
                            info.triangle_count = static_cast<int>(g_hull.getTriangleCount());
                            info.total_area_m2 = g_hull.getTotalArea();
                            g_shared.setHullMeshInfo(info);
                        } else {
                            g_shared.setErrorMessage("Failed to load hull mesh");
                        }
                    }
                    break;
                }

                case CommandType::CLEAR_COVERAGE:
                    g_hull.clearCoverage();
                    g_shared.setStatusMessage("Coverage cleared");
                    break;

                case CommandType::EXPORT_COVERAGE: {
                    auto* file_cmd = std::get_if<FileCommand>(&cmd->payload);
                    if (file_cmd && g_hull.isLoaded()) {
                        if (g_hull.exportCoverage(file_cmd->path)) {
                            g_shared.setStatusMessage("Coverage exported to " + file_cmd->path);
                        } else {
                            g_shared.setErrorMessage("Failed to export coverage");
                        }
                    }
                    break;
                }

                default:
                    break;
            }
        }

        // 3. Process sensor data
        // CRITICAL: SLAM must run during RELOCALIZING to build the local map for matching!
        AppState state = g_shared.getAppState();
        bool do_slam = (state == AppState::MAPPING || state == AppState::LOCALIZED ||
                        state == AppState::RELOCALIZING);

        if (do_slam && g_slam) {
            // Process IMU data
            {
                std::lock_guard<std::mutex> lock(g_imu_mutex);
                for (const auto& imu : g_imu_queue) {
                    g_slam->addImuData(imu);
                }
                g_imu_queue.clear();
            }

            // Process completed scans (accumulated in callback, matches live_slam.cpp)
            {
                std::lock_guard<std::mutex> lock(g_completed_scans_mutex);
                for (auto& cloud : g_completed_scans) {
                    if (!cloud.empty()) {
                        g_slam->addPointCloud(cloud);
                    }
                }
                g_completed_scans.clear();
            }

            // Run SLAM processing with timing
            auto slam_start = std::chrono::steady_clock::now();
            int processed = g_slam->process();
            auto slam_end = std::chrono::steady_clock::now();

            if (processed > 0) {
                g_slam_process_count++;  // Debug counter
                float slam_time_ms = std::chrono::duration<float, std::milli>(slam_end - slam_start).count();
                g_perf.recordSlamUpdate(slam_time_ms);
                g_perf.map_points.store(static_cast<int>(g_slam->getMapSize()));

                // Log SLAM timing for diagnostics
                if (g_diag_logger.isRunning()) {
                    g_diag_logger.logSlamTiming(
                        slam_time_ms,
                        0.0f, 0.0f, 0.0f, slam_time_ms, 0.0f,  // Detailed breakdown (approximated)
                        0, 0, 0, 0);  // Point counts (would need SLAM engine changes)

                    // Log map statistics
                    size_t map_size = g_slam->getMapSize();
                    float extent[6] = {0, 0, 0, 0, 0, 0};  // Would need SLAM engine query
                    g_diag_logger.logSlamMapStats(
                        static_cast<uint32_t>(map_size), 0, 0, extent, 0, 0);
                }
            }

            // Update buffer sizes for performance monitoring
            int imu_buf_size = 0, scan_buf_size = 0;
            {
                std::lock_guard<std::mutex> lock(g_imu_mutex);
                imu_buf_size = static_cast<int>(g_imu_queue.size());
                g_perf.buffer_imu.store(imu_buf_size);
            }
            {
                std::lock_guard<std::mutex> lock(g_completed_scans_mutex);
                scan_buf_size = static_cast<int>(g_completed_scans.size());
                g_perf.buffer_scans.store(scan_buf_size);
            }

            // Log buffer status periodically (10 Hz)
            static auto last_buffer_log = std::chrono::steady_clock::now();
            auto now_buf = std::chrono::steady_clock::now();
            if (g_diag_logger.isRunning() &&
                std::chrono::duration<float>(now_buf - last_buffer_log).count() >= 0.1f) {
                last_buffer_log = now_buf;
                g_diag_logger.logBufferStatus(
                    static_cast<uint32_t>(imu_buf_size),
                    static_cast<uint32_t>(scan_buf_size),
                    static_cast<uint32_t>(g_commands.size()));
            }

            // Update SLAM state to GUI
            UpdateSlamState();

            // Update map points for 3D visualization (throttled internally)
            UpdateMapPointsForVisualization();

            // Check progressive localization if in RELOCALIZING state
            if (g_shared.getAppState() == AppState::RELOCALIZING) {
                // Check localization status
                slam::LocalizationResult loc_result = g_slam->checkProgressiveLocalization();

                // Update progress in shared state
                RelocalizationProgress prog = g_shared.getRelocalizationProgress();
                prog.running = true;
                prog.local_map_voxels = loc_result.local_map_voxels;
                prog.local_map_points = loc_result.local_map_points;
                prog.rotation_deg = static_cast<float>(loc_result.rotation_deg);
                prog.distance_m = static_cast<float>(loc_result.distance_m);
                prog.attempt_number = loc_result.attempt_number;

                // Calculate progress based on coverage
                float voxel_progress = static_cast<float>(loc_result.local_map_voxels) / 800.0f;
                float rotation_progress = static_cast<float>(loc_result.rotation_deg) / 90.0f;
                prog.progress = std::min(1.0f, std::max(voxel_progress, rotation_progress));

                // Update status text based on state
                switch (loc_result.status) {
                    case slam::LocalizationStatus::ACCUMULATING:
                        prog.accumulating = true;
                        prog.ready_to_localize = false;
                        prog.icp_running = false;
                        prog.status_text = "Building map: " +
                            std::to_string(loc_result.local_map_voxels) + " voxels, " +
                            std::to_string(int(loc_result.rotation_deg)) + " deg rotation";
                        break;

                    case slam::LocalizationStatus::VIEW_SATURATED:
                        prog.accumulating = true;
                        prog.ready_to_localize = false;
                        prog.icp_running = false;
                        prog.status_text = "View saturated - rotate robot for more coverage";
                        break;

                    case slam::LocalizationStatus::READY_FOR_LOCALIZATION:
                        // Coverage is sufficient - robot should stop and user clicks "Localize Now"
                        prog.accumulating = false;
                        prog.ready_to_localize = true;
                        prog.icp_running = false;
                        prog.status_text = "Ready! Stop robot and click 'Localize Now'";
                        break;

                    case slam::LocalizationStatus::AWAITING_ROBOT_STOP:
                        prog.accumulating = false;
                        prog.ready_to_localize = true;
                        prog.icp_running = false;
                        prog.status_text = "Waiting for robot to stop...";
                        break;

                    case slam::LocalizationStatus::ATTEMPTING:
                        prog.accumulating = false;
                        prog.ready_to_localize = false;
                        prog.icp_running = true;
                        prog.status_text = "Running ICP alignment...";
                        break;

                    case slam::LocalizationStatus::LOW_CONFIDENCE:
                        prog.accumulating = false;
                        prog.ready_to_localize = true;  // Can try again
                        prog.icp_running = false;
                        prog.status_text = "Low confidence (" + std::to_string(int(loc_result.confidence * 100)) +
                            "%) - try again or move robot";
                        break;

                    case slam::LocalizationStatus::SUCCESS:
                        // Localization succeeded!
                        prog.running = false;
                        prog.accumulating = false;
                        prog.success = true;
                        prog.confidence = static_cast<float>(loc_result.confidence);
                        prog.status_text = "Localized! (confidence: " +
                            std::to_string(int(loc_result.confidence * 100)) + "%)";

                        // Swap to pre-built map for continued tracking
                        g_slam->swapToPrebuiltMap(loc_result.transform);

                        g_shared.setAppState(AppState::LOCALIZED);
                        g_shared.setStatusMessage("Localization successful - tracking on pre-built map");
                        break;

                    case slam::LocalizationStatus::FAILED:
                        // Max attempts reached
                        prog.running = false;
                        prog.accumulating = false;
                        prog.success = false;
                        prog.status_text = loc_result.message;

                        g_shared.setAppState(AppState::RELOCALIZE_FAILED);
                        g_shared.setStatusMessage("Localization failed after " +
                            std::to_string(loc_result.attempt_number) + " attempts - try different position");
                        break;

                    default:
                        break;
                }

                g_shared.setRelocalizationProgress(prog);
            }
        }

        // 4. Update wheel odometry and sensor fusion
        // Read from cache (updated by ControlThread) - avoids cross-thread VESC access
        if (g_fusion) {
            slam::VescOdometry odom;
            slam::VescStatus status_l, status_r;

            // Read from cache (single-thread VESC architecture)
            {
                std::lock_guard<std::mutex> lock(g_vesc_cache_mutex);
                odom = g_vesc_odom_cache;
                status_l = g_vesc_status_left_cache;
                status_r = g_vesc_status_right_cache;
            }

            // Calculate linear velocity from wheel velocities
            float linear_vel = (odom.velocity_left_mps + odom.velocity_right_mps) / 2.0f;

            // Get actual angular velocity from motion controller (not the command!)
            // This uses wheel differential for more accurate heading estimation
            float actual_angular = 0.0f;
            if (g_motion) {
                slam::Velocity2D vel = g_motion->getVelocity();
                actual_angular = vel.angular;  // rad/s from wheel differential
            } else {
                actual_angular = g_target_angular.load();  // Fallback to command
            }

            // Update sensor fusion with actual measured velocities
            g_fusion->updateWheelOdometry(
                linear_vel,
                actual_angular,  // Use measured angular, not commanded
                status_l.erpm,
                status_r.erpm,
                dt
            );

            // Get fused output
            slam::Pose3D fused = g_fusion->getFusedPose();
            slam::Velocity3D fused_vel = g_fusion->getFusedVelocity();
            slam::MotionState motion = g_fusion->getMotionState();

            // Update shared state
            Pose3D gui_fused;
            gui_fused.x = fused.x;
            gui_fused.y = fused.y;
            gui_fused.z = fused.z;
            gui_fused.roll = fused.roll;
            gui_fused.pitch = fused.pitch;
            gui_fused.yaw = fused.yaw;
            g_shared.setFusedPose(gui_fused);

            Velocity2D gui_vel;
            gui_vel.linear = fused_vel.linear_x;
            gui_vel.angular = fused_vel.angular_z;
            g_shared.setVelocity(gui_vel);

            // Convert motion state
            switch (motion) {
                case slam::MotionState::STATIONARY:
                    g_shared.setMotionState(MotionState::STATIONARY);
                    break;
                case slam::MotionState::STRAIGHT_LINE:
                    g_shared.setMotionState(MotionState::STRAIGHT_LINE);
                    break;
                case slam::MotionState::TURNING:
                    g_shared.setMotionState(MotionState::TURNING);
                    break;
            }

            // Update wheel odom pose for diagnostics
            slam::Pose2D wheel_pose = g_motion ? g_motion->getPose() : slam::Pose2D{};
            Pose3D gui_wheel;
            gui_wheel.x = wheel_pose.x;
            gui_wheel.y = wheel_pose.y;
            gui_wheel.yaw = wheel_pose.theta;
            g_shared.setWheelOdomPose(gui_wheel);

            // Log fused pose and wheel odometry for diagnostics
            if (g_diag_logger.isRunning()) {
                // Log fused pose
                int motion_int = (motion == slam::MotionState::STATIONARY) ? 0 :
                                 (motion == slam::MotionState::STRAIGHT_LINE) ? 1 : 2;
                g_diag_logger.logFusedPose(
                    fused.x, fused.y, fused.z,
                    fused.roll, fused.pitch, fused.yaw,
                    fused_vel.linear_x, fused_vel.angular_z,
                    motion_int, 0.5f, 0.5f);  // Weights not exposed yet

                // Log wheel odometry (simplified - get from VESC cache)
                g_diag_logger.logWheelOdom(
                    wheel_pose.x, wheel_pose.y, wheel_pose.theta,
                    static_cast<float>(gui_vel.linear), static_cast<float>(gui_vel.angular),
                    0, 0,  // Tach values from VESC status logged separately
                    0.0f, 0.0f);  // Distance not tracked here
            }

            // Add trajectory point
            if (state == AppState::MAPPING || state == AppState::LOCALIZED || state == AppState::OPERATING) {
                g_shared.addTrajectoryPoint(gui_fused);
            }

            // Update hull coverage
            if (g_hull.isLoaded() && g_config.probe.enabled) {
                g_hull.updateCoverage(fused.x, fused.y, fused.yaw, g_config.probe, g_current_time);

                HullMeshInfo info;
                info.loaded = true;
                info.filename = g_hull.getFilename();
                info.vertex_count = static_cast<int>(g_hull.getVertexCount());
                info.triangle_count = static_cast<int>(g_hull.getTriangleCount());
                info.coverage_percent = g_hull.getCoveragePercent() * 100.0f;
                info.covered_area_m2 = g_hull.getCoveredArea();
                info.total_area_m2 = g_hull.getTotalArea();
                g_shared.setHullMeshInfo(info);
            }
        }

        // 5. Update g_can_drive for control thread
        // (Motor status is updated by control thread)
        bool can_drive = (state == AppState::MAPPING || state == AppState::LOCALIZED ||
                          state == AppState::OPERATING || state == AppState::RELOCALIZING);
        g_can_drive.store(can_drive);

        // Note: Motor control has moved to high-priority ControlThread
        // Velocity targets are set directly by main thread via g_target_linear/angular

        // 6. Update LiDAR status
        if (g_lidar) {
            LidarStatus lidar_status;
            lidar_status.connection = g_lidar->isConnected() ? ConnectionStatus::CONNECTED : ConnectionStatus::DISCONNECTED;
            lidar_status.ip_address = g_config.lidar_ip;
            // TODO: Get actual rates from driver
            lidar_status.point_rate = 200000;
            lidar_status.imu_rate = 200;
            lidar_status.imu_initialized = g_slam ? g_slam->isInitialized() : false;
            if (g_slam && g_slam->isInitialized()) {
                slam::SlamState slam_state = g_slam->getState();
                lidar_status.gravity_mag = static_cast<float>(slam_state.gravity.norm());
            }
            g_shared.setLidarStatus(lidar_status);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));  // 50Hz loop
    }

    // Cleanup on exit
    ShutdownHardware();
}

#else  // !ENABLE_HARDWARE - Simulation mode

void WorkerThread() {
    g_shared.setAppState(AppState::STARTUP);
    g_shared.setStatusMessage("Initializing...");

    // Simulation mode initialization
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    g_shared.setAppState(AppState::IDLE);
    g_shared.setStatusMessage("Running in simulation mode");
    g_shared.hardware_connected.store(false);
    g_hardware_initialized.store(true);

    auto last_update = std::chrono::steady_clock::now();

    // Simulation state
    float sim_x = 0.0f, sim_y = 0.0f, sim_yaw = 0.0f;

    while (g_running.load()) {
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_update).count();
        last_update = now;

        // 1. Check E-STOP first
        if (g_e_stop.load()) {
            g_shared.setAppState(AppState::E_STOPPED);

            // Wait for E-STOP reset
            while (g_e_stop.load() && g_running.load()) {
                while (auto cmd = g_commands.pop()) {
                    if (cmd->type == CommandType::RESET_E_STOP) {
                        g_e_stop.store(false);
                        g_shared.e_stop.store(false);
                        g_shared.setAppState(AppState::IDLE);
                        g_shared.setStatusMessage("E-STOP reset");
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            continue;
        }

        // 2. Process commands from GUI
        while (auto cmd = g_commands.pop()) {
            switch (cmd->type) {
                case CommandType::E_STOP:
                    g_e_stop.store(true);
                    g_shared.e_stop.store(true);
                    break;

                case CommandType::RESET_E_STOP:
                    g_e_stop.store(false);
                    g_shared.e_stop.store(false);
                    g_shared.setAppState(AppState::IDLE);
                    break;

                case CommandType::START_MAPPING:
                    g_shared.setAppState(AppState::MAPPING);
                    g_shared.setStatusMessage("Mapping started (simulation)");
                    g_shared.clearTrajectory();
                    sim_x = sim_y = sim_yaw = 0.0f;
                    break;

                case CommandType::STOP_MAPPING:
                    g_shared.setAppState(AppState::IDLE);
                    g_shared.setStatusMessage("Mapping stopped");
                    break;

                case CommandType::START_OPERATING:
                    g_shared.setAppState(AppState::OPERATING);
                    g_shared.setStatusMessage("Operating mode (simulation)");
                    break;

                case CommandType::STOP_OPERATING:
                    g_shared.setAppState(AppState::IDLE);
                    g_shared.setStatusMessage("Operating stopped");
                    break;

                case CommandType::SAVE_MAP: {
                    auto* file_cmd = std::get_if<FileCommand>(&cmd->payload);
                    if (file_cmd) {
                        g_shared.setAppState(AppState::SAVE_MAP);
                        g_shared.setStatusMessage("Saving map (simulation)...");
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        g_shared.setStatusMessage("Map saved to " + file_cmd->path);
                        g_shared.setAppState(AppState::IDLE);
                    }
                    break;
                }

                case CommandType::LOAD_MAP: {
                    auto* file_cmd = std::get_if<FileCommand>(&cmd->payload);
                    if (file_cmd) {
                        g_shared.setStatusMessage("Loading map (simulation)...");
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        g_shared.setAppState(AppState::MAP_LOADED);
                        g_shared.setStatusMessage("Map loaded: " + file_cmd->path);
                    }
                    break;
                }

                case CommandType::RELOCALIZE:
                    if (g_shared.getAppState() == AppState::MAP_LOADED) {
                        g_shared.setAppState(AppState::RELOCALIZING);
                        g_shared.setStatusMessage("Running global localization (simulation)...");

                        RelocalizationProgress prog;
                        prog.running = true;
                        prog.progress = 0.5f;
                        prog.status_text = "Localizing...";
                        g_shared.setRelocalizationProgress(prog);

                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

                        g_shared.setAppState(AppState::LOCALIZED);
                        g_shared.setStatusMessage("Localized successfully (simulation)");

                        prog.running = false;
                        prog.success = true;
                        prog.confidence = 0.95f;
                        prog.status_text = "Localized";
                        g_shared.setRelocalizationProgress(prog);
                    }
                    break;

                case CommandType::RUN_CALIBRATION: {
                    g_shared.setAppState(AppState::CALIBRATING);
                    g_shared.setStatusMessage("Calibration running (simulation)...");

                    CalibrationProgress cal_prog;
                    cal_prog.running = true;
                    cal_prog.phase = CalibrationPhase::FORWARD_SWEEP;
                    cal_prog.progress = 0.0f;
                    cal_prog.status_text = "Forward sweep";
                    g_shared.setCalibrationProgress(cal_prog);

                    // Simulate calibration progress
                    for (int i = 0; i <= 100 && g_running.load() && !g_e_stop.load(); i += 5) {
                        cal_prog.progress = i / 100.0f;
                        if (i < 25) {
                            cal_prog.phase = CalibrationPhase::FORWARD_SWEEP;
                            cal_prog.status_text = "Forward sweep";
                        } else if (i < 50) {
                            cal_prog.phase = CalibrationPhase::REVERSE_SWEEP;
                            cal_prog.status_text = "Reverse sweep";
                        } else if (i < 75) {
                            cal_prog.phase = CalibrationPhase::ROTATION_SWEEP;
                            cal_prog.status_text = "Rotation sweep";
                        } else {
                            cal_prog.phase = CalibrationPhase::TURNING_MIN_DUTY;
                            cal_prog.status_text = "Finding min duty";
                        }
                        g_shared.setCalibrationProgress(cal_prog);
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }

                    cal_prog.running = false;
                    cal_prog.success = true;
                    cal_prog.phase = CalibrationPhase::COMPLETE;
                    cal_prog.progress = 1.0f;
                    cal_prog.status_text = "Complete";
                    g_shared.setCalibrationProgress(cal_prog);

                    g_shared.setAppState(AppState::IDLE);
                    g_shared.setStatusMessage("Calibration complete (simulation)");
                    break;
                }

                case CommandType::CANCEL_CALIBRATION:
                    g_shared.setAppState(AppState::IDLE);
                    g_shared.setStatusMessage("Calibration cancelled");
                    break;

                case CommandType::LOAD_HULL_MESH: {
                    auto* file_cmd = std::get_if<FileCommand>(&cmd->payload);
                    if (file_cmd) {
                        std::string path = file_cmd->path;
                        std::string ext = path.substr(path.find_last_of('.'));
                        bool loaded = false;
                        if (ext == ".obj" || ext == ".OBJ") {
                            loaded = g_hull.loadOBJ(path);
                        } else if (ext == ".stl" || ext == ".STL") {
                            loaded = g_hull.loadSTL(path);
                        }
                        if (loaded) {
                            g_shared.setStatusMessage("Hull mesh loaded");
                            HullMeshInfo info;
                            info.loaded = true;
                            info.filename = g_hull.getFilename();
                            info.vertex_count = static_cast<int>(g_hull.getVertexCount());
                            info.triangle_count = static_cast<int>(g_hull.getTriangleCount());
                            info.total_area_m2 = g_hull.getTotalArea();
                            g_shared.setHullMeshInfo(info);
                        } else {
                            g_shared.setErrorMessage("Failed to load hull mesh");
                        }
                    }
                    break;
                }

                case CommandType::CLEAR_COVERAGE:
                    g_hull.clearCoverage();
                    g_shared.setStatusMessage("Coverage cleared");
                    break;

                case CommandType::EXPORT_COVERAGE: {
                    auto* file_cmd = std::get_if<FileCommand>(&cmd->payload);
                    if (file_cmd && g_hull.isLoaded()) {
                        if (g_hull.exportCoverage(file_cmd->path)) {
                            g_shared.setStatusMessage("Coverage exported to " + file_cmd->path);
                        } else {
                            g_shared.setErrorMessage("Failed to export coverage");
                        }
                    }
                    break;
                }

                default:
                    break;
            }
        }

        // 3. Simulation: Update pose based on gamepad input
        AppState state = g_shared.getAppState();
        bool can_drive = (state == AppState::MAPPING || state == AppState::LOCALIZED ||
                          state == AppState::OPERATING || state == AppState::RELOCALIZING);
        if (can_drive) {
            // Use pre-computed drive command from getDriveCommand() (matches manual_drive.cpp)
            float linear = g_drive_cmd.linear_velocity;
            float angular = g_drive_cmd.angular_velocity;

            sim_yaw += angular * dt;
            sim_x += linear * std::cos(sim_yaw) * dt;
            sim_y += linear * std::sin(sim_yaw) * dt;

            Pose3D pose;
            pose.x = sim_x;
            pose.y = sim_y;
            pose.yaw = sim_yaw;
            g_shared.setFusedPose(pose);
            g_shared.setSlamPose(pose);
            g_shared.addTrajectoryPoint(pose);

            Velocity2D vel;
            vel.linear = linear;
            vel.angular = angular;
            g_shared.setVelocity(vel);

            // Update motion state
            if (std::abs(linear) < 0.01f && std::abs(angular) < 0.05f) {
                g_shared.setMotionState(MotionState::STATIONARY);
            } else if (std::abs(angular) > 0.2f) {
                g_shared.setMotionState(MotionState::TURNING);
            } else {
                g_shared.setMotionState(MotionState::STRAIGHT_LINE);
            }

            // Update hull coverage
            if (g_hull.isLoaded() && g_config.probe.enabled) {
                g_hull.updateCoverage(sim_x, sim_y, sim_yaw, g_config.probe, g_current_time);

                HullMeshInfo info;
                info.loaded = true;
                info.filename = g_hull.getFilename();
                info.vertex_count = static_cast<int>(g_hull.getVertexCount());
                info.triangle_count = static_cast<int>(g_hull.getTriangleCount());
                info.coverage_percent = g_hull.getCoveragePercent() * 100.0f;
                info.covered_area_m2 = g_hull.getCoveredArea();
                info.total_area_m2 = g_hull.getTotalArea();
                g_shared.setHullMeshInfo(info);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));  // 50Hz loop
    }
}

#endif  // ENABLE_HARDWARE

//==============================================================================
// UI: Status Bar
//==============================================================================
void DrawStatusBar() {
    ImGuiViewport* viewport = ImGui::GetMainViewport();

    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x, viewport->Pos.y));
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x, 40));
    ImGui::SetNextWindowBgAlpha(0.9f);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoSavedSettings;

    ImGui::Begin("##StatusBar", nullptr, flags);

    // LiDAR status (clickable for diagnostics)
    LidarStatus lidar = g_shared.getLidarStatus();
    // More detailed status: green only if actually streaming data
    bool lidar_streaming = g_lidar && g_lidar->isStreaming();
    bool lidar_connected = (lidar.connection == ConnectionStatus::CONNECTED);
    ImVec4 lidar_color;
    const char* lidar_tooltip;
    if (lidar_streaming) {
        lidar_color = ImVec4(0.2f, 1.0f, 0.2f, 1.0f);  // Green - fully working
        lidar_tooltip = "LiDAR streaming data - click for diagnostics";
    } else if (lidar_connected) {
        lidar_color = ImVec4(1.0f, 1.0f, 0.2f, 1.0f);  // Yellow - connected but not streaming
        lidar_tooltip = "LiDAR connected but NOT streaming - click for diagnostics";
    } else {
        lidar_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);  // Gray - disconnected
        lidar_tooltip = "LiDAR disconnected - click for diagnostics";
    }

    // Make LiDAR text clickable
    ImGui::PushStyleColor(ImGuiCol_Text, lidar_color);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.4f, 0.5f));
    if (ImGui::SmallButton("LiDAR")) {
        g_show_lidar_diagnostics = true;
    }
    ImGui::PopStyleColor(4);
    SetTooltip(lidar_tooltip);
    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();

    // VESC status (clickable for diagnostics when disconnected)
    MotorStatus motor_l = g_shared.getMotorStatus(0);
    MotorStatus motor_r = g_shared.getMotorStatus(1);
    bool vesc_ok = motor_l.connected || motor_r.connected;
    ImVec4 vesc_color = vesc_ok ? ImVec4(0.2f, 1.0f, 0.2f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);

    // Make VESC text clickable
    ImGui::PushStyleColor(ImGuiCol_Text, vesc_color);
    if (!vesc_ok) {
        // Show as clickable when disconnected
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.4f, 0.5f));
        if (ImGui::SmallButton("VESC")) {
            // Run VESC diagnostics
            g_show_vesc_diagnostics = true;
        }
        ImGui::PopStyleColor(3);
        SetTooltip("VESC disconnected - click to run diagnostics");
    } else {
        ImGui::Text("VESC");
        SetTooltip("Motor controller connected.");
    }
    ImGui::PopStyleColor();
    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();

    // Battery voltage and SOC% (5S Li-ion: 21V=100%, 14V=0%)
    float voltage = std::max(motor_l.voltage_in, motor_r.voltage_in);
    float soc_percent = 0.0f;
    if (voltage > 0.0f) {
        soc_percent = std::clamp((voltage - 14.0f) / (21.0f - 14.0f) * 100.0f, 0.0f, 100.0f);
    }
    ImVec4 batt_color = (soc_percent > 60.0f) ? ImVec4(0.2f, 1.0f, 0.2f, 1.0f) :
                        (soc_percent > 30.0f) ? ImVec4(1.0f, 1.0f, 0.2f, 1.0f) :
                                                ImVec4(1.0f, 0.3f, 0.2f, 1.0f);
    ImGui::TextColored(batt_color, "%.1fV (%.0f%%)", voltage > 0 ? voltage : 0.0f, soc_percent);
    SetTooltip("Battery: 5S Li-ion pack. 21V=100%%, 14V=0%%. Green>60%%, Yellow>30%%, Red<30%%.");
    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();

    // Gamepad status with battery
    ImVec4 gp_color = g_gamepad_connected ? ImVec4(0.2f, 1.0f, 0.2f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
#ifdef HAS_SDL2
    int gp_battery = g_gamepad ? g_gamepad->getBatteryLevel() : -1;
    if (g_gamepad_connected && gp_battery >= 0) {
        // Show gamepad with battery level
        ImVec4 gp_batt_color = (gp_battery > 50) ? ImVec4(0.2f, 1.0f, 0.2f, 1.0f) :
                               (gp_battery > 20) ? ImVec4(1.0f, 1.0f, 0.2f, 1.0f) :
                                                   ImVec4(1.0f, 0.3f, 0.2f, 1.0f);
        ImGui::TextColored(gp_color, "Gamepad");
        ImGui::SameLine(0, 2);
        ImGui::TextColored(gp_batt_color, "(%d%%)", gp_battery);
        char gp_tooltip[64];
        snprintf(gp_tooltip, sizeof(gp_tooltip), "Gamepad connected. Battery: %d%%", gp_battery);
        SetTooltip(gp_tooltip);
    } else {
        ImGui::TextColored(gp_color, "Gamepad");
        SetTooltip("Xbox/PS5 controller. Use for manual robot control.");
    }
#else
    ImGui::TextColored(gp_color, "Gamepad");
    SetTooltip("Xbox/PS5 controller. Use for manual robot control.");
#endif
    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();

    // Application state
    AppState state = g_shared.getAppState();
    float r, g, b, a;
    getStateColor(state, r, g, b, a);
    ImGui::TextColored(ImVec4(r, g, b, a), "%s", getStateName(state));
    SetTooltip("Current application mode.");
    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();

    // System Log button - show count of errors/warnings
    int error_count = 0, warn_count = 0;
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        for (const auto& entry : g_system_log) {
            if (entry.level == LogLevel::ERROR_LEVEL) error_count++;
            else if (entry.level == LogLevel::WARNING) warn_count++;
        }
    }
    if (error_count > 0) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.1f, 0.1f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
    } else if (warn_count > 0) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.5f, 0.1f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.6f, 0.2f, 1.0f));
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
    }
    if (ImGui::SmallButton("Log")) {
        g_show_system_log = true;
    }
    ImGui::PopStyleColor(2);
    char log_tooltip[64];
    snprintf(log_tooltip, sizeof(log_tooltip), "System Log: %d errors, %d warnings. Click to view.", error_count, warn_count);
    SetTooltip(log_tooltip);

    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();

    // Diagnostic Recording button (REC)
    bool is_logging = g_diag_logger.isRunning();
    if (is_logging) {
        // Blinking red when recording
        float blink = (std::sin(g_current_time * 6.0f) + 1.0f) * 0.5f;
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f * blink, 0.1f * blink, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.2f, 0.2f, 1.0f));
        if (ImGui::SmallButton("REC")) {
            g_diag_logger.stop();
            g_logging_enabled = false;
            AddLogEntry(LogLevel::INFO, "Logger", "Diagnostic recording stopped");
        }
        ImGui::PopStyleColor(2);
        char rec_tooltip[128];
        snprintf(rec_tooltip, sizeof(rec_tooltip), "RECORDING: %.1fs, %.1f MB, %zu msgs. Click to stop.",
                 g_diag_logger.getElapsedSeconds(),
                 static_cast<float>(g_diag_logger.getFileSizeMB()),
                 g_diag_logger.getMessageCount());
        SetTooltip(rec_tooltip);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.2f, 0.2f, 1.0f));
        if (ImGui::SmallButton("REC")) {
            // Create logs directory if needed
            std::filesystem::create_directories("logs");

            LoggingConfig config;
            config.output_dir = "logs";
            config.point_downsample = 10;  // Log every 10th point
            config.imu_downsample = 5;     // Log every 5th IMU sample
            if (g_diag_logger.start(g_log_session_name, config)) {
                g_logging_enabled = true;
                AddLogEntry(LogLevel::INFO, "Logger", "Diagnostic recording started: " + g_diag_logger.getFilePath());
            } else {
                AddLogEntry(LogLevel::ERROR_LEVEL, "Logger", "Failed to start diagnostic recording");
            }
        }
        ImGui::PopStyleColor(2);
        SetTooltip("Start diagnostic recording. Logs all sensor data, SLAM state, and performance metrics.");
    }

    // E-STOP button on the right
    ImGui::SameLine(ImGui::GetWindowWidth() - 120);

    bool e_stopped = g_e_stop.load();
    if (e_stopped) {
        // Reset button when E-STOPped
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.6f, 0.1f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.7f, 0.2f, 1.0f));
        if (ImGui::Button("RESET", ImVec2(100, 28))) {
            g_commands.push(Command::simple(CommandType::RESET_E_STOP));
        }
        SetTooltip("Reset E-STOP and resume operation.");
        ImGui::PopStyleColor(2);
    } else {
        // E-STOP button
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.1f, 0.1f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
        if (ImGui::Button("E-STOP", ImVec2(100, 28))) {
            g_e_stop.store(true);
            g_shared.e_stop.store(true);
            g_commands.push(Command::simple(CommandType::E_STOP));
        }
        SetTooltip("EMERGENCY STOP - Immediately stops all motors. Press ESC or B on gamepad.");
        ImGui::PopStyleColor(2);
    }

    ImGui::End();
}

//==============================================================================
// UI: Tab - Operate
//==============================================================================
void DrawOperateTab() {
    ImGui::BeginChild("OperateContent", ImVec2(0, 0), false);

    // Split into left (3D view placeholder) and right (status) panels
    float panel_width = 300;

    // Left side - 3D viewer placeholder
    ImGui::BeginChild("Viewer", ImVec2(-panel_width - 10, 0), true);
    ImGui::Text("3D Viewer");
    ImGui::Separator();

    // Camera mode buttons
    ImGui::Text("Camera:");
    ImGui::SameLine();
    if (ImGui::RadioButton("Free", g_config.camera_mode == CameraMode::FREE)) {
        g_config.camera_mode = CameraMode::FREE;
    }
    SetTooltip("Orbit, pan, and zoom freely around the map.");
    ImGui::SameLine();
    if (ImGui::RadioButton("Follow", g_config.camera_mode == CameraMode::FOLLOW)) {
        g_config.camera_mode = CameraMode::FOLLOW;
    }
    SetTooltip("Camera follows behind the robot.");
    ImGui::SameLine();
    if (ImGui::RadioButton("Top", g_config.camera_mode == CameraMode::TOP_DOWN)) {
        g_config.camera_mode = CameraMode::TOP_DOWN;
    }
    SetTooltip("Bird's eye view from above, centered on robot.");

    // 3D Map View with perspective (using ImGui drawing)
    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImGui::BeginChild("MapView3D", avail, true, ImGuiWindowFlags_NoScrollbar);

    // Static view state for pan/zoom/rotation
    static float view_scale = 50.0f;    // pixels per meter
    static float view_offset_x = 0.0f;  // World X offset
    static float view_offset_y = 0.0f;  // World Y offset
    static float view_pitch = 90.0f;    // Camera pitch (90=top-down, 30=tilted)
    static float view_yaw = 0.0f;       // Camera yaw rotation
    static bool follow_robot = true;

    // Camera presets based on mode
    if (g_config.camera_mode == CameraMode::TOP_DOWN) {
        view_pitch = 90.0f;  // Pure top-down
    } else if (g_config.camera_mode == CameraMode::FOLLOW) {
        view_pitch = 45.0f;  // Tilted follow view
    }

    // Get canvas info
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    ImVec2 canvas_center = ImVec2(canvas_pos.x + canvas_size.x / 2,
                                   canvas_pos.y + canvas_size.y / 2);

    // Get draw list
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Clip to canvas
    draw_list->PushClipRect(canvas_pos,
                            ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                            true);

    // Background (darker for 3D effect)
    draw_list->AddRectFilled(canvas_pos,
                             ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                             IM_COL32(15, 15, 20, 255));

    // Get robot pose
    Pose3D pose = g_shared.getFusedPose();

    // Follow robot mode - center on robot
    if (follow_robot || g_config.camera_mode == CameraMode::FOLLOW || g_config.camera_mode == CameraMode::TOP_DOWN) {
        view_offset_x = -pose.x;
        view_offset_y = -pose.y;
    }

    // 3D to 2D projection with pitch and yaw
    // Uses simple perspective: higher pitch = more top-down, lower = tilted
    float pitch_rad = view_pitch * 3.14159f / 180.0f;
    float yaw_rad = view_yaw * 3.14159f / 180.0f;
    float cos_pitch = std::cos(pitch_rad);
    float sin_pitch = std::sin(pitch_rad);
    float cos_yaw = std::cos(yaw_rad);
    float sin_yaw = std::sin(yaw_rad);

    // Lambda to convert world 3D coords to screen coords
    auto worldToScreen3D = [&](float wx, float wy, float wz) -> ImVec2 {
        // Translate to view center
        float tx = wx + view_offset_x;
        float ty = wy + view_offset_y;

        // Apply yaw rotation (around Z axis)
        float rx = tx * cos_yaw - ty * sin_yaw;
        float ry = tx * sin_yaw + ty * cos_yaw;
        float rz = wz;

        // Apply pitch rotation (tilt the view)
        // When pitch=90: Y maps to screen Y, Z ignored (top-down)
        // When pitch=0: Z maps to screen Y (side view)
        float screen_x = rx;
        float screen_y = ry * sin_pitch + rz * cos_pitch;

        // Scale and convert to screen coordinates
        float sx = canvas_center.x + screen_x * view_scale;
        float sy = canvas_center.y - screen_y * view_scale;
        return ImVec2(sx, sy);
    };

    // 2D wrapper for backward compatibility (z=0)
    auto worldToScreen = [&](float wx, float wy) -> ImVec2 {
        return worldToScreen3D(wx, wy, 0.0f);
    };

    // Draw grid (on ground plane z=0)
    float grid_spacing = 1.0f;  // 1 meter grid
    if (view_scale < 20.0f) grid_spacing = 5.0f;
    if (view_scale > 100.0f) grid_spacing = 0.5f;

    float view_width = canvas_size.x / view_scale;
    float view_height = canvas_size.y / view_scale;
    float min_x = -view_offset_x - view_width / 2;
    float max_x = -view_offset_x + view_width / 2;
    float min_y = -view_offset_y - view_height / 2;
    float max_y = -view_offset_y + view_height / 2;

    // Vertical grid lines (using 3D projection at z=0)
    for (float x = std::floor(min_x / grid_spacing) * grid_spacing; x <= max_x; x += grid_spacing) {
        ImVec2 p1 = worldToScreen3D(x, min_y, 0.0f);
        ImVec2 p2 = worldToScreen3D(x, max_y, 0.0f);
        draw_list->AddLine(p1, p2, IM_COL32(40, 40, 50, 255), 1.0f);
    }
    // Horizontal grid lines
    for (float y = std::floor(min_y / grid_spacing) * grid_spacing; y <= max_y; y += grid_spacing) {
        ImVec2 p1 = worldToScreen3D(min_x, y, 0.0f);
        ImVec2 p2 = worldToScreen3D(max_x, y, 0.0f);
        draw_list->AddLine(p1, p2, IM_COL32(40, 40, 50, 255), 1.0f);
    }

    // Draw origin axes (at z=0)
    ImVec2 origin = worldToScreen3D(0, 0, 0);
    draw_list->AddLine(worldToScreen3D(-100, 0, 0), worldToScreen3D(100, 0, 0), IM_COL32(80, 40, 40, 255), 1.0f);  // X axis (red)
    draw_list->AddLine(worldToScreen3D(0, -100, 0), worldToScreen3D(0, 100, 0), IM_COL32(40, 80, 40, 255), 1.0f);  // Y axis (green)
    // Add Z axis indicator when tilted
    if (view_pitch < 85.0f) {
        draw_list->AddLine(worldToScreen3D(0, 0, 0), worldToScreen3D(0, 0, 3), IM_COL32(40, 40, 100, 255), 1.0f);  // Z axis (blue)
    }

    // Get and draw map points (cached locally for performance)
    // WARNING: Drawing many points via ImGui is CPU-intensive! Use g_config.operate_show_map to disable
    static std::vector<RenderPoint> cached_map_points;
    static std::vector<RenderPoint> cached_scan_points;
    g_shared.getMapPointsIfUpdated(cached_map_points);
    g_shared.getCurrentScanIfUpdated(cached_scan_points);

    // Draw map points only if enabled (disabled by default for performance)
    if (g_config.operate_show_map && !cached_map_points.empty()) {
        // Limit points for performance
        int step = std::max(1, (int)cached_map_points.size() / g_config.operate_max_points);
        int rendered = 0;
        for (size_t i = 0; i < cached_map_points.size() && rendered < g_config.operate_max_points; i += step) {
            const auto& pt = cached_map_points[i];
            ImVec2 screen_pos = worldToScreen3D(pt.x, pt.y, pt.z);
            // Skip points outside view
            if (screen_pos.x < canvas_pos.x - 5 || screen_pos.x > canvas_pos.x + canvas_size.x + 5 ||
                screen_pos.y < canvas_pos.y - 5 || screen_pos.y > canvas_pos.y + canvas_size.y + 5) continue;
            // Vary point size slightly based on Z for depth cue
            float z_factor = 1.0f + (pt.z * 0.05f);  // Higher points slightly larger
            draw_list->AddCircleFilled(screen_pos, 1.5f * z_factor, IM_COL32(pt.r, pt.g, pt.b, 180));
            rendered++;
        }
    }

    // Draw current scan points (disabled by default for performance)
    if (g_config.operate_show_scan && !cached_scan_points.empty()) {
        for (const auto& pt : cached_scan_points) {
            ImVec2 screen_pos = worldToScreen3D(pt.x, pt.y, pt.z);
            if (screen_pos.x < canvas_pos.x - 5 || screen_pos.x > canvas_pos.x + canvas_size.x + 5 ||
                screen_pos.y < canvas_pos.y - 5 || screen_pos.y > canvas_pos.y + canvas_size.y + 5) continue;
            draw_list->AddCircleFilled(screen_pos, 2.5f, IM_COL32(200, 255, 255, 255));
        }
    }

    // Draw trajectory
    auto trajectory = g_shared.getTrajectory();
    if (trajectory.size() > 1) {
        for (size_t i = 1; i < trajectory.size(); i++) {
            ImVec2 p1 = worldToScreen(trajectory[i-1].x, trajectory[i-1].y);
            ImVec2 p2 = worldToScreen(trajectory[i].x, trajectory[i].y);
            draw_list->AddLine(p1, p2, IM_COL32(100, 200, 255, 150), 2.0f);
        }
    }

    // Draw robot (triangle showing heading)
    ImVec2 robot_pos = worldToScreen(pose.x, pose.y);
    float robot_size = 0.15f * view_scale;  // 15cm robot size
    float heading = pose.yaw;

    // Triangle vertices (pointing in heading direction)
    float cos_h = std::cos(heading);
    float sin_h = std::sin(heading);
    ImVec2 front = ImVec2(robot_pos.x + cos_h * robot_size, robot_pos.y - sin_h * robot_size);
    ImVec2 back_left = ImVec2(robot_pos.x - cos_h * robot_size * 0.6f + sin_h * robot_size * 0.4f,
                               robot_pos.y + sin_h * robot_size * 0.6f + cos_h * robot_size * 0.4f);
    ImVec2 back_right = ImVec2(robot_pos.x - cos_h * robot_size * 0.6f - sin_h * robot_size * 0.4f,
                                robot_pos.y + sin_h * robot_size * 0.6f - cos_h * robot_size * 0.4f);

    draw_list->AddTriangleFilled(front, back_left, back_right, IM_COL32(50, 200, 50, 255));
    draw_list->AddTriangle(front, back_left, back_right, IM_COL32(100, 255, 100, 255), 2.0f);

    draw_list->PopClipRect();

    // Handle mouse input for pan/zoom/rotate
    ImGui::InvisibleButton("MapCanvas", canvas_size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
    if (ImGui::IsItemHovered()) {
        // Zoom with scroll wheel
        float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f) {
            view_scale *= (wheel > 0) ? 1.2f : 0.833f;
            view_scale = std::clamp(view_scale, 5.0f, 500.0f);
        }

        // Pan with right mouse button
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Right) && !ImGui::GetIO().KeyCtrl) {
            ImVec2 delta = ImGui::GetIO().MouseDelta;
            view_offset_x += delta.x / view_scale;
            view_offset_y -= delta.y / view_scale;
            follow_robot = false;  // Disable follow when panning
        }

        // Rotate view with middle mouse button or Ctrl+Right drag
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
            (ImGui::IsMouseDragging(ImGuiMouseButton_Right) && ImGui::GetIO().KeyCtrl)) {
            ImVec2 delta = ImGui::GetIO().MouseDelta;
            // Horizontal drag = yaw rotation
            view_yaw += delta.x * 0.3f;
            // Vertical drag = pitch rotation (clamp to reasonable range)
            view_pitch -= delta.y * 0.3f;
            view_pitch = std::clamp(view_pitch, 15.0f, 90.0f);  // 15° to 90° (top-down)
            // Switch to FREE mode when manually rotating
            if (g_config.camera_mode != CameraMode::FREE) {
                g_config.camera_mode = CameraMode::FREE;
            }
        }

        // Double-click to re-center on robot
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            follow_robot = true;
        }

        // Reset view with 'R' key when hovered
        if (ImGui::IsKeyPressed(ImGuiKey_R)) {
            view_pitch = 90.0f;
            view_yaw = 0.0f;
            view_scale = 50.0f;
            follow_robot = true;
        }
    }

    // Status overlay
    ImGui::SetCursorPos(ImVec2(5, 5));
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Scale: %.1f px/m | Pitch: %.0f | Yaw: %.0f | Points: %zu",
                       view_scale, view_pitch, view_yaw, cached_map_points.size());
    ImGui::SetCursorPos(ImVec2(5, 20));
    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Robot: (%.2f, %.2f) %.1f deg",
                       pose.x, pose.y, pose.yaw * 180.0f / 3.14159f);
    ImGui::SetCursorPos(ImVec2(5, 35));
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Scroll=Zoom | RightDrag=Pan | MiddleDrag/Ctrl+Drag=Rotate | R=Reset");

    ImGui::EndChild();
    ImGui::EndChild();

    ImGui::SameLine();

    // Right side - Status panel
    ImGui::BeginChild("StatusPanel", ImVec2(panel_width, 0), true);

    // Pose display
    ImGui::Text("Fused Position");
    ImGui::Separator();
    Pose3D fused = g_shared.getFusedPose();
    ImGui::Text("X: %.3f m", fused.x);
    ImGui::Text("Y: %.3f m", fused.y);
    ImGui::Text("Heading: %.1f deg", fused.yaw * 180.0f / 3.14159f);

    ImGui::Spacing();
    ImGui::Separator();

    // Velocity display
    ImGui::Text("Velocity");
    ImGui::Separator();
    Velocity2D vel = g_shared.getVelocity();
    ImGui::Text("Linear:  %.2f m/s", vel.linear);
    ImGui::Text("Angular: %.2f rad/s", vel.angular);

    ImGui::Spacing();
    ImGui::Separator();

    // Motion state
    ImGui::Text("Motion State");
    ImGui::Separator();
    MotionState motion = g_shared.getMotionState();
    ImVec4 motion_color;
    switch (motion) {
        case MotionState::STATIONARY:
            motion_color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
            break;
        case MotionState::STRAIGHT_LINE:
            motion_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
            break;
        case MotionState::TURNING:
            motion_color = ImVec4(1.0f, 0.8f, 0.2f, 1.0f);
            break;
    }
    ImGui::TextColored(motion_color, "[%s]", getMotionStateName(motion));

    ImGui::Spacing();
    ImGui::Separator();

    // Motor status
    ImGui::Text("Motors");
    ImGui::Separator();
    MotorStatus ml = g_shared.getMotorStatus(0);
    MotorStatus mr = g_shared.getMotorStatus(1);

    ImGui::Columns(3, "motor_cols", false);
    ImGui::SetColumnWidth(0, 60);
    ImGui::SetColumnWidth(1, 110);
    ImGui::Text(""); ImGui::NextColumn();
    ImGui::Text("Left"); ImGui::NextColumn();
    ImGui::Text("Right"); ImGui::NextColumn();

    ImGui::Text("ERPM"); ImGui::NextColumn();
    ImGui::Text("%d", ml.erpm); ImGui::NextColumn();
    ImGui::Text("%d", mr.erpm); ImGui::NextColumn();

    ImGui::Text("Temp"); ImGui::NextColumn();
    ImGui::Text("%.0f C", ml.temp_motor); ImGui::NextColumn();
    ImGui::Text("%.0f C", mr.temp_motor); ImGui::NextColumn();

    ImGui::Columns(1);

    ImGui::Spacing();
    ImGui::Separator();

    // Manual control
    ImGui::Text("Manual Control");
    ImGui::Separator();

    // Start/Stop Operating button
    AppState state = g_shared.getAppState();
    bool is_operating = (state == AppState::OPERATING);

    if (!is_operating) {
        // Allow starting operating from IDLE, MAPPING, or LOCALIZED (auto-switch)
        bool can_operate = (state == AppState::IDLE || state == AppState::MAPPING ||
                           state == AppState::LOCALIZED);
        ImGui::BeginDisabled(!can_operate);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.8f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.9f, 1.0f));
        if (ImGui::Button("Start Operating", ImVec2(200, 40))) {
            // Auto-stop current mode if needed
            if (state == AppState::MAPPING) {
                g_commands.push(Command::simple(CommandType::STOP_MAPPING));
            } else if (state == AppState::LOCALIZED) {
                g_commands.push(Command::simple(CommandType::STOP_LOCALIZATION));
            }
            g_commands.push(Command::simple(CommandType::START_OPERATING));
        }
        SetTooltip("Enable manual drive without SLAM. Use for positioning or testing.");
        ImGui::PopStyleColor(2);
        ImGui::EndDisabled();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));
        if (ImGui::Button("Stop Operating", ImVec2(200, 40))) {
            g_commands.push(Command::simple(CommandType::STOP_OPERATING));
        }
        SetTooltip("Stop manual drive and return to idle.");
        ImGui::PopStyleColor(2);
    }

    ImGui::Spacing();

    // Speed limit slider
    ImGui::SliderFloat("Max Speed", &g_config.max_speed, 0.1f, 1.0f, "%.2f m/s");
    SetTooltip("Maximum speed when using gamepad control.");

    // Gamepad indicator
    if (g_gamepad_connected) {
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "Gamepad: Connected");
        ImGui::Text("L-Stick Y: %.2f (drive)", g_gamepad_left_y);
        ImGui::Text("L-Stick X: %.2f (steer)", g_gamepad_left_x);
        ImGui::Text("Velocity: %.2f m/s, %.2f rad/s",
            g_drive_cmd.linear_velocity, g_drive_cmd.angular_velocity);
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.2f, 1.0f), "Gamepad: Not connected");
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Use WASD or Arrow keys");
        SetTooltip("W/Up=Forward, S/Down=Back, A/Left=Turn Left, D/Right=Turn Right");
    }

    ImGui::EndChild();
    ImGui::EndChild();
}

//==============================================================================
// UI: Reusable Map Viewer Component
//==============================================================================

// Static state for map viewer (shared across tabs for consistency)
struct MapViewerState {
    float scale = 50.0f;        // pixels per meter
    float offset_x = 0.0f;      // World X offset
    float offset_y = 0.0f;      // World Y offset
    float pitch = 90.0f;        // Camera pitch (90=top-down, lower=tilted)
    float yaw = 0.0f;           // Camera yaw rotation
    bool follow_robot = true;
    std::vector<RenderPoint> cached_map_points;
    std::vector<RenderPoint> cached_scan_points;
};
static MapViewerState g_map_viewer;

/**
 * Draw map viewer component
 * @param show_robot Whether to show robot position marker
 * @param show_trajectory Whether to show robot trajectory
 * @param show_scan Whether to show current LiDAR scan
 */
void DrawMapViewer(bool show_robot = true, bool show_trajectory = true, bool show_scan = true) {
    // Get canvas info
    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImGui::BeginChild("MapViewer", avail, true, ImGuiWindowFlags_NoScrollbar);

    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    ImVec2 canvas_center = ImVec2(canvas_pos.x + canvas_size.x / 2,
                                   canvas_pos.y + canvas_size.y / 2);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Clip to canvas
    draw_list->PushClipRect(canvas_pos,
                            ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                            true);

    // Background
    draw_list->AddRectFilled(canvas_pos,
                             ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
                             IM_COL32(15, 15, 20, 255));

    // Get robot pose
    Pose3D pose = g_shared.getFusedPose();

    // Follow robot mode
    if (g_map_viewer.follow_robot) {
        g_map_viewer.offset_x = -pose.x;
        g_map_viewer.offset_y = -pose.y;
    }

    // 3D projection math
    float pitch_rad = g_map_viewer.pitch * 3.14159f / 180.0f;
    float yaw_rad = g_map_viewer.yaw * 3.14159f / 180.0f;
    float cos_pitch = std::cos(pitch_rad);
    float sin_pitch = std::sin(pitch_rad);
    float cos_yaw = std::cos(yaw_rad);
    float sin_yaw = std::sin(yaw_rad);

    auto worldToScreen3D = [&](float wx, float wy, float wz) -> ImVec2 {
        float tx = wx + g_map_viewer.offset_x;
        float ty = wy + g_map_viewer.offset_y;
        float rx = tx * cos_yaw - ty * sin_yaw;
        float ry = tx * sin_yaw + ty * cos_yaw;
        float rz = wz;
        float screen_x = rx;
        float screen_y = ry * sin_pitch + rz * cos_pitch;
        float sx = canvas_center.x + screen_x * g_map_viewer.scale;
        float sy = canvas_center.y - screen_y * g_map_viewer.scale;
        return ImVec2(sx, sy);
    };

    // Draw grid
    float grid_spacing = 1.0f;
    if (g_map_viewer.scale < 20.0f) grid_spacing = 5.0f;
    if (g_map_viewer.scale > 100.0f) grid_spacing = 0.5f;

    float view_width = canvas_size.x / g_map_viewer.scale;
    float view_height = canvas_size.y / g_map_viewer.scale;
    float min_x = -g_map_viewer.offset_x - view_width / 2;
    float max_x = -g_map_viewer.offset_x + view_width / 2;
    float min_y = -g_map_viewer.offset_y - view_height / 2;
    float max_y = -g_map_viewer.offset_y + view_height / 2;

    for (float x = std::floor(min_x / grid_spacing) * grid_spacing; x <= max_x; x += grid_spacing) {
        ImVec2 p1 = worldToScreen3D(x, min_y, 0.0f);
        ImVec2 p2 = worldToScreen3D(x, max_y, 0.0f);
        draw_list->AddLine(p1, p2, IM_COL32(40, 40, 50, 255), 1.0f);
    }
    for (float y = std::floor(min_y / grid_spacing) * grid_spacing; y <= max_y; y += grid_spacing) {
        ImVec2 p1 = worldToScreen3D(min_x, y, 0.0f);
        ImVec2 p2 = worldToScreen3D(max_x, y, 0.0f);
        draw_list->AddLine(p1, p2, IM_COL32(40, 40, 50, 255), 1.0f);
    }

    // Origin axes
    draw_list->AddLine(worldToScreen3D(-100, 0, 0), worldToScreen3D(100, 0, 0), IM_COL32(80, 40, 40, 255), 1.0f);
    draw_list->AddLine(worldToScreen3D(0, -100, 0), worldToScreen3D(0, 100, 0), IM_COL32(40, 80, 40, 255), 1.0f);
    if (g_map_viewer.pitch < 85.0f) {
        draw_list->AddLine(worldToScreen3D(0, 0, 0), worldToScreen3D(0, 0, 3), IM_COL32(40, 40, 100, 255), 1.0f);
    }

    // Update cached points
    g_shared.getMapPointsIfUpdated(g_map_viewer.cached_map_points);
    g_shared.getCurrentScanIfUpdated(g_map_viewer.cached_scan_points);

    // Draw map points
    for (const auto& pt : g_map_viewer.cached_map_points) {
        ImVec2 screen_pos = worldToScreen3D(pt.x, pt.y, pt.z);
        if (screen_pos.x < canvas_pos.x - 5 || screen_pos.x > canvas_pos.x + canvas_size.x + 5 ||
            screen_pos.y < canvas_pos.y - 5 || screen_pos.y > canvas_pos.y + canvas_size.y + 5) continue;
        float z_factor = 1.0f + (pt.z * 0.05f);
        draw_list->AddCircleFilled(screen_pos, 1.5f * z_factor, IM_COL32(pt.r, pt.g, pt.b, 180));
    }

    // Draw current scan
    if (show_scan) {
        for (const auto& pt : g_map_viewer.cached_scan_points) {
            ImVec2 screen_pos = worldToScreen3D(pt.x, pt.y, pt.z);
            if (screen_pos.x < canvas_pos.x - 5 || screen_pos.x > canvas_pos.x + canvas_size.x + 5 ||
                screen_pos.y < canvas_pos.y - 5 || screen_pos.y > canvas_pos.y + canvas_size.y + 5) continue;
            draw_list->AddCircleFilled(screen_pos, 2.5f, IM_COL32(200, 255, 255, 255));
        }
    }

    // Draw trajectory
    if (show_trajectory) {
        auto trajectory = g_shared.getTrajectory();
        if (trajectory.size() > 1) {
            for (size_t i = 1; i < trajectory.size(); i++) {
                ImVec2 p1 = worldToScreen3D(trajectory[i-1].x, trajectory[i-1].y, 0);
                ImVec2 p2 = worldToScreen3D(trajectory[i].x, trajectory[i].y, 0);
                draw_list->AddLine(p1, p2, IM_COL32(100, 200, 255, 150), 2.0f);
            }
        }
    }

    // Draw robot
    if (show_robot) {
        ImVec2 robot_pos = worldToScreen3D(pose.x, pose.y, 0);
        float robot_size = 0.15f * g_map_viewer.scale;
        float heading = pose.yaw;
        float cos_h = std::cos(heading);
        float sin_h = std::sin(heading);
        ImVec2 front = ImVec2(robot_pos.x + cos_h * robot_size, robot_pos.y - sin_h * robot_size);
        ImVec2 back_left = ImVec2(robot_pos.x - cos_h * robot_size * 0.6f + sin_h * robot_size * 0.4f,
                                   robot_pos.y + sin_h * robot_size * 0.6f + cos_h * robot_size * 0.4f);
        ImVec2 back_right = ImVec2(robot_pos.x - cos_h * robot_size * 0.6f - sin_h * robot_size * 0.4f,
                                    robot_pos.y + sin_h * robot_size * 0.6f - cos_h * robot_size * 0.4f);
        draw_list->AddTriangleFilled(front, back_left, back_right, IM_COL32(50, 200, 50, 255));
        draw_list->AddTriangle(front, back_left, back_right, IM_COL32(100, 255, 100, 255), 2.0f);
    }

    draw_list->PopClipRect();

    // Handle mouse input
    ImGui::InvisibleButton("MapCanvas", canvas_size,
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
    if (ImGui::IsItemHovered()) {
        float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f) {
            g_map_viewer.scale *= (wheel > 0) ? 1.2f : 0.833f;
            g_map_viewer.scale = std::clamp(g_map_viewer.scale, 5.0f, 500.0f);
        }
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Right) && !ImGui::GetIO().KeyCtrl) {
            ImVec2 delta = ImGui::GetIO().MouseDelta;
            g_map_viewer.offset_x += delta.x / g_map_viewer.scale;
            g_map_viewer.offset_y -= delta.y / g_map_viewer.scale;
            g_map_viewer.follow_robot = false;
        }
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
            (ImGui::IsMouseDragging(ImGuiMouseButton_Right) && ImGui::GetIO().KeyCtrl)) {
            ImVec2 delta = ImGui::GetIO().MouseDelta;
            g_map_viewer.yaw += delta.x * 0.3f;
            g_map_viewer.pitch -= delta.y * 0.3f;
            g_map_viewer.pitch = std::clamp(g_map_viewer.pitch, 15.0f, 90.0f);
        }
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            g_map_viewer.follow_robot = true;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_R)) {
            g_map_viewer.pitch = 90.0f;
            g_map_viewer.yaw = 0.0f;
            g_map_viewer.scale = 50.0f;
            g_map_viewer.follow_robot = true;
        }
    }

    // Status overlay
    ImGui::SetCursorPos(ImVec2(5, 5));
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Scale: %.0f | Pitch: %.0f | Points: %zu",
                       g_map_viewer.scale, g_map_viewer.pitch, g_map_viewer.cached_map_points.size());
    ImGui::SetCursorPos(ImVec2(5, 20));
    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Pos: (%.2f, %.2f) Hdg: %.1f",
                       pose.x, pose.y, pose.yaw * 180.0f / 3.14159f);
    ImGui::SetCursorPos(ImVec2(5, 35));
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Scroll=Zoom | RDrag=Pan | MDrag=Rotate | R=Reset");

    ImGui::EndChild();
}

//==============================================================================
// UI: Tab - Mapping
//==============================================================================
void DrawMappingTab() {
    AppState state = g_shared.getAppState();
    bool is_mapping = (state == AppState::MAPPING);

    ImGui::BeginChild("MappingContent", ImVec2(0, 0), false);

    // Split into left (viewer) and right (controls) panels
    float panel_width = 280;

    // Left side - Map viewer (GPU-accelerated when available)
    ImGui::BeginChild("MappingViewer", ImVec2(-panel_width - 10, 0), true);

    // Header with point count
    size_t point_count = g_shared.getMapPointCount();
    if (point_count > 1000000) {
        ImGui::Text("Live Map View (%.2fM points)", point_count / 1000000.0f);
    } else if (point_count > 1000) {
        ImGui::Text("Live Map View (%.1fK points)", point_count / 1000.0f);
    } else {
        ImGui::Text("Live Map View (%zu points)", point_count);
    }
    ImGui::Separator();

    if (g_viewers_initialized && g_mapping_viewer) {
        // Update robot pose in viewer
        Pose3D robot_pose = g_shared.getFusedPose();
        g_mapping_viewer->setRobotPose(robot_pose.x, robot_pose.y, robot_pose.yaw);

        // Use GPU-accelerated viewer
        ImVec2 avail = ImGui::GetContentRegionAvail();
        g_mapping_viewer->renderWidget(avail.x, avail.y);
    } else {
        // Fallback to CPU-based viewer
        DrawMapViewer(true, true, true);
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // Right side - Controls panel
    ImGui::BeginChild("MappingControls", ImVec2(panel_width, 0), true);

    ImGui::Text("Map Building");
    ImGui::Separator();
    ImGui::Spacing();

    // Calibration reminder
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Tip: Calibrate first");
    ImGui::Spacing();

    // Start/Stop mapping
    if (!is_mapping) {
        // Allow starting mapping from IDLE or OPERATING (auto-switch)
        bool can_start = (state == AppState::IDLE || state == AppState::OPERATING);
        ImGui::BeginDisabled(!can_start);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
        if (ImGui::Button("Start Mapping", ImVec2(-1, 50))) {
            // Auto-stop operating if currently in that mode
            if (state == AppState::OPERATING) {
                g_commands.push(Command::simple(CommandType::STOP_OPERATING));
            }
            g_commands.push(Command::simple(CommandType::START_MAPPING));
        }
        SetTooltip("Begin building a new map.");
        ImGui::PopStyleColor(2);
        ImGui::EndDisabled();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));
        if (ImGui::Button("Stop Mapping", ImVec2(-1, 50))) {
            g_commands.push(Command::simple(CommandType::STOP_MAPPING));
        }
        SetTooltip("Stop map building.");
        ImGui::PopStyleColor(2);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Map statistics
    ImGui::Text("Map Statistics");
    size_t map_points = g_shared.getMapPointCount();
    ImGui::Text("Points: %zu", map_points);
    ImGui::Text("Trajectory: %zu pts", g_shared.getTrajectory().size());

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Viewer disabled warning
    if (g_config.disable_map_viewer) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "[VIEWER DISABLED]");
        SetTooltip("Map viewer updates disabled for performance testing. Enable in Settings > 3D Viewer.");
    }

    // Performance metrics section
    ImGui::Text("Performance Metrics");

    // SLAM rate with color coding (green=good, yellow=slow, red=critical)
    float slam_rate = g_perf.slam_rate_hz.load();
    float slam_time = g_perf.slam_time_ms.load();
    float avg_slam_time = g_perf.getAvgSlamTime();

    ImVec4 rate_color;
    if (slam_rate >= 9.0f) {
        rate_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);  // Green - good
    } else if (slam_rate >= 5.0f) {
        rate_color = ImVec4(0.9f, 0.7f, 0.1f, 1.0f);  // Yellow - slow
    } else {
        rate_color = ImVec4(0.9f, 0.2f, 0.2f, 1.0f);  // Red - critical
    }
    ImGui::TextColored(rate_color, "SLAM Rate: %.1f Hz", slam_rate);
    SetTooltip("Target: 10 Hz (LiDAR scan rate). Below 5 Hz = falling behind.");

    // SLAM processing time with color coding
    ImVec4 time_color;
    if (avg_slam_time < 50.0f) {
        time_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);  // Green - plenty of headroom
    } else if (avg_slam_time < 80.0f) {
        time_color = ImVec4(0.9f, 0.7f, 0.1f, 1.0f);  // Yellow - getting tight
    } else {
        time_color = ImVec4(0.9f, 0.2f, 0.2f, 1.0f);  // Red - over budget
    }
    ImGui::TextColored(time_color, "SLAM Time: %.1f ms (avg %.1f)", slam_time, avg_slam_time);
    SetTooltip("Time per SLAM update. Budget: <100ms (10Hz). If >100ms, falling behind.");

    // Buffer status - critical indicator for falling behind
    int imu_buffer = g_perf.buffer_imu.load();
    int scan_buffer = g_perf.buffer_scans.load();

    ImVec4 buffer_color;
    if (scan_buffer > 3 || imu_buffer > 100) {
        buffer_color = ImVec4(0.9f, 0.2f, 0.2f, 1.0f);  // Red - backed up!
    } else if (scan_buffer > 1 || imu_buffer > 50) {
        buffer_color = ImVec4(0.9f, 0.7f, 0.1f, 1.0f);  // Yellow - getting behind
    } else {
        buffer_color = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);  // Gray - normal
    }
    ImGui::TextColored(buffer_color, "Buffers: %d scans, %d IMU", scan_buffer, imu_buffer);
    SetTooltip("Waiting data. Scans>2 or IMU>50 means CPU can't keep up.");

    // Map point count from performance tracker with limit indicator
    int map_pts = g_perf.map_points.load();
    int map_limit = g_config.max_map_points;
    ImVec4 map_color;
    if (map_limit > 0 && map_pts >= map_limit) {
        map_color = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);  // Red - at limit
    } else if (map_limit > 0 && map_pts >= map_limit * 0.8f) {
        map_color = ImVec4(0.9f, 0.7f, 0.1f, 1.0f);  // Yellow - approaching limit
    } else {
        map_color = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);  // Gray - normal
    }
    if (map_limit > 0) {
        ImGui::TextColored(map_color, "Map: %dK / %dK pts", map_pts / 1000, map_limit / 1000);
        SetTooltip("Map size vs limit. Red = at limit, oldest points will be culled.");
    } else {
        ImGui::Text("Map Size: %dK pts (no limit)", map_pts / 1000);
    }

    // LiDAR input rates
    LidarStatus lidar = g_shared.getLidarStatus();
    if (lidar.point_rate > 0) {
        float pts_per_sec_k = lidar.point_rate / 1000.0f;
        ImGui::Text("LiDAR: %.0fK pts/s, %d Hz IMU", pts_per_sec_k, lidar.imu_rate);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Current pose
    Pose3D pose = g_shared.getFusedPose();
    ImGui::Spacing();
    ImGui::Text("Robot Position");
    ImGui::Text("X: %.2f m", pose.x);
    ImGui::Text("Y: %.2f m", pose.y);
    ImGui::Text("Heading: %.1f deg", pose.yaw * 180.0f / 3.14159f);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Save map
    ImGui::Text("Save Map");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##Filename", g_map_filename, sizeof(g_map_filename));
    SetTooltip("Name for the saved map file.");

    bool can_save = (state == AppState::IDLE) && map_points > 0;
    ImGui::BeginDisabled(!can_save);
    if (ImGui::Button("Save Map", ImVec2(-1, 30))) {
        std::string full_path = std::string(g_config.map_directory) + "/" + g_map_filename;
        g_commands.push(Command::file(CommandType::SAVE_MAP, full_path));
    }
    SetTooltip("Save the current map to a PLY file.");
    ImGui::EndDisabled();

    if (ImGui::Button("Clear Map", ImVec2(-1, 30))) {
        g_commands.push(Command::simple(CommandType::CLEAR_MAP));
        g_shared.setStatusMessage("Map cleared");
    }
    SetTooltip("Clear the current map and start fresh.");

    ImGui::EndChild();
    ImGui::EndChild();
}

//==============================================================================
// UI: Tab - Localization
//==============================================================================
void DrawLocalizationTab() {
    AppState state = g_shared.getAppState();

    ImGui::BeginChild("LocalizationContent", ImVec2(0, 0), false);

    // Split into left (viewer) and right (controls) panels
    float panel_width = 280;

    // Left side - Map viewer (GPU-accelerated with dual-map display)
    ImGui::BeginChild("LocalizationViewer", ImVec2(-panel_width - 10, 0), true);

    // Header with point count
    size_t local_points = g_shared.getMapPointCount();
    if (local_points > 1000000) {
        ImGui::Text("Map View (%.2fM local points)", local_points / 1000000.0f);
    } else if (local_points > 1000) {
        ImGui::Text("Map View (%.1fK local points)", local_points / 1000.0f);
    } else {
        ImGui::Text("Map View (%zu local points)", local_points);
    }
    ImGui::Separator();

    if (g_viewers_initialized && g_localization_viewer) {
        // Update robot pose in viewer
        Pose3D robot_pose = g_shared.getFusedPose();
        g_localization_viewer->setRobotPose(robot_pose.x, robot_pose.y, robot_pose.yaw);

        // Use GPU-accelerated viewer
        ImVec2 avail = ImGui::GetContentRegionAvail();
        g_localization_viewer->renderWidget(avail.x, avail.y);
    } else {
        // Fallback to CPU-based viewer
        DrawMapViewer(true, true, true);
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // Right side - Controls panel
    ImGui::BeginChild("LocalizationControls", ImVec2(panel_width, 0), true);

    ImGui::Text("Localization");
    ImGui::Separator();
    ImGui::Spacing();

    // Calibration reminder
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Tip: Calibrate first");
    ImGui::Spacing();

    // Map selection
    ImGui::Text("Select Map");
    if (ImGui::Button("Refresh", ImVec2(70, 0))) {
        ScanForMaps();
    }
    SetTooltip("Scan directory for PLY maps.");
    ImGui::SameLine();
    ImGui::Text("(%zu found)", g_available_maps.size());

    static int selected_map = -1;
    if (ImGui::BeginListBox("##Maps", ImVec2(-1, 80))) {
        for (int i = 0; i < (int)g_available_maps.size(); i++) {
            bool is_selected = (selected_map == i);
            if (ImGui::Selectable(g_available_maps[i].c_str(), is_selected)) {
                selected_map = i;
            }
        }
        ImGui::EndListBox();
    }

    // Load map button
    bool can_load = (selected_map >= 0) && (state == AppState::IDLE);
    ImGui::BeginDisabled(!can_load);
    if (ImGui::Button("Load Map", ImVec2(-1, 30))) {
        std::string full_path = std::string(g_config.map_directory) + "/" + g_available_maps[selected_map];
        g_commands.push(Command::file(CommandType::LOAD_MAP, full_path));
    }
    SetTooltip("Load the selected map.");
    ImGui::EndDisabled();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Relocalization
    ImGui::Text("Global Relocalization");

    // Initial pose hint (optional)
    static float hint_x = 0.0f, hint_y = 0.0f, hint_yaw = 0.0f;
    static bool use_pose_hint = false;
    static bool click_to_pose_active = false;

    ImGui::Checkbox("Use pose hint", &use_pose_hint);
    ImGui::SameLine();
    HelpMarker("Provide approximate position to help.");

    if (use_pose_hint) {
        // Visual pose hint button - click on map
        if (g_viewers_initialized && g_localization_viewer) {
            bool was_active = click_to_pose_active;
            if (click_to_pose_active) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.3f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.8f, 0.4f, 1.0f));
            }
            if (ImGui::Button(click_to_pose_active ? "Click on Map..." : "Click on Map", ImVec2(-1, 25))) {
                click_to_pose_active = !click_to_pose_active;
            }
            if (click_to_pose_active) {
                ImGui::PopStyleColor(2);
            }
            SetTooltip("Click on map to set position. Drag to set heading direction.");

            // Set up/remove callback when mode changes
            if (click_to_pose_active != was_active) {
                if (click_to_pose_active) {
                    g_localization_viewer->setMapClickCallback([](float x, float y, float heading) {
                        hint_x = x;
                        hint_y = y;
                        if (!std::isnan(heading)) {
                            hint_yaw = heading * 180.0f / 3.14159f;  // Convert to degrees
                        }
                        // Don't auto-disable - user can click again to refine
                    });
                    g_localization_viewer->setClickToPoseMode(true);
                } else {
                    g_localization_viewer->setMapClickCallback(nullptr);
                    g_localization_viewer->setClickToPoseMode(false);
                }
            }

            if (click_to_pose_active) {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Click map to set position");
            }
        }

        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##HintX", &hint_x, 0.1f, -100.0f, 100.0f, "X: %.1f m");
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##HintY", &hint_y, 0.1f, -100.0f, 100.0f, "Y: %.1f m");
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##HintYaw", &hint_yaw, 1.0f, -180.0f, 180.0f, "Yaw: %.0f deg");
    } else if (click_to_pose_active) {
        // Disable click-to-pose if hint was unchecked
        click_to_pose_active = false;
        if (g_localization_viewer) {
            g_localization_viewer->setMapClickCallback(nullptr);
            g_localization_viewer->setClickToPoseMode(false);
        }
    }

    // Allow relocalize from MAP_LOADED, RELOCALIZE_FAILED, OPERATING, or MAPPING (auto-switch)
    bool can_relocalize = (state == AppState::MAP_LOADED || state == AppState::RELOCALIZE_FAILED ||
                          state == AppState::OPERATING || state == AppState::MAPPING);
    ImGui::BeginDisabled(!can_relocalize);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.7f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 0.8f, 1.0f));
    if (ImGui::Button("Relocalize", ImVec2(-1, 40))) {
        // Auto-stop current mode if needed
        if (state == AppState::OPERATING) {
            g_commands.push(Command::simple(CommandType::STOP_OPERATING));
        } else if (state == AppState::MAPPING) {
            g_commands.push(Command::simple(CommandType::STOP_MAPPING));
        }
        if (use_pose_hint) {
            g_commands.push(Command::poseHint(hint_x, hint_y, hint_yaw));
        }
        g_commands.push(Command::simple(CommandType::RELOCALIZE));
    }
    SetTooltip("Find robot position using global search.");
    ImGui::PopStyleColor(2);
    ImGui::EndDisabled();

    // Relocalization status
    RelocalizationProgress reloc = g_shared.getRelocalizationProgress();
    if (reloc.running) {
        // Show progress bar
        ImGui::ProgressBar(reloc.progress, ImVec2(-1, 20));
        ImGui::Text("%s", reloc.status_text.c_str());

        // Show detailed coverage metrics while accumulating
        if (reloc.accumulating) {
            ImGui::Spacing();
            ImGui::Text("Coverage Metrics:");

            // Voxel progress bar
            float voxel_ratio = static_cast<float>(reloc.local_map_voxels) / reloc.GOOD_VOXELS;
            ImVec4 voxel_color = voxel_ratio >= 1.0f ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) :
                                 voxel_ratio >= 0.5f ? ImVec4(0.8f, 0.8f, 0.2f, 1.0f) :
                                                       ImVec4(0.8f, 0.4f, 0.2f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, voxel_color);
            ImGui::ProgressBar(std::min(1.0f, voxel_ratio), ImVec2(-1, 15),
                ("Voxels: " + std::to_string(reloc.local_map_voxels) + "/" +
                 std::to_string(reloc.GOOD_VOXELS)).c_str());
            ImGui::PopStyleColor();

            // Rotation progress bar
            float rot_ratio = reloc.rotation_deg / reloc.GOOD_ROTATION;
            ImVec4 rot_color = rot_ratio >= 1.0f ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) :
                               rot_ratio >= 0.5f ? ImVec4(0.8f, 0.8f, 0.2f, 1.0f) :
                                                   ImVec4(0.8f, 0.4f, 0.2f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, rot_color);
            char rot_buf[64];
            snprintf(rot_buf, sizeof(rot_buf), "Rotation: %.0f/%.0f deg",
                     reloc.rotation_deg, reloc.GOOD_ROTATION);
            ImGui::ProgressBar(std::min(1.0f, rot_ratio), ImVec2(-1, 15), rot_buf);
            ImGui::PopStyleColor();

            // Coverage quality
            ImGui::Text("Quality: %s", reloc.getCoverageQuality());

            // Guidance
            if (!reloc.isReadyToAttempt()) {
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                    "Slowly rotate robot to build geometry");
            }
        }

        // Show "Localize Now" button when ready
        if (reloc.ready_to_localize && !reloc.icp_running) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f),
                "Coverage sufficient! Stop robot before continuing.");
            ImGui::Spacing();

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.8f, 0.4f, 1.0f));
            if (ImGui::Button("Localize Now", ImVec2(-1, 40))) {
                g_commands.push(Command::simple(CommandType::RUN_GLOBAL_LOCALIZATION));
            }
            ImGui::PopStyleColor(2);
            SetTooltip("Run global ICP alignment. Robot should be stationary.");
        }

        // Show ICP progress when running
        if (reloc.icp_running) {
            ImGui::Spacing();
            ImGui::Text("Stage: %s", reloc.icp_stage.c_str());
            ImGui::ProgressBar(reloc.icp_progress, ImVec2(-1, 20));

            if (reloc.icp_hypotheses > 0) {
                ImGui::Text("Hypotheses: %d kept of %d", reloc.icp_kept, reloc.icp_hypotheses);
            }
            if (reloc.icp_max_iterations > 0) {
                ImGui::Text("Iteration: %d/%d", reloc.icp_iteration, reloc.icp_max_iterations);
            }
            if (reloc.icp_fitness > 0) {
                ImGui::Text("Fitness: %.1f%%", reloc.icp_fitness * 100.0f);
            }

            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.3f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.4f, 0.3f, 1.0f));
            if (ImGui::Button("Cancel", ImVec2(-1, 30))) {
                g_commands.push(Command::simple(CommandType::CANCEL_GLOBAL_LOCALIZATION));
            }
            ImGui::PopStyleColor(2);
        }

        // Show attempt info if attempting
        if (!reloc.accumulating && !reloc.icp_running && reloc.attempt_number > 0) {
            ImGui::Text("Attempt: %d", reloc.attempt_number);
        }
    } else if (state == AppState::LOCALIZED) {
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f),
            "LOCALIZED (%.0f%%)", reloc.confidence * 100.0f);

        // Show current pose
        Pose3D pose = g_shared.getFusedPose();
        ImGui::Text("Position: (%.2f, %.2f)", pose.x, pose.y);
        ImGui::Text("Heading: %.1f deg", pose.yaw * 180.0f / 3.14159f);
    } else if (state == AppState::RELOCALIZE_FAILED) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.2f, 1.0f),
            "FAILED after %d attempts", reloc.attempt_number);
        ImGui::Text("Try different position or hint");
    } else if (state == AppState::MAP_LOADED) {
        ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f),
            "Map loaded - ready");
        ImGui::Text("Click Relocalize to start");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Restart button (appears when localized or failed)
    bool can_restart = (state == AppState::LOCALIZED || state == AppState::RELOCALIZE_FAILED);
    if (can_restart) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.4f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.5f, 0.3f, 1.0f));
        if (ImGui::Button("Re-localize", ImVec2(-1, 30))) {
            // Stop current localization and restart
            g_commands.push(Command::simple(CommandType::STOP_LOCALIZATION));
            g_commands.push(Command::simple(CommandType::RELOCALIZE));
        }
        SetTooltip("Stop current localization and start fresh from the beginning.");
        ImGui::PopStyleColor(2);
    }

    // Stop/Cancel button
    bool can_stop = (state == AppState::LOCALIZED || state == AppState::RELOCALIZING);
    ImGui::BeginDisabled(!can_stop);
    const char* stop_text = (state == AppState::RELOCALIZING) ? "Cancel" : "Stop Localization";
    ImVec4 stop_color = (state == AppState::RELOCALIZING) ?
        ImVec4(0.8f, 0.5f, 0.2f, 1.0f) : ImVec4(0.7f, 0.2f, 0.2f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, stop_color);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(stop_color.x + 0.1f, stop_color.y + 0.1f, stop_color.z + 0.1f, 1.0f));
    if (ImGui::Button(stop_text, ImVec2(-1, 30))) {
        g_commands.push(Command::simple(CommandType::STOP_LOCALIZATION));
    }
    SetTooltip(state == AppState::RELOCALIZING ?
        "Cancel localization process" : "Stop localization and return to idle.");
    ImGui::PopStyleColor(2);
    ImGui::EndDisabled();

    ImGui::EndChild();
    ImGui::EndChild();
}

//==============================================================================
// UI: Tab - Calibration
//==============================================================================
void DrawCalibrationTab() {
    AppState state = g_shared.getAppState();
    CalibrationProgress cal = g_shared.getCalibrationProgress();

    ImGui::Text("Motor Calibration");
    ImGui::Separator();
    ImGui::Spacing();

    // Current calibration file
    ImGui::Text("Calibration File:");
    ImGui::InputText("##CalFile", g_config.calibration_file, sizeof(g_config.calibration_file));
    SetTooltip("Path to the calibration INI file.");

    ImGui::SameLine();
    if (ImGui::Button("Load")) {
        g_commands.push(Command::file(CommandType::LOAD_CALIBRATION, g_config.calibration_file));
    }
    SetTooltip("Load calibration from file.");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Run calibration
    bool can_calibrate = (state == AppState::IDLE);
    ImGui::BeginDisabled(!can_calibrate);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.4f, 0.1f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.7f, 0.5f, 0.2f, 1.0f));
    if (ImGui::Button("Run Calibration", ImVec2(200, 50))) {
        g_commands.push(Command::simple(CommandType::RUN_CALIBRATION));
    }
    SetTooltip("Run the 4-phase calibration sequence:\n"
               "1. Forward duty sweep\n"
               "2. Reverse duty sweep\n"
               "3. Rotation calibration\n"
               "4. Turning min duty calibration\n\n"
               "Robot will move! Ensure clear space.");
    ImGui::PopStyleColor(2);
    ImGui::EndDisabled();

    // Calibration progress
    if (cal.running) {
        ImGui::Spacing();
        ImGui::Text("Phase: %s", getCalibrationPhaseName(cal.phase));
        ImGui::ProgressBar(cal.progress, ImVec2(-1, 25));
        ImGui::Text("%s", cal.status_text.c_str());

        ImGui::Spacing();
        if (ImGui::Button("Cancel", ImVec2(100, 30))) {
            // Set flag directly for immediate response (bypasses command queue)
            g_calibration_cancel.store(true);
            // Also push command to handle state transition after calibration stops
            g_commands.push(Command::simple(CommandType::CANCEL_CALIBRATION));
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Calibration parameters display
    if (ImGui::CollapsingHeader("Calibration Values", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Min Duty Thresholds (with %.0f%% margin):", g_config.safety_margin * 100);

        ImGui::Columns(5, "min_duty_table", true);
        ImGui::SetColumnWidth(0, 100);
        ImGui::Text("Mode"); ImGui::NextColumn();
        ImGui::Text("Start L"); ImGui::NextColumn();
        ImGui::Text("Start R"); ImGui::NextColumn();
        ImGui::Text("Keep L"); ImGui::NextColumn();
        ImGui::Text("Keep R"); ImGui::NextColumn();
        ImGui::Separator();

        // These would be populated from actual calibration data
        ImGui::Text("Forward"); ImGui::NextColumn();
        ImGui::Text("0.018"); ImGui::NextColumn();
        ImGui::Text("0.002"); ImGui::NextColumn();
        ImGui::Text("0.023"); ImGui::NextColumn();
        ImGui::Text("0.017"); ImGui::NextColumn();

        ImGui::Text("Reverse"); ImGui::NextColumn();
        ImGui::Text("0.004"); ImGui::NextColumn();
        ImGui::Text("0.010"); ImGui::NextColumn();
        ImGui::Text("0.017"); ImGui::NextColumn();
        ImGui::Text("0.023"); ImGui::NextColumn();

        ImGui::Text("Turn Fwd"); ImGui::NextColumn();
        ImGui::Text("0.016"); ImGui::NextColumn();
        ImGui::Text("0.006"); ImGui::NextColumn();
        ImGui::Text("0.027"); ImGui::NextColumn();
        ImGui::Text("0.021"); ImGui::NextColumn();

        ImGui::Text("Turn Rev"); ImGui::NextColumn();
        ImGui::Text("0.020"); ImGui::NextColumn();
        ImGui::Text("0.002"); ImGui::NextColumn();
        ImGui::Text("0.024"); ImGui::NextColumn();
        ImGui::Text("0.027"); ImGui::NextColumn();

        ImGui::Columns(1);
    }

    // Safety margin
    ImGui::Spacing();
    ImGui::SliderFloat("Safety Margin", &g_config.safety_margin, 0.0f, 0.25f, "%.0f%%");
    SetTooltip("Additional margin applied to all min duty thresholds to prevent stuttering.");
}

//==============================================================================
// UI: Tab - Hull & Probe
//==============================================================================
void DrawHullTab() {
    ImGui::Text("Hull Mesh & PAUT Probe Coverage");
    ImGui::Separator();
    ImGui::Spacing();

    // Hull mesh loading
    ImGui::Text("Hull Mesh");
    ImGui::InputText("Mesh File", g_hull_filename, sizeof(g_hull_filename));
    SetTooltip("Path to hull mesh file (OBJ or STL format).");

    ImGui::SameLine();
    if (ImGui::Button("Browse...")) {
        // TODO: Open file dialog
    }

    if (ImGui::Button("Load Hull Mesh", ImVec2(150, 30))) {
        if (strlen(g_hull_filename) > 0) {
            // Send command to worker thread for thread-safe loading
            g_commands.push(Command::file(CommandType::LOAD_HULL_MESH, g_hull_filename));
        }
    }
    SetTooltip("Load a 3D mesh representing the hull surface for coverage tracking.");

    // Hull info
    HullMeshInfo info = g_shared.getHullMeshInfo();
    if (info.loaded) {
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "Mesh loaded: %s", info.filename.c_str());
        ImGui::Text("Vertices: %d  Triangles: %d", info.vertex_count, info.triangle_count);
        ImGui::Text("Total area: %.2f m²", info.total_area_m2);
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No hull mesh loaded");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // PAUT Probe configuration
    ImGui::Text("PAUT Probe Configuration");
    ImGui::Spacing();

    ImGui::Checkbox("Enable Coverage Tracking", &g_config.probe.enabled);
    SetTooltip("Track which areas of the hull have been inspected by the probe.");

    ImGui::Spacing();

    // Probe offsets
    ImGui::Text("Probe Position (relative to robot center):");
    ImGui::DragFloat("X Offset (m)", &g_config.probe.offset_x, 0.01f, -1.0f, 1.0f, "%.3f");
    SetTooltip("Distance behind robot center. Negative = behind.");
    ImGui::SameLine();
    HelpMarker("The probe is typically mounted behind the robot.");

    ImGui::DragFloat("Y Offset (m)", &g_config.probe.offset_y, 0.01f, -0.5f, 0.5f, "%.3f");
    SetTooltip("Lateral offset from center. Positive = left.");

    ImGui::Spacing();

    // Probe dimensions
    ImGui::Text("Probe Dimensions:");
    ImGui::DragFloat("Width (m)", &g_config.probe.width, 0.005f, 0.01f, 0.5f, "%.3f");
    SetTooltip("Width of the probe inspection swath.");

    ImGui::DragFloat("Length (m)", &g_config.probe.length, 0.005f, 0.01f, 0.2f, "%.3f");
    SetTooltip("Length of the probe in travel direction.");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Coverage statistics
    ImGui::Text("Coverage Statistics");
    if (info.loaded) {
        // Progress bar for coverage
        ImGui::ProgressBar(info.coverage_percent / 100.0f, ImVec2(-1, 25),
            std::to_string((int)info.coverage_percent).append("%").c_str());

        ImGui::Text("Covered: %.2f m² / %.2f m²", info.covered_area_m2, info.total_area_m2);

        ImGui::Spacing();

        if (ImGui::Button("Clear Coverage", ImVec2(120, 30))) {
            g_hull.clearCoverage();
            g_shared.setStatusMessage("Coverage cleared");
        }
        SetTooltip("Reset coverage tracking to start fresh.");

        ImGui::SameLine();
        if (ImGui::Button("Export Coverage", ImVec2(120, 30))) {
            g_hull.exportCoverage("coverage_report.csv");
            g_shared.setStatusMessage("Coverage exported to coverage_report.csv");
        }
        SetTooltip("Export coverage data to a CSV file.");
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Load a hull mesh to track coverage");
    }
}

//==============================================================================
// UI: Tab - Diagnostics
//==============================================================================
void DrawDiagnosticsTab() {
    ImGui::Text("Diagnostics & Recording");
    ImGui::Separator();
    ImGui::Spacing();

#if ENABLE_HARDWARE
    // LiDAR / SLAM data flow
    if (ImGui::CollapsingHeader("Data Flow (Debug)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Point Frames: %llu", g_point_frame_count.load());
        ImGui::Text("IMU Samples:  %llu", g_imu_count.load());
        ImGui::Text("Scans Built:  %llu", g_scan_count.load());
        ImGui::Text("SLAM Processed: %llu", g_slam_process_count.load());

        LidarStatus lidar = g_shared.getLidarStatus();
        ImGui::Text("IMU Initialized: %s", lidar.imu_initialized ? "Yes" : "No");
        ImGui::Text("Gravity Mag: %.2f m/s^2", lidar.gravity_mag);

        if (ImGui::Button("Reset Counters")) {
            g_point_frame_count = 0;
            g_imu_count = 0;
            g_scan_count = 0;
            g_slam_process_count = 0;
        }
    }

    ImGui::Spacing();

    // IMU CSV Recording for vibration analysis
    if (ImGui::CollapsingHeader("IMU Recording", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool is_recording = g_imu_recording.load();

        if (is_recording) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "RECORDING: %zu samples", g_imu_record_count);
        } else {
            ImGui::Text("Record raw IMU data for vibration analysis");
        }

        ImGui::Spacing();

        if (!is_recording) {
            if (ImGui::Button("Start Recording", ImVec2(150, 30))) {
                std::lock_guard<std::mutex> lock(g_imu_csv_mutex);
                // Generate filename with timestamp
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                std::tm tm_buf;
                localtime_s(&tm_buf, &time_t);
                char filename[64];
                std::strftime(filename, sizeof(filename), "imu_data_%Y%m%d_%H%M%S.csv", &tm_buf);

                g_imu_csv_file.open(filename);
                if (g_imu_csv_file.is_open()) {
                    // Write header
                    g_imu_csv_file << "time_s,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z\n";
                    g_imu_record_start_ns = 0;
                    g_imu_record_count = 0;
                    g_imu_recording.store(true);
                }
            }
        } else {
            if (ImGui::Button("Stop Recording", ImVec2(150, 30))) {
                g_imu_recording.store(false);
                std::lock_guard<std::mutex> lock(g_imu_csv_mutex);
                if (g_imu_csv_file.is_open()) {
                    g_imu_csv_file.close();
                }
            }
        }

        ImGui::SameLine();
        ImGui::TextDisabled("(saves to slam_gui.exe directory)");
    }

    ImGui::Spacing();
#endif

    // Sensor fusion comparison
    if (ImGui::CollapsingHeader("Sensor Fusion", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Columns(4, "fusion_cols", true);
        ImGui::SetColumnWidth(0, 80);

        ImGui::Text(""); ImGui::NextColumn();
        ImGui::Text("Wheel Odom"); ImGui::NextColumn();
        ImGui::Text("SLAM"); ImGui::NextColumn();
        ImGui::Text("Fused"); ImGui::NextColumn();
        ImGui::Separator();

        Pose3D wheel = g_shared.getWheelOdomPose();
        Pose3D slam = g_shared.getSlamPose();
        Pose3D fused = g_shared.getFusedPose();

        ImGui::Text("X (m)"); ImGui::NextColumn();
        ImGui::Text("%.3f", wheel.x); ImGui::NextColumn();
        ImGui::Text("%.3f", slam.x); ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "%.3f", fused.x); ImGui::NextColumn();

        ImGui::Text("Y (m)"); ImGui::NextColumn();
        ImGui::Text("%.3f", wheel.y); ImGui::NextColumn();
        ImGui::Text("%.3f", slam.y); ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "%.3f", fused.y); ImGui::NextColumn();

        ImGui::Text("Yaw (deg)"); ImGui::NextColumn();
        ImGui::Text("%.1f", wheel.yaw * 180.0f / 3.14159f); ImGui::NextColumn();
        ImGui::Text("%.1f", slam.yaw * 180.0f / 3.14159f); ImGui::NextColumn();
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "%.1f", fused.yaw * 180.0f / 3.14159f); ImGui::NextColumn();

        ImGui::Columns(1);

        ImGui::Spacing();
        MotionState motion = g_shared.getMotionState();
        ImGui::Text("Motion State: %s", getMotionStateName(motion));
    }

    ImGui::Spacing();

    // VESC detailed telemetry
    if (ImGui::CollapsingHeader("VESC Telemetry", ImGuiTreeNodeFlags_DefaultOpen)) {
        MotorStatus ml = g_shared.getMotorStatus(0);
        MotorStatus mr = g_shared.getMotorStatus(1);

        ImGui::Columns(3, "vesc_cols", true);
        ImGui::SetColumnWidth(0, 100);

        ImGui::Text("Parameter"); ImGui::NextColumn();
        ImGui::Text("Left (ID %d)", g_config.vesc_left_id); ImGui::NextColumn();
        ImGui::Text("Right (ID %d)", g_config.vesc_right_id); ImGui::NextColumn();
        ImGui::Separator();

        ImGui::Text("ERPM"); ImGui::NextColumn();
        ImGui::Text("%d", ml.erpm); ImGui::NextColumn();
        ImGui::Text("%d", mr.erpm); ImGui::NextColumn();

        ImGui::Text("Duty"); ImGui::NextColumn();
        ImGui::Text("%.3f", ml.duty); ImGui::NextColumn();
        ImGui::Text("%.3f", mr.duty); ImGui::NextColumn();

        ImGui::Text("Current (A)"); ImGui::NextColumn();
        ImGui::Text("%.1f", ml.current_motor); ImGui::NextColumn();
        ImGui::Text("%.1f", mr.current_motor); ImGui::NextColumn();

        ImGui::Text("FET Temp (C)"); ImGui::NextColumn();
        ImGui::Text("%.0f", ml.temp_fet); ImGui::NextColumn();
        ImGui::Text("%.0f", mr.temp_fet); ImGui::NextColumn();

        ImGui::Text("Motor Temp (C)"); ImGui::NextColumn();
        ImGui::Text("%.0f", ml.temp_motor); ImGui::NextColumn();
        ImGui::Text("%.0f", mr.temp_motor); ImGui::NextColumn();

        ImGui::Text("Voltage (V)"); ImGui::NextColumn();
        ImGui::Text("%.1f", ml.voltage_in); ImGui::NextColumn();
        ImGui::Text("%.1f", mr.voltage_in); ImGui::NextColumn();

        ImGui::Text("Tachometer"); ImGui::NextColumn();
        ImGui::Text("%d", ml.tachometer); ImGui::NextColumn();
        ImGui::Text("%d", mr.tachometer); ImGui::NextColumn();

        ImGui::Columns(1);
    }

    ImGui::Spacing();

    // LiDAR status
    if (ImGui::CollapsingHeader("LiDAR Status", ImGuiTreeNodeFlags_DefaultOpen)) {
        LidarStatus lidar = g_shared.getLidarStatus();

        ImGui::Text("Connection: %s", getConnectionStatusName(lidar.connection));
        ImGui::Text("IP Address: %s", lidar.ip_address.c_str());
        ImGui::Text("Point Rate: %d pts/s", lidar.point_rate);
        ImGui::Text("IMU Rate: %d Hz", lidar.imu_rate);
        ImGui::Text("Gravity: %.2f m/s²", lidar.gravity_mag);
        ImGui::Text("IMU Initialized: %s", lidar.imu_initialized ? "Yes" : "No");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Recording
    ImGui::Text("Data Recording");
    bool is_recording = g_shared.isRecording();

    if (!is_recording) {
        static char record_filename[256] = "fusion_data.bin";
        ImGui::InputText("Filename", record_filename, sizeof(record_filename));
        SetTooltip("Filename for recorded sensor data.");

        if (ImGui::Button("Start Recording", ImVec2(150, 30))) {
            g_commands.push(Command::file(CommandType::START_RECORDING, record_filename));
        }
        SetTooltip("Record raw sensor data for offline analysis.");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "RECORDING: %s",
            g_shared.getRecordingFilename().c_str());
        ImGui::Text("Duration: %.1f s", g_shared.getRecordingDuration());

        if (ImGui::Button("Stop Recording", ImVec2(150, 30))) {
            g_commands.push(Command::simple(CommandType::STOP_RECORDING));
        }
    }
}

//==============================================================================
// UI: Tab - Settings
//==============================================================================
void DrawSettingsTab() {
    ImGui::Text("Settings");
    ImGui::Separator();
    ImGui::Spacing();

    // Connection settings
    if (ImGui::CollapsingHeader("Connection", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputText("LiDAR IP", g_config.lidar_ip, sizeof(g_config.lidar_ip));
        SetTooltip("IP address of the Livox Mid-360 LiDAR.");

        ImGui::InputText("Host IP", g_config.host_ip, sizeof(g_config.host_ip));
        SetTooltip("IP address of this computer (must be on same subnet as LiDAR).");

        ImGui::InputText("CAN Port", g_config.can_port, sizeof(g_config.can_port));
        SetTooltip("Serial port for CAN adapter (e.g., COM3).");

        ImGui::InputInt("VESC Left ID", &g_config.vesc_left_id);
        SetTooltip("CAN ID of left motor controller.");

        ImGui::InputInt("VESC Right ID", &g_config.vesc_right_id);
        SetTooltip("CAN ID of right motor controller.");

        ImGui::Spacing();
        if (ImGui::Button("Connect", ImVec2(100, 30))) {
            g_commands.push(Command::simple(CommandType::CONNECT_HARDWARE));
        }
        SetTooltip("Connect to LiDAR and motor controllers.");
        ImGui::SameLine();
        if (ImGui::Button("Disconnect", ImVec2(100, 30))) {
            g_commands.push(Command::simple(CommandType::DISCONNECT_HARDWARE));
        }
    }

    ImGui::Spacing();

    // Fusion parameters
    if (ImGui::CollapsingHeader("Sensor Fusion")) {
        ImGui::SliderFloat("SLAM Position Alpha", &g_config.slam_alpha_pos, 0.1f, 1.0f, "%.2f");
        SetTooltip("Low-pass filter coefficient for SLAM position. Higher = faster response, more noise.");

        ImGui::SliderFloat("SLAM Heading Alpha", &g_config.slam_alpha_hdg, 0.1f, 1.0f, "%.2f");
        SetTooltip("Low-pass filter coefficient for SLAM heading.");

        ImGui::SliderFloat("Straight Correction", &g_config.straight_correction, 0.5f, 10.0f, "%.1f /s");
        SetTooltip("Rate of SLAM correction during straight-line motion.");

        ImGui::SliderFloat("Turning Correction", &g_config.turning_correction, 1.0f, 20.0f, "%.1f /s");
        SetTooltip("Rate of SLAM correction during turning.");

        ImGui::SliderFloat("Stationary Correction", &g_config.stationary_correction, 0.1f, 1.0f, "%.2f /s");
        SetTooltip("Slow drift correction rate when stationary.");

        ImGui::SliderFloat("Stationary Threshold", &g_config.stationary_threshold, 0.01f, 0.1f, "%.3f m");
        SetTooltip("Minimum error to trigger stationary correction.");
    }

    ImGui::Spacing();

    // Motion parameters
    if (ImGui::CollapsingHeader("Motion Controller")) {
        ImGui::SliderFloat("Max Duty", &g_config.max_duty, 0.05f, 0.3f, "%.2f");
        SetTooltip("Maximum motor duty cycle.");

        ImGui::SliderFloat("Ramp Rate", &g_config.ramp_rate, 0.1f, 1.0f, "%.2f /s");
        SetTooltip("Duty cycle change rate for smooth acceleration.");

        ImGui::SliderFloat("Safety Margin", &g_config.safety_margin, 0.0f, 0.25f, "%.0f%%");
        SetTooltip("Additional margin on min duty thresholds.");
    }

    ImGui::Spacing();

    // SLAM parameters
    if (ImGui::CollapsingHeader("SLAM")) {
        ImGui::SliderFloat("Voxel Size", &g_config.voxel_size, 0.01f, 0.5f, "%.2f m");
        SetTooltip("Voxel filter size for map points. Applied when mapping starts.");

        ImGui::SliderFloat("Blind Distance", &g_config.blind_distance, 0.1f, 2.0f, "%.2f m");
        SetTooltip("Ignore LiDAR returns closer than this. (Requires restart)");

        ImGui::SliderInt("Max Iterations", &g_config.max_iterations, 1, 6);
        SetTooltip("IEKF iterations per scan. (Requires restart)");

        ImGui::SliderInt("Max Points ICP", &g_config.max_points_icp, 500, 5000);
        SetTooltip("CRITICAL: Max points for IEKF matching. Lower = faster. 2000 recommended. (Requires restart)");

        // Display in K for readability
        int max_map_k = g_config.max_map_points / 1000;
        if (ImGui::SliderInt("Max Map Points (K)", &max_map_k, 100, 2000)) {
            g_config.max_map_points = max_map_k * 1000;
        }
        SetTooltip("Max points in map. Prevents ikd-tree slowdown at large maps. 500K recommended. (Requires restart)");

        ImGui::SliderFloat("Gyro Covariance", &g_config.gyr_cov, 0.01f, 1.0f, "%.2f");
        SetTooltip("IMU gyroscope noise covariance. Applied when mapping starts.");

        ImGui::SliderInt("Point Filter", &g_config.point_filter, 1, 5);
        SetTooltip("Keep every Nth valid point. (Requires restart)");

        ImGui::Checkbox("Deskew Enabled", &g_config.deskew_enabled);
        SetTooltip("Enable motion compensation for LiDAR scans. Applied when mapping starts.");

        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Flyaway Protection");

        ImGui::SliderFloat("Max Pos Jump (m)", &g_config.max_position_jump, 0.1f, 2.0f, "%.2f");
        SetTooltip("Reject IEKF update if position jumps more than this. 0.3m recommended for slow robots. (Requires restart)");

        ImGui::SliderFloat("Max Rot Jump (deg)", &g_config.max_rotation_jump, 5.0f, 60.0f, "%.1f");
        SetTooltip("Reject IEKF update if rotation jumps more than this. 20 deg recommended. (Requires restart)");

        ImGui::SliderInt("Min Effective Pts", &g_config.min_effective_points, 10, 200);
        SetTooltip("Minimum matched points for valid IEKF update. Below this, update is rejected. (Requires restart)");

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Vibration Compensation");

        if (ImGui::SliderFloat("IMU LPF Alpha", &g_config.imu_lpf_alpha, 0.0f, 0.9f, "%.2f")) {
            // Apply immediately if SLAM is running
            if (g_slam) {
                g_slam->setImuLpfAlpha(g_config.imu_lpf_alpha);
            }
        }
        SetTooltip("Low-pass filter for IMU vibration. 0=disabled, 0.2-0.4=moderate (fast platforms), 0.5-0.7=heavy (vibrating platforms). Lower = more filtering.");
    }

    ImGui::Spacing();

    // Viewer settings
    if (ImGui::CollapsingHeader("3D Viewer", ImGuiTreeNodeFlags_DefaultOpen)) {
        static bool viewer_changed = false;

        // Performance test toggle - prominent at top
        ImGui::PushStyleColor(ImGuiCol_Text, g_config.disable_map_viewer ?
            ImVec4(1.0f, 0.3f, 0.3f, 1.0f) : ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
        ImGui::Checkbox("Disable Map Viewer", &g_config.disable_map_viewer);
        ImGui::PopStyleColor();
        SetTooltip("Disable all map visualization updates. Use to test SLAM performance without viewer overhead.");

        ImGui::Separator();

        ImGui::SliderFloat("Point Size", &g_config.viewer_point_size, 1.0f, 10.0f, "%.1f px");
        SetTooltip("Size of rendered points in pixels.");
        if (ImGui::IsItemDeactivatedAfterEdit()) viewer_changed = true;

        const char* colormap_names[] = { "Grayscale", "Turbo (Rainbow)", "Viridis", "Plasma", "Inferno", "Height" };
        ImGui::Combo("Colormap", &g_config.viewer_colormap, colormap_names, IM_ARRAYSIZE(colormap_names));
        SetTooltip("Color scheme for point cloud visualization.");
        if (ImGui::IsItemDeactivatedAfterEdit()) viewer_changed = true;

        ImGui::Checkbox("Enable LOD", &g_config.viewer_enable_lod);
        SetTooltip("Level of Detail - reduces points at distance for performance. Disable to see all points.");
        if (ImGui::IsItemDeactivatedAfterEdit()) viewer_changed = true;

        if (g_config.viewer_enable_lod) {
            ImGui::SliderFloat("LOD Distance", &g_config.viewer_lod_distance, 5.0f, 50.0f, "%.0f m");
            SetTooltip("Distance at which LOD reduction starts.");
            if (ImGui::IsItemDeactivatedAfterEdit()) viewer_changed = true;
        }

        ImGui::SliderFloat("Min Zoom", &g_config.viewer_min_zoom, 0.1f, 5.0f, "%.1f m");
        SetTooltip("Minimum camera distance (closest zoom).");
        if (ImGui::IsItemDeactivatedAfterEdit()) viewer_changed = true;

        ImGui::SliderFloat("Max Zoom", &g_config.viewer_max_zoom, 50.0f, 1000.0f, "%.0f m");
        SetTooltip("Maximum camera distance (furthest zoom out).");
        if (ImGui::IsItemDeactivatedAfterEdit()) viewer_changed = true;

        int max_points_k = g_config.viewer_max_points / 1000;
        ImGui::SliderInt("Max Points (K)", &max_points_k, 100, 10000, "%dK");
        g_config.viewer_max_points = max_points_k * 1000;
        SetTooltip("Maximum points to render (performance limit).");
        if (ImGui::IsItemDeactivatedAfterEdit()) viewer_changed = true;

        // Apply changes to viewers
        if (viewer_changed && g_viewers_initialized) {
            slam::viz::ViewerConfig config;
            config.point_cloud.point_size = g_config.viewer_point_size;
            config.point_cloud.colormap = static_cast<slam::viz::Colormap>(g_config.viewer_colormap);
            config.point_cloud.enable_lod = g_config.viewer_enable_lod;
            config.point_cloud.lod_distance = g_config.viewer_lod_distance;
            config.point_cloud.max_visible_points = g_config.viewer_max_points;

            if (g_mapping_viewer) g_mapping_viewer->setConfig(config);
            if (g_localization_viewer) g_localization_viewer->setConfig(config);
            viewer_changed = false;
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Save/Load settings
    if (ImGui::Button("Save Settings", ImVec2(120, 30))) {
        if (SaveSettings()) {
            g_shared.setStatusMessage("Settings saved to " + std::string(SETTINGS_FILE));
        } else {
            g_shared.setErrorMessage("Failed to save settings");
        }
    }
    SetTooltip("Save all settings to configuration file.");

    ImGui::SameLine();
    if (ImGui::Button("Load Defaults", ImVec2(120, 30))) {
        g_config = AppConfig();  // Reset to defaults
        // Apply viewer defaults
        if (g_viewers_initialized) {
            slam::viz::ViewerConfig config;
            config.point_cloud.point_size = g_config.viewer_point_size;
            config.point_cloud.colormap = static_cast<slam::viz::Colormap>(g_config.viewer_colormap);
            config.point_cloud.enable_lod = g_config.viewer_enable_lod;
            config.point_cloud.lod_distance = g_config.viewer_lod_distance;
            config.point_cloud.max_visible_points = g_config.viewer_max_points;
            if (g_mapping_viewer) g_mapping_viewer->setConfig(config);
            if (g_localization_viewer) g_localization_viewer->setConfig(config);
        }
        g_shared.setStatusMessage("Settings reset to defaults");
    }
    SetTooltip("Reset all settings to default values.");
}

//==============================================================================
// Main Window
//==============================================================================
void DrawMainWindow() {
    ImGuiViewport* viewport = ImGui::GetMainViewport();

    // Main window below status bar
    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x, viewport->Pos.y + 40));
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x, viewport->Size.y - 40));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoBringToFrontOnFocus;

    ImGui::Begin("##Main", nullptr, flags);

    // Tab bar
    if (ImGui::BeginTabBar("MainTabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Operate")) {
            g_current_tab = 0;
            DrawOperateTab();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Mapping")) {
            g_current_tab = 1;
            DrawMappingTab();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Localization")) {
            g_current_tab = 2;
            DrawLocalizationTab();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Calibration")) {
            g_current_tab = 3;
            DrawCalibrationTab();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Hull & Probe")) {
            g_current_tab = 4;
            DrawHullTab();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Diagnostics")) {
            g_current_tab = 5;
            DrawDiagnosticsTab();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Settings")) {
            g_current_tab = 6;
            DrawSettingsTab();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    // Status message at bottom
    std::string status = g_shared.getStatusMessage();
    if (!status.empty()) {
        ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 30);
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.4f, 1.0f), "%s", status.c_str());
    }

    // Error message popup
    std::string error = g_shared.getErrorMessage();
    if (!error.empty()) {
        ImGui::OpenPopup("Error");
        if (ImGui::BeginPopupModal("Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.2f, 1.0f), "%s", error.c_str());
            if (ImGui::Button("OK", ImVec2(100, 30))) {
                g_shared.clearErrorMessage();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }

    // VESC Diagnostics popup
    if (g_show_vesc_diagnostics) {
        ImGui::OpenPopup("VESC Diagnostics");
    }
    if (ImGui::BeginPopupModal("VESC Diagnostics", &g_show_vesc_diagnostics, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("VESC Connection Diagnostics");
        ImGui::Separator();

        // Current settings
        ImGui::Text("Current Settings:");
        ImGui::BulletText("COM Port: %s", g_config.can_port);
        ImGui::BulletText("Left VESC ID: %d", g_config.vesc_left_id);
        ImGui::BulletText("Right VESC ID: %d", g_config.vesc_right_id);
        ImGui::Spacing();

        // Run diagnostics button - sets flag for ControlThread to process
        // (Single-thread VESC architecture: all VESC operations in ControlThread)
        if (!g_vesc_diag_running && !g_vesc_diag_requested.load()) {
            if (ImGui::Button("Run Diagnostics", ImVec2(150, 30))) {
                // Just set the flag - ControlThread will process it
                g_vesc_diag_requested.store(true);
            }
        } else {
            ImGui::Text("Running diagnostics...");
        }

        ImGui::Spacing();
        ImGui::Separator();

        // Diagnostic log
        if (!g_vesc_diag_log.empty()) {
            ImGui::Text("Results:");
            ImGui::BeginChild("DiagLog", ImVec2(400, 200), true);
            ImGui::TextUnformatted(g_vesc_diag_log.c_str());
            ImGui::EndChild();
        }

        // Tips
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Troubleshooting Tips:");
        ImGui::BulletText("Unplug and replug CANable to reset");
        ImGui::BulletText("Check CAN bus termination (120 ohm)");
        ImGui::BulletText("Verify VESC IDs match in Settings tab");
        ImGui::BulletText("Ensure battery is connected to VESCs");

        ImGui::Spacing();
        if (ImGui::Button("Close", ImVec2(100, 30))) {
            g_show_vesc_diagnostics = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // LiDAR Diagnostics popup
    if (g_show_lidar_diagnostics) {
        ImGui::OpenPopup("LiDAR Diagnostics");
    }
    if (ImGui::BeginPopupModal("LiDAR Diagnostics", &g_show_lidar_diagnostics, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("LiDAR Connection Diagnostics");
        ImGui::Separator();

        // Current settings
        ImGui::Text("Current Settings:");
        ImGui::BulletText("LiDAR IP: %s", g_config.lidar_ip);
        ImGui::BulletText("Host IP: %s", g_config.host_ip);
        ImGui::Spacing();

        // Current status
        ImGui::Text("Current Status:");
        bool has_instance = (g_lidar != nullptr);
        bool is_streaming = has_instance && g_lidar->isStreaming();
        LidarStatus status = g_shared.getLidarStatus();
        bool is_connected = (status.connection == ConnectionStatus::CONNECTED);

        if (is_streaming) {
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "  Status: Streaming data");
            ImGui::BulletText("Point rate: %d pts/s", status.point_rate);
            ImGui::BulletText("IMU rate: %d Hz", status.imu_rate);
        } else if (is_connected) {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.2f, 1.0f), "  Status: Connected but NOT streaming");
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "  LiDAR may be powered off or in standby");
        } else {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "  Status: Disconnected");
        }
        ImGui::Spacing();

        // Run diagnostics button
        if (!g_lidar_diag_running) {
            if (ImGui::Button("Run Diagnostics", ImVec2(150, 30))) {
                g_lidar_diag_running = true;
                g_lidar_diag_log.clear();
                g_lidar_diag_log += "=== LiDAR Diagnostics ===\n\n";

                // Step 1: Ping the LiDAR IP
                g_lidar_diag_log += "1. Network Connectivity:\n";
                std::string ping_cmd = "ping -n 1 -w 1000 " + std::string(g_config.lidar_ip);
                FILE* pipe = _popen(ping_cmd.c_str(), "r");
                if (pipe) {
                    char buffer[256];
                    std::string ping_output;
                    while (fgets(buffer, sizeof(buffer), pipe)) {
                        ping_output += buffer;
                    }
                    int result = _pclose(pipe);
                    if (result == 0) {
                        g_lidar_diag_log += "   [OK] Ping to " + std::string(g_config.lidar_ip) + " successful\n";
                    } else {
                        g_lidar_diag_log += "   [FAIL] Cannot ping " + std::string(g_config.lidar_ip) + "\n";
                        g_lidar_diag_log += "   -> Check LiDAR power cable\n";
                        g_lidar_diag_log += "   -> Check Ethernet connection\n";
                        g_lidar_diag_log += "   -> Verify IP address in Settings\n";
                    }
                } else {
                    g_lidar_diag_log += "   [ERROR] Could not run ping command\n";
                }

                // Step 2: Check host IP configuration
                g_lidar_diag_log += "\n2. Host Network Configuration:\n";
                std::string host_ip = g_config.host_ip;
                // Check if host IP is in same subnet as LiDAR
                std::string lidar_subnet = std::string(g_config.lidar_ip).substr(0, std::string(g_config.lidar_ip).rfind('.'));
                std::string host_subnet = host_ip.substr(0, host_ip.rfind('.'));
                if (lidar_subnet == host_subnet) {
                    g_lidar_diag_log += "   [OK] Host IP (" + host_ip + ") in same subnet as LiDAR\n";
                } else {
                    g_lidar_diag_log += "   [WARN] Host IP (" + host_ip + ") may be in different subnet\n";
                    g_lidar_diag_log += "   -> LiDAR subnet: " + lidar_subnet + ".x\n";
                    g_lidar_diag_log += "   -> Configure network adapter to " + lidar_subnet + ".x\n";
                }

                // Step 3: Try to connect and stream
                g_lidar_diag_log += "\n3. LiDAR Connection Test:\n";
                if (!g_lidar) {
                    g_lidar = std::make_unique<slam::LivoxMid360>();
                    g_lidar->setPointCloudCallback(OnPointCloud);
                    g_lidar->setIMUCallback(OnIMU);
                    g_lidar_diag_log += "   Created new LiDAR instance\n";
                } else {
                    g_lidar_diag_log += "   Using existing LiDAR instance\n";
                }

                if (g_lidar->isStreaming()) {
                    g_lidar_diag_log += "   [OK] Already streaming!\n";
                } else {
                    g_lidar_diag_log += "   Attempting connection...\n";
                    if (g_lidar->connect(g_config.lidar_ip, g_config.host_ip)) {
                        g_lidar_diag_log += "   [OK] Connection command sent\n";

                        // Wait for streaming to start
                        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

                        if (g_lidar->isStreaming()) {
                            g_lidar_diag_log += "   [OK] LiDAR is now streaming!\n";
                            LidarStatus new_status = g_shared.getLidarStatus();
                            char buf[128];
                            snprintf(buf, sizeof(buf), "   Point rate: %d pts/s, IMU rate: %d Hz\n",
                                     new_status.point_rate, new_status.imu_rate);
                            g_lidar_diag_log += buf;
                        } else {
                            g_lidar_diag_log += "   [FAIL] Connected but no data streaming\n";
                            g_lidar_diag_log += "   -> LiDAR may not be powered (only network chip active)\n";
                            g_lidar_diag_log += "   -> Check the power LED on the LiDAR\n";
                            g_lidar_diag_log += "   -> Try power cycling the LiDAR\n";
                        }
                    } else {
                        g_lidar_diag_log += "   [FAIL] Connection failed\n";
                        g_lidar_diag_log += "   -> Check firewall settings (UDP ports 56000-56100)\n";
                        g_lidar_diag_log += "   -> Try disabling Windows Firewall temporarily\n";
                    }
                }

                g_lidar_diag_log += "\n=== Diagnostics Complete ===\n";
                g_lidar_diag_running = false;
            }
        } else {
            ImGui::Text("Running diagnostics...");
        }

        ImGui::Spacing();
        ImGui::Separator();

        // Diagnostic log
        if (!g_lidar_diag_log.empty()) {
            ImGui::Text("Results:");
            ImGui::BeginChild("LidarDiagLog", ImVec2(450, 250), true);
            ImGui::TextUnformatted(g_lidar_diag_log.c_str());
            ImGui::EndChild();
        }

        // Tips
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Troubleshooting Tips:");
        ImGui::BulletText("Green = streaming, Yellow = connected but no data");
        ImGui::BulletText("If yellow: LiDAR chip has power but motor may be off");
        ImGui::BulletText("Check power LED on LiDAR unit itself");
        ImGui::BulletText("Try power cycling the LiDAR (wait 10s after power-on)");
        ImGui::BulletText("Ensure host network adapter is set to %s", g_config.host_ip);

        ImGui::Spacing();
        if (ImGui::Button("Close", ImVec2(100, 30))) {
            g_show_lidar_diagnostics = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // System Log popup
    if (g_show_system_log) {
        ImGui::OpenPopup("System Log");
    }
    if (ImGui::BeginPopupModal("System Log", &g_show_system_log, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("System Error/Warning Log");
        ImGui::Separator();

        // Filter options
        static bool show_info = true;
        static bool show_warnings = true;
        static bool show_errors = true;
        ImGui::Checkbox("Info", &show_info);
        ImGui::SameLine();
        ImGui::Checkbox("Warnings", &show_warnings);
        ImGui::SameLine();
        ImGui::Checkbox("Errors", &show_errors);
        ImGui::SameLine(300);
        if (ImGui::Button("Clear Log")) {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            g_system_log.clear();
        }
        ImGui::Spacing();

        // Log display
        ImGui::BeginChild("LogContent", ImVec2(600, 400), true, ImGuiWindowFlags_HorizontalScrollbar);
        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            for (auto it = g_system_log.rbegin(); it != g_system_log.rend(); ++it) {
                const auto& entry = *it;

                // Filter by level
                if (entry.level == LogLevel::INFO && !show_info) continue;
                if (entry.level == LogLevel::WARNING && !show_warnings) continue;
                if (entry.level == LogLevel::ERROR_LEVEL && !show_errors) continue;

                // Format timestamp
                auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    entry.timestamp.time_since_epoch()) % 1000;
                struct tm* tm_info = localtime(&time_t);
                char time_str[32];
                snprintf(time_str, sizeof(time_str), "%02d:%02d:%02d.%03d",
                         tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec, (int)ms.count());

                // Color by level
                ImVec4 color;
                const char* level_str;
                switch (entry.level) {
                    case LogLevel::ERROR_LEVEL:
                        color = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
                        level_str = "ERROR";
                        break;
                    case LogLevel::WARNING:
                        color = ImVec4(1.0f, 0.8f, 0.2f, 1.0f);
                        level_str = "WARN ";
                        break;
                    default:
                        color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
                        level_str = "INFO ";
                        break;
                }

                ImGui::TextColored(color, "[%s] [%s] [%s] %s",
                                   time_str, level_str, entry.source.c_str(), entry.message.c_str());
            }
        }
        ImGui::EndChild();

        ImGui::Spacing();
        if (ImGui::Button("Close", ImVec2(100, 30))) {
            g_show_system_log = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::End();
}

//==============================================================================
// Main Entry Point
//==============================================================================
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    // Log application startup
    LOG_INFO("System", "SLAM Control GUI starting...");

    // Get executable directory
    char exePath[MAX_PATH];
    GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    strncpy(g_config.map_directory, exeDir.c_str(), sizeof(g_config.map_directory) - 1);

    // Load saved settings (before anything else)
    if (LoadSettings()) {
        LOG_INFO("System", "Settings loaded successfully");
    } else {
        LOG_INFO("System", "Using default settings");
    }

    // Create window
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L,
                       GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr,
                       L"SLAM Control GUI", nullptr };
    RegisterClassExW(&wc);

    HWND hwnd = CreateWindowW(wc.lpszClassName, L"SLAM Control GUI",
                              WS_OVERLAPPEDWINDOW, 50, 50, 1400, 900,
                              nullptr, nullptr, wc.hInstance, nullptr);
    g_hwnd = hwnd;  // Store for focus checking

    // Initialize Direct3D
    if (!CreateDeviceD3D(hwnd)) {
        CleanupDeviceD3D();
        UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);

    // Setup Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Style
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.FramePadding = ImVec2(8, 4);
    style.ItemSpacing = ImVec2(8, 6);
    style.TabRounding = 4.0f;

    // Colors - dark blue theme
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.12f, 0.15f, 1.00f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.12f, 0.14f, 0.18f, 1.00f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.18f, 0.22f, 1.00f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.20f, 0.22f, 0.28f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.20f, 0.35f, 0.50f, 1.00f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.45f, 0.60f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.20f, 0.35f, 0.50f, 1.00f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.45f, 0.60f, 1.00f);
    colors[ImGuiCol_Tab] = ImVec4(0.18f, 0.30f, 0.42f, 1.00f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.26f, 0.45f, 0.60f, 1.00f);
    colors[ImGuiCol_TabSelected] = ImVec4(0.24f, 0.40f, 0.55f, 1.00f);

    // Setup backends
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    // Initialize GPU-accelerated map viewers
    g_mapping_viewer = std::make_unique<slam::viz::SlamViewer>();
    g_localization_viewer = std::make_unique<slam::viz::SlamViewer>();
    if (g_mapping_viewer->initWithDevice(g_pd3dDevice, g_pd3dDeviceContext) &&
        g_localization_viewer->initWithDevice(g_pd3dDevice, g_pd3dDeviceContext)) {
        g_viewers_initialized = true;
        // Configure viewers from loaded settings
        slam::viz::ViewerConfig config;
        config.point_cloud.point_size = g_config.viewer_point_size;
        config.point_cloud.colormap = static_cast<slam::viz::Colormap>(g_config.viewer_colormap);
        config.point_cloud.enable_lod = g_config.viewer_enable_lod;
        config.point_cloud.lod_distance = g_config.viewer_lod_distance;
        config.point_cloud.max_visible_points = g_config.viewer_max_points;
        g_mapping_viewer->setConfig(config);
        g_mapping_viewer->setMode(slam::viz::ViewMode::SCANNING);
        g_localization_viewer->setConfig(config);
        g_localization_viewer->setMode(slam::viz::ViewMode::SCANNING);  // Can switch to LOCALIZATION for hull
    } else {
        g_shared.setErrorMessage("Failed to initialize GPU viewers");
    }

    // Scan for maps
    ScanForMaps();

    // Initialize gamepad (SDL2 - supports PS5 DualSense, Xbox, etc.)
#ifdef HAS_SDL2
    g_gamepad = std::make_unique<slam::Gamepad>();
    if (g_gamepad->init()) {
        std::string name = g_gamepad->getControllerName();
        g_shared.setStatusMessage("Gamepad connected: " + name);
        g_gamepad->setLEDColor(255, 255, 255);  // White = ready
        g_gamepad->rumble(0.3f, 0.3f, 200);     // Startup rumble
    } else {
        g_shared.setStatusMessage("No gamepad found - connect controller");
    }
    // Configure drive parameters
    g_gamepad_config.max_linear_velocity = 0.6f;
    g_gamepad_config.max_angular_velocity = 3.5f;
    g_gamepad_config.stick_deadzone = 0.12f;
    g_gamepad_config.max_speed_scale = 0.8f;
#endif

    // Start worker thread (SLAM processing, state management)
    g_worker_thread = std::thread(WorkerThread);

    // Start high-priority control thread (motor control, CAN bus)
    // This runs at THREAD_PRIORITY_HIGHEST to ensure consistent motor timing
    g_control_thread = std::thread(ControlThread);

    // Main loop
    bool done = false;
    while (!done) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
                done = true;
        }
        if (done) break;

        // Update time
        auto now = std::chrono::steady_clock::now();
        g_current_time = std::chrono::duration<float>(now - g_start_time).count();

        // Update gamepad
        UpdateGamepad();

        // Check keyboard E-STOP (Escape key)
        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
            if (!g_e_stop.load()) {
                g_e_stop.store(true);
                g_shared.e_stop.store(true);
                g_commands.push(Command::simple(CommandType::E_STOP));
            }
        }

        // Keyboard control (fallback when gamepad not connected)
        // WASD or Arrow keys: W/Up=forward, S/Down=backward, A/Left=turn left, D/Right=turn right
        float kb_linear = 0.0f, kb_angular = 0.0f;
        bool kb_active = false;
        // Only process keyboard when:
        // 1. Our window has focus (not typing in another app)
        // 2. Not typing in a text input field within our GUI
        bool window_has_focus = (GetForegroundWindow() == g_hwnd);
        bool typing_in_textfield = io.WantTextInput;
        if (window_has_focus && !typing_in_textfield) {
            if (GetAsyncKeyState('W') & 0x8000 || GetAsyncKeyState(VK_UP) & 0x8000) {
                kb_linear = 1.0f;
                kb_active = true;
            }
            if (GetAsyncKeyState('S') & 0x8000 || GetAsyncKeyState(VK_DOWN) & 0x8000) {
                kb_linear = -1.0f;
                kb_active = true;
            }
            if (GetAsyncKeyState('A') & 0x8000 || GetAsyncKeyState(VK_LEFT) & 0x8000) {
                kb_angular = 1.0f;
                kb_active = true;
            }
            if (GetAsyncKeyState('D') & 0x8000 || GetAsyncKeyState(VK_RIGHT) & 0x8000) {
                kb_angular = -1.0f;
                kb_active = true;
            }
        }

        // Update velocity targets directly (bypasses command queue for low latency)
        // Control thread reads these atomics at 50Hz with highest priority
        if (!g_e_stop.load()) {
            float linear = 0.0f, angular = 0.0f;

            if (g_gamepad_connected) {
                // Use pre-computed drive command from getDriveCommand()
                linear = g_drive_cmd.linear_velocity;
                angular = g_drive_cmd.angular_velocity;
            } else if (kb_active) {
                // Use keyboard (scaled to max_speed)
                linear = kb_linear * g_config.max_speed * 0.5f;  // 50% max for keyboard
                angular = kb_angular * 1.5f;
            }

            // Direct atomic update - control thread reads this immediately
            g_target_linear.store(linear);
            g_target_angular.store(angular);
        } else {
            // E-STOP active - ensure zero velocity
            g_target_linear.store(0.0f);
            g_target_angular.store(0.0f);
        }

        // Handle resize
        if (g_SwapChainOccluded && g_pSwapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED) {
            Sleep(10);
            continue;
        }
        g_SwapChainOccluded = false;

        if (g_ResizeWidth != 0 && g_ResizeHeight != 0) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        // Start frame
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        // Draw GUI
        DrawStatusBar();
        DrawMainWindow();

        // Render
        ImGui::Render();
        const float clear_color[] = { 0.08f, 0.10f, 0.12f, 1.0f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        g_pSwapChain->Present(1, 0);
    }

    // Cleanup
    g_running.store(false);
    if (g_control_thread.joinable()) {
        g_control_thread.join();  // Join control thread first (faster shutdown)
    }
    if (g_worker_thread.joinable()) {
        g_worker_thread.join();
    }

    // Shutdown gamepad
#ifdef HAS_SDL2
    if (g_gamepad) {
        g_gamepad->setLEDColor(0, 0, 0);  // Turn off LED
        g_gamepad->shutdown();
        g_gamepad.reset();
    }
#endif

    // Cleanup GPU viewers before D3D11 shutdown
    g_mapping_viewer.reset();
    g_localization_viewer.reset();
    g_viewers_initialized = false;

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);

    return 0;
}

//==============================================================================
// DirectX 11 Helper Functions
//==============================================================================
bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };
    HRESULT res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                                 createDeviceFlags, featureLevelArray, 2,
                                                 D3D11_SDK_VERSION, &sd, &g_pSwapChain,
                                                 &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res == DXGI_ERROR_UNSUPPORTED)
        res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr,
                                            createDeviceFlags, featureLevelArray, 2,
                                            D3D11_SDK_VERSION, &sd, &g_pSwapChain,
                                            &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res != S_OK)
        return false;

    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}

void CreateRenderTarget() {
    ID3D11Texture2D* pBackBuffer;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget() {
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = nullptr; }
}

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED)
            return 0;
        g_ResizeWidth = (UINT)LOWORD(lParam);
        g_ResizeHeight = (UINT)HIWORD(lParam);
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU)
            return 0;
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}
