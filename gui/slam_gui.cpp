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
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <fstream>
#include <sstream>
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

// Hardware drivers
#define ENABLE_HARDWARE 1  // Set to 0 for simulation mode

#if ENABLE_HARDWARE
#include "slam/slam_engine.hpp"
#include "slam/livox_mid360.hpp"
#include "slam/vesc_driver.hpp"
#include "slam/sensor_fusion.hpp"
#include "slam/motion_controller.hpp"
#endif

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
    int max_iterations = 3;
    float gyr_cov = 0.1f;
    bool deskew_enabled = true;
    int point_filter = 3;  // Keep every Nth point (1=all, 3=keep 1/3)

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
static std::thread g_worker_thread;

// Hardware drivers
static std::atomic<bool> g_hardware_initialized{false};

#if ENABLE_HARDWARE
static std::unique_ptr<slam::LivoxMid360> g_lidar;
static std::unique_ptr<slam::VescDriver> g_vesc;
static std::unique_ptr<slam::SlamEngine> g_slam;
static std::unique_ptr<slam::SensorFusion> g_fusion;
static std::unique_ptr<slam::MotionController> g_motion;

// IMU queue for thread-safe callback handling
static std::mutex g_imu_mutex;
static std::vector<slam::ImuData> g_imu_queue;

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

        // Update LED color based on motion and state
        if (!g_e_stop.load()) {
            AppState app_state = g_shared.getAppState();
            bool is_moving = (std::abs(g_drive_cmd.linear_velocity) > 0.01f ||
                             std::abs(g_drive_cmd.angular_velocity) > 0.1f);

            if (is_moving) {
                g_gamepad->setLEDColor(0, 100, 255);  // Blue = moving
            } else if (app_state == AppState::MAPPING) {
                g_gamepad->setLEDColor(0, 255, 0);    // Green = mapping
            } else if (app_state == AppState::LOCALIZED) {
                g_gamepad->setLEDColor(0, 200, 100);  // Teal = localized
            } else if (app_state == AppState::OPERATING) {
                g_gamepad->setLEDColor(100, 200, 255); // Cyan = operating
            } else {
                g_gamepad->setLEDColor(255, 255, 255); // White = idle
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
    // Keep buffer bounded
    if (g_imu_queue.size() > 100) {
        g_imu_queue.erase(g_imu_queue.begin(), g_imu_queue.begin() + 50);
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
}

// Helper: Update map points for 3D visualization
// Called periodically (not every frame) to avoid performance issues
void UpdateMapPointsForVisualization() {
    if (!g_slam || !g_slam->isInitialized()) return;

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
        g_shared.setErrorMessage("Failed to connect to LiDAR at " + std::string(g_config.lidar_ip));
        ShutdownHardware();  // Cleanup partial init
        return false;
    }

    // Update LiDAR status
    LidarStatus lidar_status;
    lidar_status.connection = ConnectionStatus::CONNECTED;
    lidar_status.ip_address = g_config.lidar_ip;
    g_shared.setLidarStatus(lidar_status);

    g_shared.setStatusMessage("Initializing VESC...");

    // Create VESC driver
    g_vesc = std::make_unique<slam::VescDriver>();
    if (!g_vesc->init(g_config.can_port, g_config.vesc_left_id, g_config.vesc_right_id)) {
        g_shared.setErrorMessage("Failed to initialize VESC on " + std::string(g_config.can_port));
        ShutdownHardware();  // Cleanup partial init
        return false;
    }

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
    slam_config.deskew_enabled = g_config.deskew_enabled;
    slam_config.save_map = true;
    if (!g_slam->init(slam_config)) {
        g_shared.setErrorMessage("Failed to initialize SLAM engine");
        ShutdownHardware();  // Cleanup partial init
        return false;
    }

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
    if (g_vesc) {
        g_vesc->stop();
        g_vesc->shutdown();
    }

    g_lidar.reset();
    g_vesc.reset();
    g_slam.reset();
    g_fusion.reset();
    g_motion.reset();

    g_shared.hardware_connected.store(false);
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
    float target_linear = 0.0f;
    float target_angular = 0.0f;

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
            switch (cmd->type) {
                case CommandType::E_STOP:
                    g_e_stop.store(true);
                    g_shared.e_stop.store(true);
                    if (g_motion) g_motion->emergencyStop();
                    break;

                case CommandType::RESET_E_STOP:
                    g_e_stop.store(false);
                    g_shared.e_stop.store(false);
                    g_shared.setAppState(AppState::IDLE);
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
                    if (g_slam) {
                        g_slam->reset();
                        g_slam->setLocalizationMode(false);
                    }
                    if (g_fusion) {
                        g_fusion->reset();
                    }
                    g_shared.setAppState(AppState::MAPPING);
                    g_shared.setStatusMessage("Mapping started");
                    g_shared.clearTrajectory();
                    break;

                case CommandType::STOP_MAPPING:
                    target_linear = target_angular = 0.0f;
                    if (g_motion) g_motion->stop();
                    g_shared.setAppState(AppState::IDLE);
                    g_shared.setStatusMessage("Mapping stopped");
                    break;

                case CommandType::START_OPERATING:
                    // Drive-only mode (no SLAM processing)
                    target_linear = target_angular = 0.0f;
                    g_shared.setAppState(AppState::OPERATING);
                    g_shared.setStatusMessage("Operating mode - manual drive only");
                    break;

                case CommandType::STOP_OPERATING:
                    target_linear = target_angular = 0.0f;
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
                    if (file_cmd && g_slam) {
                        g_shared.setStatusMessage("Loading map...");
                        if (g_slam->loadMap(file_cmd->path)) {
                            g_slam->setLocalizationMode(true);
                            g_shared.setAppState(AppState::MAP_LOADED);
                            g_shared.setStatusMessage("Map loaded: " + file_cmd->path);
                        } else {
                            g_shared.setErrorMessage("Failed to load map");
                        }
                    }
                    break;
                }

                case CommandType::RELOCALIZE:
                    if (g_shared.getAppState() == AppState::MAP_LOADED && g_slam) {
                        g_shared.setAppState(AppState::RELOCALIZING);
                        g_shared.setStatusMessage("Running coarse-to-fine ICP localization...");

                        RelocalizationProgress prog;
                        prog.running = true;
                        prog.progress = 0.0f;
                        prog.status_text = "Starting ICP alignment...";
                        g_shared.setRelocalizationProgress(prog);

                        // Check we have scan data
                        auto scan_points = g_slam->getCurrentScan();
                        if (scan_points.size() < 100) {
                            g_shared.setAppState(AppState::RELOCALIZE_FAILED);
                            g_shared.setErrorMessage("Insufficient scan data - wait for LiDAR");
                            prog.running = false;
                            prog.success = false;
                            prog.status_text = "Need more scan data";
                            g_shared.setRelocalizationProgress(prog);
                            break;
                        }

                        // Update progress
                        prog.status_text = "Running coarse alignment...";
                        prog.progress = 0.2f;
                        g_shared.setRelocalizationProgress(prog);

                        // Run the existing coarse-to-fine ICP (3-stage)
                        // Note: If pose hint was set via SET_POSE_HINT, it's already in g_slam
                        bool success = g_slam->globalRelocalize();

                        prog.progress = 0.9f;
                        prog.status_text = "Verifying alignment...";
                        g_shared.setRelocalizationProgress(prog);

                        // Get fitness score for confidence
                        double fitness = g_slam->getLocalizationFitness();

                        if (success) {
                            g_shared.setAppState(AppState::LOCALIZED);
                            g_shared.setStatusMessage("Localized successfully");

                            prog.running = false;
                            prog.success = true;
                            prog.confidence = static_cast<float>(fitness);
                            prog.status_text = "Aligned (fitness: " + std::to_string(int(fitness * 100)) + "%)";
                        } else {
                            g_shared.setAppState(AppState::RELOCALIZE_FAILED);
                            g_shared.setStatusMessage("Relocalization failed - try different position or pose hint");

                            prog.running = false;
                            prog.success = false;
                            prog.confidence = static_cast<float>(fitness);
                            prog.status_text = "Failed (fitness: " + std::to_string(int(fitness * 100)) + "%)";
                        }
                        g_shared.setRelocalizationProgress(prog);
                    }
                    break;

                case CommandType::SET_VELOCITY: {
                    auto* vel_cmd = std::get_if<VelocityCommand>(&cmd->payload);
                    if (vel_cmd) {
                        target_linear = vel_cmd->linear_mps;
                        target_angular = vel_cmd->angular_radps;
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

                    // Check for cancellation (peek at queue without blocking)
                    static std::atomic<bool> cancel_requested{false};
                    cancel_requested.store(false);
                    auto should_cancel = [&]() -> bool {
                        // Quick check - actual cancellation handled in main command loop
                        return cancel_requested.load();
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
                    target_linear = target_angular = 0.0f;
                    if (g_motion) g_motion->stop();
                    g_shared.setAppState(AppState::IDLE);
                    g_shared.setStatusMessage("Calibration cancelled");
                    break;

                case CommandType::CLEAR_MAP:
                    if (g_slam) {
                        g_slam->reset();
                    }
                    g_shared.clearTrajectory();
                    g_shared.setStatusMessage("Map cleared");
                    break;

                case CommandType::STOP_LOCALIZATION:
                    target_linear = target_angular = 0.0f;
                    if (g_motion) g_motion->stop();
                    if (g_slam) {
                        g_slam->setLocalizationMode(false);
                    }
                    g_shared.setAppState(AppState::IDLE);
                    g_shared.setStatusMessage("Localization stopped");
                    break;

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
        AppState state = g_shared.getAppState();
        bool do_slam = (state == AppState::MAPPING || state == AppState::LOCALIZED);

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

            // Run SLAM processing
            int processed = g_slam->process();
            if (processed > 0) {
                g_slam_process_count++;  // Debug counter
            }

            // Update SLAM state to GUI
            UpdateSlamState();

            // Update map points for 3D visualization (throttled internally)
            UpdateMapPointsForVisualization();
        }

        // 4. Update wheel odometry and sensor fusion
        if (g_vesc && g_fusion) {
            // Get odometry from VESC
            slam::VescOdometry odom = g_vesc->getOdometry();

            // Get ERPM for motion state detection
            slam::VescStatus status_l = g_vesc->getStatus(g_config.vesc_left_id);
            slam::VescStatus status_r = g_vesc->getStatus(g_config.vesc_right_id);

            // Calculate linear velocity from wheel velocities
            float linear_vel = (odom.velocity_left_mps + odom.velocity_right_mps) / 2.0f;

            // Get actual angular velocity from motion controller (not the command!)
            // This uses wheel differential for more accurate heading estimation
            float actual_angular = 0.0f;
            if (g_motion) {
                slam::Velocity2D vel = g_motion->getVelocity();
                actual_angular = vel.angular;  // rad/s from wheel differential
            } else {
                actual_angular = target_angular;  // Fallback to command
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

        // 5. Update motor status
        UpdateMotorStatus();

        // 6. Apply gamepad input (if connected and in operational state)
        bool can_drive = (state == AppState::MAPPING || state == AppState::LOCALIZED || state == AppState::OPERATING);
        if (g_gamepad_connected && can_drive) {
            // Use pre-computed drive command from getDriveCommand() (matches manual_drive.cpp)
            target_linear = g_drive_cmd.linear_velocity;
            target_angular = g_drive_cmd.angular_velocity;
        }

        // 7. Send motor commands
        if (g_motion && can_drive) {
            g_motion->setVelocity(target_linear, target_angular);
            g_motion->update(dt);
        }

        // 8. Update LiDAR status
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
        bool can_drive = (state == AppState::MAPPING || state == AppState::LOCALIZED || state == AppState::OPERATING);
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

    // LiDAR status
    LidarStatus lidar = g_shared.getLidarStatus();
    ImVec4 lidar_color = (lidar.connection == ConnectionStatus::CONNECTED)
        ? ImVec4(0.2f, 1.0f, 0.2f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
    ImGui::TextColored(lidar_color, "LiDAR");
    SetTooltip("LiDAR connection status. Green = connected.");
    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();

    // VESC status
    MotorStatus motor_l = g_shared.getMotorStatus(0);
    MotorStatus motor_r = g_shared.getMotorStatus(1);
    bool vesc_ok = motor_l.connected || motor_r.connected;
    ImVec4 vesc_color = vesc_ok ? ImVec4(0.2f, 1.0f, 0.2f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
    ImGui::TextColored(vesc_color, "VESC");
    SetTooltip("Motor controller connection. Green = connected.");
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

    // Gamepad status
    ImVec4 gp_color = g_gamepad_connected ? ImVec4(0.2f, 1.0f, 0.2f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
    ImGui::TextColored(gp_color, "Gamepad");
    SetTooltip("Xbox controller connection. Use for manual robot control.");
    ImGui::SameLine();
    ImGui::Text("|");
    ImGui::SameLine();

    // Application state
    AppState state = g_shared.getAppState();
    float r, g, b, a;
    getStateColor(state, r, g, b, a);
    ImGui::TextColored(ImVec4(r, g, b, a), "%s", getStateName(state));
    SetTooltip("Current application mode.");

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
    static std::vector<RenderPoint> cached_map_points;
    static std::vector<RenderPoint> cached_scan_points;
    g_shared.getMapPointsIfUpdated(cached_map_points);
    g_shared.getCurrentScanIfUpdated(cached_scan_points);

    // Draw map points (small dots, colored by height)
    // Use 3D projection for tilted views (actual Z coordinate)
    for (const auto& pt : cached_map_points) {
        ImVec2 screen_pos = worldToScreen3D(pt.x, pt.y, pt.z);
        // Skip points outside view
        if (screen_pos.x < canvas_pos.x - 5 || screen_pos.x > canvas_pos.x + canvas_size.x + 5 ||
            screen_pos.y < canvas_pos.y - 5 || screen_pos.y > canvas_pos.y + canvas_size.y + 5) continue;
        // Vary point size slightly based on Z for depth cue
        float z_factor = 1.0f + (pt.z * 0.05f);  // Higher points slightly larger
        draw_list->AddCircleFilled(screen_pos, 1.5f * z_factor, IM_COL32(pt.r, pt.g, pt.b, 180));
    }

    // Draw current scan points (brighter, larger)
    for (const auto& pt : cached_scan_points) {
        ImVec2 screen_pos = worldToScreen3D(pt.x, pt.y, pt.z);
        if (screen_pos.x < canvas_pos.x - 5 || screen_pos.x > canvas_pos.x + canvas_size.x + 5 ||
            screen_pos.y < canvas_pos.y - 5 || screen_pos.y > canvas_pos.y + canvas_size.y + 5) continue;
        draw_list->AddCircleFilled(screen_pos, 2.5f, IM_COL32(200, 255, 255, 255));
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
        bool can_operate = (state == AppState::IDLE);
        ImGui::BeginDisabled(!can_operate);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.8f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.9f, 1.0f));
        if (ImGui::Button("Start Operating", ImVec2(200, 40))) {
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

    // Left side - Map viewer
    ImGui::BeginChild("MappingViewer", ImVec2(-panel_width - 10, 0), true);
    ImGui::Text("Live Map View");
    ImGui::Separator();
    DrawMapViewer(true, true, true);
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
        if (state == AppState::IDLE) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
            if (ImGui::Button("Start Mapping", ImVec2(-1, 50))) {
                g_commands.push(Command::simple(CommandType::START_MAPPING));
            }
            SetTooltip("Begin building a new map.");
            ImGui::PopStyleColor(2);
        } else {
            ImGui::BeginDisabled();
            ImGui::Button("Start Mapping", ImVec2(-1, 50));
            ImGui::EndDisabled();
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.2f, 1.0f),
                "Stop current op first");
        }
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

    // Left side - Map viewer
    ImGui::BeginChild("LocalizationViewer", ImVec2(-panel_width - 10, 0), true);
    ImGui::Text("Map View");
    ImGui::Separator();
    DrawMapViewer(true, true, true);
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
    ImGui::Checkbox("Use pose hint", &use_pose_hint);
    ImGui::SameLine();
    HelpMarker("Provide approximate position to help.");

    if (use_pose_hint) {
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##HintX", &hint_x, 0.1f, -100.0f, 100.0f, "X: %.1f m");
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##HintY", &hint_y, 0.1f, -100.0f, 100.0f, "Y: %.1f m");
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##HintYaw", &hint_yaw, 1.0f, -180.0f, 180.0f, "Yaw: %.0f deg");
    }

    bool can_relocalize = (state == AppState::MAP_LOADED || state == AppState::RELOCALIZE_FAILED);
    ImGui::BeginDisabled(!can_relocalize);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.7f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 0.8f, 1.0f));
    if (ImGui::Button("Relocalize", ImVec2(-1, 40))) {
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
        ImGui::ProgressBar(reloc.progress, ImVec2(-1, 20));
        ImGui::Text("%s", reloc.status_text.c_str());
    } else if (state == AppState::LOCALIZED) {
        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f),
            "LOCALIZED (%.0f%%)", reloc.confidence * 100.0f);

        // Show current pose
        Pose3D pose = g_shared.getFusedPose();
        ImGui::Text("Position: (%.2f, %.2f)", pose.x, pose.y);
        ImGui::Text("Heading: %.1f deg", pose.yaw * 180.0f / 3.14159f);
    } else if (state == AppState::RELOCALIZE_FAILED) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.2f, 1.0f),
            "FAILED - try different hint");
    } else if (state == AppState::MAP_LOADED) {
        ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f),
            "Map loaded - ready");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Stop localization
    bool can_stop = (state == AppState::LOCALIZED);
    ImGui::BeginDisabled(!can_stop);
    if (ImGui::Button("Stop Localization", ImVec2(-1, 30))) {
        g_commands.push(Command::simple(CommandType::STOP_LOCALIZATION));
    }
    SetTooltip("Stop localization and return to idle.");
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
        ImGui::SliderFloat("Voxel Size", &g_config.voxel_size, 0.05f, 0.5f, "%.2f m");
        SetTooltip("Voxel filter size for map points. Larger = sparser map.");

        ImGui::SliderFloat("Blind Distance", &g_config.blind_distance, 0.1f, 2.0f, "%.2f m");
        SetTooltip("Ignore LiDAR returns closer than this.");

        ImGui::SliderInt("Max Iterations", &g_config.max_iterations, 1, 6);
        SetTooltip("IEKF iterations per scan. More = accurate but slower.");

        ImGui::SliderFloat("Gyro Covariance", &g_config.gyr_cov, 0.01f, 1.0f, "%.2f");
        SetTooltip("IMU gyroscope noise covariance.");

        ImGui::SliderInt("Point Filter", &g_config.point_filter, 1, 5);
        SetTooltip("Keep every Nth valid point. 1=all, 3=keep 1/3 (reduces noise).");

        ImGui::Checkbox("Deskew Enabled", &g_config.deskew_enabled);
        SetTooltip("Enable motion compensation for LiDAR scans.");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Save/Load settings
    if (ImGui::Button("Save Settings", ImVec2(120, 30))) {
        // TODO: Save to INI file
        g_shared.setStatusMessage("Settings saved");
    }
    SetTooltip("Save all settings to configuration file.");

    ImGui::SameLine();
    if (ImGui::Button("Load Defaults", ImVec2(120, 30))) {
        g_config = AppConfig();  // Reset to defaults
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

    ImGui::End();
}

//==============================================================================
// Main Entry Point
//==============================================================================
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    // Get executable directory
    char exePath[MAX_PATH];
    GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    strncpy(g_config.map_directory, exeDir.c_str(), sizeof(g_config.map_directory) - 1);

    // Create window
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L,
                       GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr,
                       L"SLAM Control GUI", nullptr };
    RegisterClassExW(&wc);

    HWND hwnd = CreateWindowW(wc.lpszClassName, L"SLAM Control GUI",
                              WS_OVERLAPPEDWINDOW, 50, 50, 1400, 900,
                              nullptr, nullptr, wc.hInstance, nullptr);

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

    // Start worker thread
    g_worker_thread = std::thread(WorkerThread);

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
        if (!io.WantCaptureKeyboard) {  // Don't capture if typing in ImGui
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

        // Send velocity command (gamepad has priority, keyboard as fallback)
        if (!g_e_stop.load()) {
            AppState state = g_shared.getAppState();
            bool can_drive = (state == AppState::MAPPING || state == AppState::LOCALIZED || state == AppState::OPERATING);
            if (can_drive) {
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

                g_commands.push(Command::velocity(linear, angular));
            }
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
