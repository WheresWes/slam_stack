/**
 * @file visualization_interface.hpp
 * @brief SLAM visualization interface (Rerun-free header for fast compilation)
 *
 * This header provides the interface to the Visualizer without including
 * Rerun headers, enabling fast incremental builds.
 */

#ifndef SLAM_VISUALIZATION_INTERFACE_HPP
#define SLAM_VISUALIZATION_INTERFACE_HPP

#include <string>
#include <vector>
#include <memory>
#include <array>

#include "slam/types.hpp"

namespace slam {

//=============================================================================
// Visualization Configuration
//=============================================================================

struct VisualizerConfig {
    // Recording settings
    std::string application_id = "slam_visualizer";
    std::string recording_id = "";
    bool spawn_viewer = true;
    std::string connect_addr = "";

    // Point cloud settings
    float point_size = 2.0f;
    bool color_by_intensity = true;
    bool color_by_height = false;
    float intensity_scale = 1.0f;

    // Map settings
    float map_point_size = 1.5f;
    bool show_map_points = true;
    int map_update_interval_ms = 500;

    // Trajectory settings
    bool show_trajectory = true;
    float trajectory_line_width = 2.0f;

    // Mesh settings
    bool show_reference_mesh = true;
    float mesh_opacity = 0.5f;

    // Performance
    int max_points_per_frame = 50000;
    bool enable_logging = true;
};

//=============================================================================
// Color Utilities
//=============================================================================

struct Color3 {
    uint8_t r, g, b;
    Color3(uint8_t r_ = 255, uint8_t g_ = 255, uint8_t b_ = 255) : r(r_), g(g_), b(b_) {}
};

namespace colors {
    const Color3 RED(255, 0, 0);
    const Color3 GREEN(0, 255, 0);
    const Color3 BLUE(0, 0, 255);
    const Color3 YELLOW(255, 255, 0);
    const Color3 CYAN(0, 255, 255);
    const Color3 MAGENTA(255, 0, 255);
    const Color3 WHITE(255, 255, 255);
    const Color3 GRAY(128, 128, 128);
    const Color3 ORANGE(255, 165, 0);
}

// Height-based coloring (rainbow)
inline Color3 heightToColor(float z, float z_min, float z_max) {
    float t = (z - z_min) / (z_max - z_min + 1e-6f);
    t = std::max(0.0f, std::min(1.0f, t));

    float r, g, b;
    if (t < 0.25f) {
        r = 1.0f; g = t * 4.0f; b = 0.0f;
    } else if (t < 0.5f) {
        r = 1.0f - (t - 0.25f) * 4.0f; g = 1.0f; b = 0.0f;
    } else if (t < 0.75f) {
        r = 0.0f; g = 1.0f; b = (t - 0.5f) * 4.0f;
    } else {
        r = 0.0f; g = 1.0f - (t - 0.75f) * 4.0f; b = 1.0f;
    }

    return Color3(
        static_cast<uint8_t>(r * 255),
        static_cast<uint8_t>(g * 255),
        static_cast<uint8_t>(b * 255)
    );
}

//=============================================================================
// Visualizer Class (PIMPL pattern - implementation hidden)
//=============================================================================

class Visualizer {
public:
    Visualizer();
    explicit Visualizer(const VisualizerConfig& config);
    explicit Visualizer(const std::string& app_id);
    ~Visualizer();

    // Non-copyable, movable
    Visualizer(const Visualizer&) = delete;
    Visualizer& operator=(const Visualizer&) = delete;
    Visualizer(Visualizer&&) noexcept;
    Visualizer& operator=(Visualizer&&) noexcept;

    // Initialization
    bool init(const VisualizerConfig& config);
    bool init(const std::string& app_id);
    bool isInitialized() const;

    // Time management
    void setTime(const std::string& timeline, double time_sec);
    void setTime(const std::string& timeline, int64_t time_ns);

    // Point clouds
    void logPointCloud(const std::string& entity_path,
                       const std::vector<V3D>& points,
                       const Color3& color = colors::WHITE);

    void logPointCloud(const std::string& entity_path,
                       const std::vector<V3D>& points,
                       const std::vector<float>& intensities);

    void logPointCloud(const std::string& entity_path,
                       const PointCloud& cloud);

    // Poses and transforms
    void logPose(const std::string& entity_path,
                 const V3D& position,
                 const M3D& rotation);

    void logPose(const std::string& entity_path,
                 const SlamState& state);

    // Trajectory
    void logTrajectory(const std::string& entity_path,
                       const std::vector<V3D>& positions,
                       const Color3& color = colors::CYAN);

    // Meshes
    void logMesh(const std::string& entity_path,
                 const std::vector<V3D>& vertices,
                 const std::vector<std::array<uint32_t, 3>>& triangles,
                 const Color3& color = colors::GRAY);

    void logMeshFromPLY(const std::string& entity_path,
                        const std::string& ply_path);

    // IMU data
    void logIMU(const std::string& entity_path,
                const ImuData& imu);

    // Text annotations
    void logText(const std::string& entity_path,
                 const std::string& text);

    // Coordinate frames
    void logCoordinateFrame(const std::string& entity_path,
                            float scale = 1.0f);

    // Statistics
    struct Stats {
        uint64_t points_logged = 0;
        uint64_t poses_logged = 0;
        uint64_t frames_logged = 0;
    };
    Stats getStats() const;

private:
    // PIMPL - hides Rerun types from header
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace slam

#endif // SLAM_VISUALIZATION_INTERFACE_HPP
