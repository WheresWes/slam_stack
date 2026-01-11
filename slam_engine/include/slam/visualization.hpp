/**
 * @file visualization.hpp
 * @brief Real-time SLAM visualization using Rerun
 *
 * Provides efficient, configurable visualization for:
 * - Point clouds (live scan, map)
 * - Meshes (reference models)
 * - Trajectories
 * - IMU data
 * - Coordinate frames
 *
 * Usage:
 *   slam::Visualizer viz("slam_session");
 *   viz.logPointCloud("scan", points);
 *   viz.logPose("pose", position, rotation);
 */

#ifndef SLAM_VISUALIZATION_HPP
#define SLAM_VISUALIZATION_HPP

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <chrono>
#include <array>

#include "slam/types.hpp"

// Forward declare Rerun types
#ifdef HAS_RERUN
#include <rerun.hpp>
#endif

namespace slam {

//=============================================================================
// Visualization Configuration
//=============================================================================

struct VisualizerConfig {
    // Recording settings
    std::string application_id = "slam_visualizer";
    std::string recording_id = "";  // Auto-generated if empty
    bool spawn_viewer = true;       // Launch viewer automatically
    std::string connect_addr = "";  // Connect to remote viewer (e.g., "127.0.0.1:9876")

    // Point cloud settings
    float point_size = 2.0f;        // Point size in pixels
    bool color_by_intensity = true; // Color points by intensity
    bool color_by_height = false;   // Color points by Z coordinate
    float intensity_scale = 1.0f;   // Scale factor for intensity coloring

    // Map settings
    float map_point_size = 1.5f;
    bool show_map_points = true;
    int map_update_interval_ms = 500;  // How often to update map visualization

    // Trajectory settings
    bool show_trajectory = true;
    float trajectory_line_width = 2.0f;

    // Mesh settings
    bool show_reference_mesh = true;
    float mesh_opacity = 0.5f;

    // Performance
    int max_points_per_frame = 50000;  // Downsample if exceeded
    bool enable_logging = true;
};

//=============================================================================
// Color Utilities
//=============================================================================

struct Color3 {
    uint8_t r, g, b;
    Color3(uint8_t r_ = 255, uint8_t g_ = 255, uint8_t b_ = 255) : r(r_), g(g_), b(b_) {}
};

// Common colors
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

    // Rainbow: red -> yellow -> green -> cyan -> blue
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

// Intensity-based coloring (grayscale with boost)
inline Color3 intensityToColor(float intensity, float scale = 1.0f) {
    float v = std::min(255.0f, intensity * scale);
    uint8_t gray = static_cast<uint8_t>(v);
    return Color3(gray, gray, gray);
}

//=============================================================================
// Visualizer Class
//=============================================================================

class Visualizer {
public:
    Visualizer() = default;
    explicit Visualizer(const VisualizerConfig& config);
    explicit Visualizer(const std::string& app_id);
    ~Visualizer();

    // Initialization
    bool init(const VisualizerConfig& config);
    bool init(const std::string& app_id);
    bool isInitialized() const { return initialized_; }

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

    // IMU data (time series)
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
    Stats getStats() const { return stats_; }

private:
    bool initialized_ = false;
    VisualizerConfig config_;
    Stats stats_;
    mutable std::mutex mutex_;

#ifdef HAS_RERUN
    std::unique_ptr<rerun::RecordingStream> rec_;
#endif

    // Downsampling helper
    std::vector<size_t> downsampleIndices(size_t total, size_t max_count);
};

//=============================================================================
// Implementation
//=============================================================================

#ifdef HAS_RERUN

inline Visualizer::Visualizer(const VisualizerConfig& config) {
    init(config);
}

inline Visualizer::Visualizer(const std::string& app_id) {
    VisualizerConfig config;
    config.application_id = app_id;
    init(config);
}

inline Visualizer::~Visualizer() {
    // RecordingStream destructor handles cleanup
}

inline bool Visualizer::init(const VisualizerConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_) return true;

    config_ = config;

    try {
        // Create recording stream
        rec_ = std::make_unique<rerun::RecordingStream>(config_.application_id);

        bool connected = false;

        // Try connecting to existing viewer first
        if (!config_.connect_addr.empty()) {
            auto result = rec_->connect_grpc(config_.connect_addr);
            if (result.is_ok()) {
                std::cout << "[Visualizer] Connected to existing viewer at " << config_.connect_addr << std::endl;
                connected = true;
            } else {
                std::cout << "[Visualizer] No existing viewer at " << config_.connect_addr << std::endl;
            }
        }

        // Spawn viewer if not connected and spawn_viewer is true
        if (!connected && config_.spawn_viewer) {
            std::cout << "[Visualizer] Spawning new viewer..." << std::endl;
            rec_->spawn().exit_on_failure();
        }

        initialized_ = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[Visualizer] Failed to initialize: " << e.what() << std::endl;
        return false;
    }
}

inline bool Visualizer::init(const std::string& app_id) {
    VisualizerConfig config;
    config.application_id = app_id;
    return init(config);
}

inline void Visualizer::setTime(const std::string& timeline, double time_sec) {
    if (!initialized_) return;
    rec_->set_time_seconds(timeline, time_sec);
}

inline void Visualizer::setTime(const std::string& timeline, int64_t time_ns) {
    if (!initialized_) return;
    rec_->set_time_nanos(timeline, time_ns);
}

inline void Visualizer::logPointCloud(const std::string& entity_path,
                                       const std::vector<V3D>& points,
                                       const Color3& color) {
    if (!initialized_ || points.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);

    // Convert points to array format
    std::vector<std::array<float, 3>> positions;

    // Downsample if needed
    if (points.size() > static_cast<size_t>(config_.max_points_per_frame)) {
        auto indices = downsampleIndices(points.size(), config_.max_points_per_frame);
        positions.reserve(indices.size());
        for (size_t i : indices) {
            positions.push_back({
                static_cast<float>(points[i].x()),
                static_cast<float>(points[i].y()),
                static_cast<float>(points[i].z())
            });
        }
    } else {
        positions.reserve(points.size());
        for (const auto& p : points) {
            positions.push_back({
                static_cast<float>(p.x()),
                static_cast<float>(p.y()),
                static_cast<float>(p.z())
            });
        }
    }

    // Create color as RGBA uint32
    uint32_t rgba = (static_cast<uint32_t>(color.r) << 24) |
                    (static_cast<uint32_t>(color.g) << 16) |
                    (static_cast<uint32_t>(color.b) << 8) |
                    0xFF;

    rec_->log(entity_path,
              rerun::Points3D(positions)
                  .with_colors(rgba)
                  .with_radii(config_.point_size * 0.001f));

    stats_.points_logged += positions.size();
}

inline void Visualizer::logPointCloud(const std::string& entity_path,
                                       const std::vector<V3D>& points,
                                       const std::vector<float>& intensities) {
    if (!initialized_ || points.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::array<float, 3>> positions;
    std::vector<uint32_t> colors;

    // Downsample if needed
    std::vector<size_t> indices;
    if (points.size() > static_cast<size_t>(config_.max_points_per_frame)) {
        indices = downsampleIndices(points.size(), config_.max_points_per_frame);
    } else {
        indices.resize(points.size());
        for (size_t i = 0; i < points.size(); i++) indices[i] = i;
    }

    positions.reserve(indices.size());
    colors.reserve(indices.size());

    // Find Z range for height coloring
    float z_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::lowest();
    if (config_.color_by_height) {
        for (size_t i : indices) {
            float z = static_cast<float>(points[i].z());
            z_min = std::min(z_min, z);
            z_max = std::max(z_max, z);
        }
    }

    for (size_t i : indices) {
        positions.push_back({
            static_cast<float>(points[i].x()),
            static_cast<float>(points[i].y()),
            static_cast<float>(points[i].z())
        });

        Color3 c;
        if (config_.color_by_height) {
            c = heightToColor(static_cast<float>(points[i].z()), z_min, z_max);
        } else if (config_.color_by_intensity && i < intensities.size()) {
            c = intensityToColor(intensities[i], config_.intensity_scale);
        } else {
            c = colors::WHITE;
        }

        // Convert to RGBA uint32
        uint32_t rgba = (static_cast<uint32_t>(c.r) << 24) |
                        (static_cast<uint32_t>(c.g) << 16) |
                        (static_cast<uint32_t>(c.b) << 8) |
                        0xFF;
        colors.push_back(rgba);
    }

    rec_->log(entity_path,
              rerun::Points3D(positions)
                  .with_colors(colors)
                  .with_radii(config_.point_size * 0.001f));

    stats_.points_logged += positions.size();
}

inline void Visualizer::logPointCloud(const std::string& entity_path,
                                       const PointCloud& cloud) {
    std::vector<V3D> points;
    std::vector<float> intensities;

    points.reserve(cloud.size());
    intensities.reserve(cloud.size());

    for (const auto& pt : cloud.points) {
        points.push_back(V3D(pt.x, pt.y, pt.z));
        intensities.push_back(pt.intensity);
    }

    logPointCloud(entity_path, points, intensities);
}

inline void Visualizer::logPose(const std::string& entity_path,
                                 const V3D& position,
                                 const M3D& rotation) {
    if (!initialized_) return;
    std::lock_guard<std::mutex> lock(mutex_);

    // Convert rotation matrix to quaternion
    Eigen::Quaterniond q(rotation);

    rec_->log(entity_path,
              rerun::Transform3D::from_translation_rotation(
                  rerun::Vec3D{
                      static_cast<float>(position.x()),
                      static_cast<float>(position.y()),
                      static_cast<float>(position.z())
                  },
                  rerun::Quaternion::from_xyzw(
                      static_cast<float>(q.x()),
                      static_cast<float>(q.y()),
                      static_cast<float>(q.z()),
                      static_cast<float>(q.w())
                  )
              ));

    stats_.poses_logged++;
}

inline void Visualizer::logPose(const std::string& entity_path,
                                 const SlamState& state) {
    logPose(entity_path, state.pos, state.rot);
}

inline void Visualizer::logTrajectory(const std::string& entity_path,
                                       const std::vector<V3D>& positions,
                                       const Color3& color) {
    if (!initialized_ || positions.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);

    // Convert positions to rerun::Vec3D collection
    std::vector<rerun::Vec3D> line_points;
    line_points.reserve(positions.size());

    for (const auto& p : positions) {
        line_points.push_back(rerun::Vec3D{
            static_cast<float>(p.x()),
            static_cast<float>(p.y()),
            static_cast<float>(p.z())
        });
    }

    // Create a single line strip from the points
    rerun::LineStrip3D strip(line_points);

    rec_->log(entity_path,
              rerun::LineStrips3D(strip)
                  .with_colors(rerun::Color(color.r, color.g, color.b))
                  .with_radii(config_.trajectory_line_width * 0.001f));
}

inline void Visualizer::logMesh(const std::string& entity_path,
                                 const std::vector<V3D>& vertices,
                                 const std::vector<std::array<uint32_t, 3>>& triangles,
                                 const Color3& color) {
    if (!initialized_ || vertices.empty() || triangles.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::array<float, 3>> positions;
    positions.reserve(vertices.size());
    for (const auto& v : vertices) {
        positions.push_back({
            static_cast<float>(v.x()),
            static_cast<float>(v.y()),
            static_cast<float>(v.z())
        });
    }

    // Create color as RGBA uint32
    uint32_t rgba = (static_cast<uint32_t>(color.r) << 24) |
                    (static_cast<uint32_t>(color.g) << 16) |
                    (static_cast<uint32_t>(color.b) << 8) |
                    0xFF;

    rec_->log(entity_path,
              rerun::Mesh3D(positions)
                  .with_triangle_indices(triangles)
                  .with_vertex_colors(rgba));
}

inline void Visualizer::logMeshFromPLY(const std::string& entity_path,
                                        const std::string& ply_path) {
    // TODO: Implement PLY loading
    (void)entity_path;
    (void)ply_path;
}

inline void Visualizer::logIMU(const std::string& entity_path,
                                const ImuData& imu) {
    if (!initialized_) return;
    std::lock_guard<std::mutex> lock(mutex_);

    // Log as time series using Scalars archetype
    rec_->log(entity_path + "/accel/x", rerun::Scalars(static_cast<float>(imu.acc.x())));
    rec_->log(entity_path + "/accel/y", rerun::Scalars(static_cast<float>(imu.acc.y())));
    rec_->log(entity_path + "/accel/z", rerun::Scalars(static_cast<float>(imu.acc.z())));
    rec_->log(entity_path + "/gyro/x", rerun::Scalars(static_cast<float>(imu.gyro.x())));
    rec_->log(entity_path + "/gyro/y", rerun::Scalars(static_cast<float>(imu.gyro.y())));
    rec_->log(entity_path + "/gyro/z", rerun::Scalars(static_cast<float>(imu.gyro.z())));
}

inline void Visualizer::logText(const std::string& entity_path,
                                 const std::string& text) {
    if (!initialized_) return;
    std::lock_guard<std::mutex> lock(mutex_);

    rec_->log(entity_path, rerun::TextLog(text));
}

inline void Visualizer::logCoordinateFrame(const std::string& entity_path,
                                            float scale) {
    if (!initialized_) return;
    std::lock_guard<std::mutex> lock(mutex_);

    // X axis (red)
    rec_->log(entity_path + "/x",
              rerun::Arrows3D::from_vectors({{scale, 0, 0}})
                  .with_colors(0xFF0000FF));
    // Y axis (green)
    rec_->log(entity_path + "/y",
              rerun::Arrows3D::from_vectors({{0, scale, 0}})
                  .with_colors(0x00FF00FF));
    // Z axis (blue)
    rec_->log(entity_path + "/z",
              rerun::Arrows3D::from_vectors({{0, 0, scale}})
                  .with_colors(0x0000FFFF));
}

inline std::vector<size_t> Visualizer::downsampleIndices(size_t total, size_t max_count) {
    std::vector<size_t> indices;
    indices.reserve(max_count);

    float step = static_cast<float>(total) / max_count;
    for (size_t i = 0; i < max_count; i++) {
        indices.push_back(static_cast<size_t>(i * step));
    }

    return indices;
}

#else // !HAS_RERUN

// Stub implementation when Rerun is not available
inline Visualizer::Visualizer(const VisualizerConfig&) {}
inline Visualizer::Visualizer(const std::string&) {}
inline Visualizer::~Visualizer() {}
inline bool Visualizer::init(const VisualizerConfig&) { return false; }
inline bool Visualizer::init(const std::string&) { return false; }
inline void Visualizer::setTime(const std::string&, double) {}
inline void Visualizer::setTime(const std::string&, int64_t) {}
inline void Visualizer::logPointCloud(const std::string&, const std::vector<V3D>&, const Color3&) {}
inline void Visualizer::logPointCloud(const std::string&, const std::vector<V3D>&, const std::vector<float>&) {}
inline void Visualizer::logPointCloud(const std::string&, const PointCloud&) {}
inline void Visualizer::logPose(const std::string&, const V3D&, const M3D&) {}
inline void Visualizer::logPose(const std::string&, const SlamState&) {}
inline void Visualizer::logTrajectory(const std::string&, const std::vector<V3D>&, const Color3&) {}
inline void Visualizer::logMesh(const std::string&, const std::vector<V3D>&, const std::vector<std::array<uint32_t, 3>>&, const Color3&) {}
inline void Visualizer::logMeshFromPLY(const std::string&, const std::string&) {}
inline void Visualizer::logIMU(const std::string&, const ImuData&) {}
inline void Visualizer::logText(const std::string&, const std::string&) {}
inline void Visualizer::logCoordinateFrame(const std::string&, float) {}
inline std::vector<size_t> Visualizer::downsampleIndices(size_t, size_t) { return {}; }

#endif // HAS_RERUN

} // namespace slam

#endif // SLAM_VISUALIZATION_HPP
