/**
 * @file visualization.cpp
 * @brief SLAM visualization implementation using Rerun
 *
 * This file contains all Rerun-dependent code, isolated from the rest of the
 * codebase for faster incremental builds.
 */

#include "slam/visualization_interface.hpp"

#include <iostream>
#include <mutex>
#include <algorithm>
#include <random>
#include <limits>
#include <set>

#ifdef HAS_RERUN
#include <rerun.hpp>
#endif

namespace slam {

//=============================================================================
// Visualizer Implementation
//=============================================================================

class Visualizer::Impl {
public:
    bool initialized = false;
    VisualizerConfig config;
    Stats stats;
    mutable std::mutex mutex;

#ifdef HAS_RERUN
    std::unique_ptr<rerun::RecordingStream> rec;
#endif

    std::vector<size_t> downsampleIndices(size_t total, size_t max_count) {
        std::vector<size_t> indices;
        if (total <= max_count) {
            indices.resize(total);
            for (size_t i = 0; i < total; i++) indices[i] = i;
        } else {
            indices.reserve(max_count);
            std::mt19937 gen(42);
            std::uniform_int_distribution<size_t> dis(0, total - 1);
            std::set<size_t> selected;
            while (selected.size() < max_count) {
                selected.insert(dis(gen));
            }
            indices.assign(selected.begin(), selected.end());
        }
        return indices;
    }
};

//=============================================================================
// Constructor / Destructor
//=============================================================================

Visualizer::Visualizer() : impl_(std::make_unique<Impl>()) {}

Visualizer::Visualizer(const VisualizerConfig& config) : impl_(std::make_unique<Impl>()) {
    init(config);
}

Visualizer::Visualizer(const std::string& app_id) : impl_(std::make_unique<Impl>()) {
    VisualizerConfig config;
    config.application_id = app_id;
    init(config);
}

Visualizer::~Visualizer() = default;

Visualizer::Visualizer(Visualizer&&) noexcept = default;
Visualizer& Visualizer::operator=(Visualizer&&) noexcept = default;

//=============================================================================
// Initialization
//=============================================================================

bool Visualizer::init(const VisualizerConfig& config) {
#ifdef HAS_RERUN
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (impl_->initialized) return true;

    impl_->config = config;

    try {
        impl_->rec = std::make_unique<rerun::RecordingStream>(config.application_id);

        bool connected = false;

        if (!config.connect_addr.empty()) {
            auto result = impl_->rec->connect_grpc(config.connect_addr);
            if (result.is_ok()) {
                std::cout << "[Visualizer] Connected to existing viewer at " << config.connect_addr << std::endl;
                connected = true;
            }
        }

        if (!connected && config.spawn_viewer) {
            std::cout << "[Visualizer] Spawning new viewer..." << std::endl;
            auto spawn_result = impl_->rec->spawn();
            spawn_result.exit_on_failure();

            // Give viewer time to start up before sending data
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            std::cout << "[Visualizer] Viewer spawned and ready" << std::endl;
        }

        // Set coordinate system: SLAM uses FLU (Forward-Left-Up)
        // X = forward, Y = left, Z = up
        impl_->rec->log_static("/", rerun::ViewCoordinates::FLU);

        impl_->initialized = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[Visualizer] Failed to initialize: " << e.what() << std::endl;
        return false;
    }
#else
    std::cerr << "[Visualizer] Rerun not available (HAS_RERUN not defined)" << std::endl;
    return false;
#endif
}

bool Visualizer::init(const std::string& app_id) {
    VisualizerConfig config;
    config.application_id = app_id;
    return init(config);
}

bool Visualizer::isInitialized() const {
    return impl_->initialized;
}

//=============================================================================
// Time Management
//=============================================================================

void Visualizer::setTime(const std::string& timeline, double time_sec) {
#ifdef HAS_RERUN
    if (!impl_->initialized) return;
    impl_->rec->set_time_seconds(timeline, time_sec);
#endif
}

void Visualizer::setTime(const std::string& timeline, int64_t time_ns) {
#ifdef HAS_RERUN
    if (!impl_->initialized) return;
    impl_->rec->set_time_nanos(timeline, time_ns);
#endif
}

//=============================================================================
// Point Clouds
//=============================================================================

void Visualizer::logPointCloud(const std::string& entity_path,
                                const std::vector<V3D>& points,
                                const Color3& color) {
#ifdef HAS_RERUN
    if (!impl_->initialized || points.empty()) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);

    std::vector<std::array<float, 3>> positions;

    if (points.size() > static_cast<size_t>(impl_->config.max_points_per_frame)) {
        auto indices = impl_->downsampleIndices(points.size(), impl_->config.max_points_per_frame);
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

    uint32_t rgba = (static_cast<uint32_t>(color.r) << 24) |
                    (static_cast<uint32_t>(color.g) << 16) |
                    (static_cast<uint32_t>(color.b) << 8) |
                    0xFF;

    impl_->rec->log(entity_path,
                    rerun::Points3D(positions)
                        .with_colors(rgba)
                        .with_radii(impl_->config.point_size * 0.001f));

    impl_->stats.points_logged += positions.size();
#endif
}

void Visualizer::logPointCloud(const std::string& entity_path,
                                const std::vector<V3D>& points,
                                const std::vector<float>& intensities) {
#ifdef HAS_RERUN
    if (!impl_->initialized || points.empty()) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);

    std::vector<std::array<float, 3>> positions;
    std::vector<uint32_t> colors;

    std::vector<size_t> indices;
    if (points.size() > static_cast<size_t>(impl_->config.max_points_per_frame)) {
        indices = impl_->downsampleIndices(points.size(), impl_->config.max_points_per_frame);
    } else {
        indices.resize(points.size());
        for (size_t i = 0; i < points.size(); i++) indices[i] = i;
    }

    positions.reserve(indices.size());
    colors.reserve(indices.size());

    float z_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::lowest();
    if (impl_->config.color_by_height) {
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
        if (impl_->config.color_by_height) {
            c = heightToColor(static_cast<float>(points[i].z()), z_min, z_max);
        } else if (impl_->config.color_by_intensity && i < intensities.size()) {
            float v = std::min(255.0f, intensities[i] * impl_->config.intensity_scale);
            uint8_t gray = static_cast<uint8_t>(v);
            c = Color3(gray, gray, gray);
        } else {
            c = colors::WHITE;
        }

        uint32_t rgba = (static_cast<uint32_t>(c.r) << 24) |
                        (static_cast<uint32_t>(c.g) << 16) |
                        (static_cast<uint32_t>(c.b) << 8) |
                        0xFF;
        colors.push_back(rgba);
    }

    impl_->rec->log(entity_path,
                    rerun::Points3D(positions)
                        .with_colors(colors)
                        .with_radii(impl_->config.point_size * 0.001f));

    impl_->stats.points_logged += positions.size();
#endif
}

void Visualizer::logPointCloud(const std::string& entity_path,
                                const PointCloud& cloud) {
#ifdef HAS_RERUN
    if (!impl_->initialized || cloud.empty()) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);

    std::vector<std::array<float, 3>> positions;
    std::vector<uint32_t> colors;

    std::vector<size_t> indices;
    if (cloud.size() > static_cast<size_t>(impl_->config.max_points_per_frame)) {
        indices = impl_->downsampleIndices(cloud.size(), impl_->config.max_points_per_frame);
    } else {
        indices.resize(cloud.size());
        for (size_t i = 0; i < cloud.size(); i++) indices[i] = i;
    }

    positions.reserve(indices.size());
    colors.reserve(indices.size());

    float z_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::lowest();
    if (impl_->config.color_by_height) {
        for (size_t i : indices) {
            z_min = std::min(z_min, cloud.points[i].z);
            z_max = std::max(z_max, cloud.points[i].z);
        }
    }

    for (size_t i : indices) {
        const auto& pt = cloud.points[i];
        positions.push_back({pt.x, pt.y, pt.z});

        Color3 c;
        if (impl_->config.color_by_height) {
            c = heightToColor(pt.z, z_min, z_max);
        } else if (impl_->config.color_by_intensity) {
            float v = std::min(255.0f, pt.intensity * impl_->config.intensity_scale);
            uint8_t gray = static_cast<uint8_t>(v);
            c = Color3(gray, gray, gray);
        } else {
            c = colors::WHITE;
        }

        uint32_t rgba = (static_cast<uint32_t>(c.r) << 24) |
                        (static_cast<uint32_t>(c.g) << 16) |
                        (static_cast<uint32_t>(c.b) << 8) |
                        0xFF;
        colors.push_back(rgba);
    }

    impl_->rec->log(entity_path,
                    rerun::Points3D(positions)
                        .with_colors(colors)
                        .with_radii(impl_->config.point_size * 0.001f));

    impl_->stats.points_logged += positions.size();
#endif
}

//=============================================================================
// Poses
//=============================================================================

void Visualizer::logPose(const std::string& entity_path,
                          const V3D& position,
                          const M3D& rotation) {
#ifdef HAS_RERUN
    if (!impl_->initialized) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);

    // Convert rotation matrix to quaternion
    Eigen::Quaterniond q(rotation);

    impl_->rec->log(entity_path,
                    rerun::Transform3D::from_translation_rotation(
                        {static_cast<float>(position.x()),
                         static_cast<float>(position.y()),
                         static_cast<float>(position.z())},
                        rerun::Quaternion::from_xyzw(
                            static_cast<float>(q.x()),
                            static_cast<float>(q.y()),
                            static_cast<float>(q.z()),
                            static_cast<float>(q.w()))));

    impl_->stats.poses_logged++;
#endif
}

void Visualizer::logPose(const std::string& entity_path,
                          const SlamState& state) {
    logPose(entity_path, state.pos, state.rot);
}

//=============================================================================
// Trajectory
//=============================================================================

void Visualizer::logTrajectory(const std::string& entity_path,
                                const std::vector<V3D>& positions,
                                const Color3& color) {
#ifdef HAS_RERUN
    if (!impl_->initialized || positions.empty()) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);

    std::vector<std::array<float, 3>> pts;
    pts.reserve(positions.size());
    for (const auto& p : positions) {
        pts.push_back({
            static_cast<float>(p.x()),
            static_cast<float>(p.y()),
            static_cast<float>(p.z())
        });
    }

    uint32_t rgba = (static_cast<uint32_t>(color.r) << 24) |
                    (static_cast<uint32_t>(color.g) << 16) |
                    (static_cast<uint32_t>(color.b) << 8) |
                    0xFF;

    impl_->rec->log(entity_path,
                    rerun::LineStrips3D(std::vector<std::vector<std::array<float, 3>>>{pts})
                        .with_colors(rgba)
                        .with_radii(impl_->config.trajectory_line_width * 0.001f));
#endif
}

//=============================================================================
// Meshes
//=============================================================================

void Visualizer::logMesh(const std::string& entity_path,
                          const std::vector<V3D>& vertices,
                          const std::vector<std::array<uint32_t, 3>>& triangles,
                          const Color3& color) {
#ifdef HAS_RERUN
    if (!impl_->initialized || vertices.empty()) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);

    std::vector<std::array<float, 3>> verts;
    verts.reserve(vertices.size());
    for (const auto& v : vertices) {
        verts.push_back({
            static_cast<float>(v.x()),
            static_cast<float>(v.y()),
            static_cast<float>(v.z())
        });
    }

    uint32_t rgba = (static_cast<uint32_t>(color.r) << 24) |
                    (static_cast<uint32_t>(color.g) << 16) |
                    (static_cast<uint32_t>(color.b) << 8) |
                    static_cast<uint32_t>(impl_->config.mesh_opacity * 255);

    impl_->rec->log(entity_path,
                    rerun::Mesh3D(verts)
                        .with_triangle_indices(triangles)
                        .with_albedo_factor(rerun::Rgba32(rgba)));
#endif
}

void Visualizer::logMeshFromPLY(const std::string& entity_path,
                                 const std::string& ply_path) {
#ifdef HAS_RERUN
    if (!impl_->initialized) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);

    impl_->rec->log_file_from_path(ply_path, entity_path);
#endif
}

//=============================================================================
// IMU
//=============================================================================

void Visualizer::logIMU(const std::string& entity_path,
                         const ImuData& imu) {
#ifdef HAS_RERUN
    if (!impl_->initialized) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);

    // Log IMU data as text (Scalar was deprecated in newer Rerun versions)
    std::string imu_text = "accel=" + std::to_string(imu.acc.norm()) +
                           " gyro=" + std::to_string(imu.gyro.norm());
    impl_->rec->log(entity_path, rerun::TextLog(imu_text));
#endif
}

//=============================================================================
// Text
//=============================================================================

void Visualizer::logText(const std::string& entity_path,
                          const std::string& text) {
#ifdef HAS_RERUN
    if (!impl_->initialized) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);

    impl_->rec->log(entity_path, rerun::TextLog(text));
#endif
}

//=============================================================================
// Coordinate Frames
//=============================================================================

void Visualizer::logCoordinateFrame(const std::string& entity_path,
                                     float scale) {
#ifdef HAS_RERUN
    if (!impl_->initialized) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);

    // X axis (red)
    impl_->rec->log(entity_path + "/x",
                    rerun::Arrows3D::from_vectors({{scale, 0, 0}})
                        .with_colors(0xFF0000FF));
    // Y axis (green)
    impl_->rec->log(entity_path + "/y",
                    rerun::Arrows3D::from_vectors({{0, scale, 0}})
                        .with_colors(0x00FF00FF));
    // Z axis (blue)
    impl_->rec->log(entity_path + "/z",
                    rerun::Arrows3D::from_vectors({{0, 0, scale}})
                        .with_colors(0x0000FFFF));
#endif
}

//=============================================================================
// Stats
//=============================================================================

Visualizer::Stats Visualizer::getStats() const {
    return impl_->stats;
}

} // namespace slam
