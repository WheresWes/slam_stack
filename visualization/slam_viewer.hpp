/**
 * @file slam_viewer.hpp
 * @brief High-performance 3D visualization for SLAM and hull inspection
 *
 * Features:
 * - GPU-accelerated point cloud rendering (D3D11)
 * - Thread-safe double-buffered updates
 * - Mesh rendering for hull visualization
 * - PAUT probe coverage painting
 * - Embeddable ImGui widget
 *
 * Optimizations:
 * - Point cloud rendering via GPU structured buffers
 * - Lock-free double buffering for SLAM thread updates
 * - Frustum culling and LOD
 * - Batched coverage cell rendering
 */

#pragma once

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
#endif

#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace slam {
namespace viz {

// Forward declarations
class SlamViewerImpl;

// Type aliases
using V3D = Eigen::Vector3d;
using V3F = Eigen::Vector3f;
using V4F = Eigen::Vector4f;
using M3D = Eigen::Matrix3d;
using M4D = Eigen::Matrix4d;
using M4F = Eigen::Matrix4f;

//==============================================================================
// Configuration Structures
//==============================================================================

enum class Colormap {
    GRAYSCALE,      // Black to white
    VIRIDIS,        // Purple → Blue → Green → Yellow
    TURBO,          // Rainbow (good for intensity)
    PLASMA,         // Purple → Pink → Orange → Yellow
    INFERNO,        // Black → Purple → Red → Yellow
    HEIGHT,         // Color by Z coordinate
    CUSTOM          // User-provided colormap
};

enum class ViewMode {
    SCANNING,       // Show point cloud map being built
    LOCALIZATION    // Show hull mesh + probe position + coverage
};

struct PointCloudConfig {
    float point_size = 2.0f;            // Point size in pixels
    Colormap colormap = Colormap::TURBO;
    float intensity_min = 0.0f;         // Intensity range for colormap
    float intensity_max = 255.0f;
    bool enable_lod = true;             // Level-of-detail for distant points
    float lod_distance = 10.0f;         // Distance for LOD transition
    size_t max_visible_points = 5000000; // Cap for performance
};

struct MeshConfig {
    V4F color = V4F(0.7f, 0.7f, 0.7f, 0.8f);  // RGBA, slightly transparent
    bool wireframe = false;
    bool show_normals = false;
    float normal_length = 0.05f;
};

struct CoverageConfig {
    V4F covered_color = V4F(0.0f, 0.8f, 0.2f, 0.6f);    // Green, semi-transparent
    V4F uncovered_color = V4F(0.5f, 0.5f, 0.5f, 0.0f);  // Invisible
    float cell_size = 0.02f;                              // 20mm cells
    bool show_probe_footprint = true;
    V4F probe_color = V4F(1.0f, 0.5f, 0.0f, 0.8f);      // Orange
};

struct ProbeConfig {
    double scan_width = 0.186;          // 186mm PAUT probe width
    V3D lidar_to_probe_offset = V3D(-0.3, 0, -0.1);  // 300mm behind, 100mm below
    V3D probe_scan_axis = V3D(0, 1, 0); // Width along Y axis (widthways)
};

struct ViewerConfig {
    PointCloudConfig point_cloud;
    MeshConfig mesh;
    CoverageConfig coverage;
    ProbeConfig probe;

    V4F background_color = V4F(0.1f, 0.1f, 0.12f, 1.0f);  // Dark gray
    float camera_fov = 60.0f;
    float camera_near = 0.01f;
    float camera_far = 1000.0f;
};

//==============================================================================
// Data Structures for Thread-Safe Updates
//==============================================================================

struct PointData {
    float x, y, z;
    uint8_t intensity;
    uint8_t padding[3];  // Align to 16 bytes for GPU
};

struct PoseData {
    M4F transform;
    uint64_t timestamp_ns;
};

//==============================================================================
// Main Viewer Class
//==============================================================================

class SlamViewer {
public:
    SlamViewer();
    ~SlamViewer();

    // Disable copy, allow move
    SlamViewer(const SlamViewer&) = delete;
    SlamViewer& operator=(const SlamViewer&) = delete;
    SlamViewer(SlamViewer&&) noexcept;
    SlamViewer& operator=(SlamViewer&&) noexcept;

    //==========================================================================
    // Initialization
    //==========================================================================

    /**
     * Initialize the viewer with an existing D3D11 device (for embedding)
     * @param device ID3D11Device pointer
     * @param context ID3D11DeviceContext pointer
     * @return true on success
     */
    bool initWithDevice(void* device, void* context);

    /**
     * Initialize with standalone window (for testing)
     * @param width Window width
     * @param height Window height
     * @param title Window title
     * @return true on success
     */
    bool initStandalone(int width, int height, const char* title = "SLAM Viewer");

    /**
     * Set configuration
     */
    void setConfig(const ViewerConfig& config);
    const ViewerConfig& getConfig() const;

    //==========================================================================
    // Mode Control
    //==========================================================================

    void setMode(ViewMode mode);
    ViewMode getMode() const;

    //==========================================================================
    // Data Updates (Thread-Safe - can be called from SLAM thread)
    //==========================================================================

    /**
     * Update entire point cloud map (double-buffered, lock-free)
     * @param points Array of point data
     * @param count Number of points
     */
    void updatePointCloud(const PointData* points, size_t count);

    /**
     * Append new scan to existing map
     * @param points Array of new points
     * @param count Number of points
     */
    void appendPointCloud(const PointData* points, size_t count);

    /**
     * Update overlay point cloud (for showing local map during localization)
     * Rendered with a different color tint to distinguish from main cloud
     * @param points Array of point data
     * @param count Number of points
     */
    void updateOverlayPointCloud(const PointData* points, size_t count);

    /**
     * Set overlay point cloud color tint (default: green)
     * @param r, g, b Color components (0-1)
     */
    void setOverlayColorTint(float r, float g, float b);

    /**
     * Clear overlay point cloud
     */
    void clearOverlayPointCloud();

    /**
     * Set robot position for visualization (draws a triangle marker)
     * @param x, y Robot position in world frame
     * @param heading Robot heading in radians
     */
    void setRobotPose(float x, float y, float heading);

    /**
     * Update LiDAR pose (triggers coverage painting in localization mode)
     * @param pose 4x4 transform matrix (column-major)
     * @param timestamp_ns Timestamp in nanoseconds
     */
    void updatePose(const M4D& pose, uint64_t timestamp_ns);

    /**
     * Clear all point cloud data
     */
    void clearPointCloud();

    //==========================================================================
    // Hull Mesh (Localization Mode)
    //==========================================================================

    /**
     * Load hull mesh from file
     * @param filename OBJ or STL file
     * @return true on success
     */
    bool loadHullMesh(const std::string& filename);

    /**
     * Clear hull mesh
     */
    void clearHullMesh();

    //==========================================================================
    // Coverage Tracking (Localization Mode)
    //==========================================================================

    /**
     * Initialize coverage grid on hull mesh
     * Must be called after loadHullMesh()
     */
    void initCoverageGrid();

    /**
     * Clear all coverage data
     */
    void clearCoverage();

    /**
     * Get coverage percentage
     */
    float getCoveragePercent() const;

    //==========================================================================
    // Rendering
    //==========================================================================

    /**
     * Render the viewer as an ImGui widget
     * Call this within ImGui::Begin()/End() block
     * @param width Widget width (0 = auto)
     * @param height Widget height (0 = auto)
     */
    void renderWidget(float width = 0, float height = 0);

    /**
     * Get the rendered texture as ImTextureID for ImGui::Image()
     * Only valid after renderWidget() has been called
     * @return ImTextureID (actually ID3D11ShaderResourceView*)
     */
    void* getImGuiTexture() const;

    /**
     * Render to standalone window (if initialized with initStandalone)
     * @return false if window was closed
     */
    bool renderStandalone();

    /**
     * Process window messages (standalone mode only)
     */
    bool processMessages();

    //==========================================================================
    // Click-to-Set-Pose (for localization initial guess)
    //==========================================================================

    /**
     * Callback type for map click events
     * @param world_x, world_y World coordinates of click
     * @param heading Heading angle (radians) if right-drag was used, or NaN if just a click
     */
    using MapClickCallback = std::function<void(float world_x, float world_y, float heading)>;

    /**
     * Set callback for map click events
     * When user clicks on the map, the callback receives world coordinates
     * Right-drag allows setting heading direction
     */
    void setMapClickCallback(MapClickCallback callback);

    /**
     * Enable/disable click-to-pose mode
     * When enabled, clicking on the map triggers the callback instead of camera control
     */
    void setClickToPoseMode(bool enabled);
    bool isClickToPoseMode() const;

    //==========================================================================
    // Camera Control
    //==========================================================================

    void setCameraTarget(const V3F& target);
    void setCameraDistance(float distance);
    void setCameraRotation(float yaw, float pitch);  // Radians
    void resetCamera();

    // Auto-fit camera to current point cloud bounds
    void fitCameraToContent();

    //==========================================================================
    // Utility
    //==========================================================================

    /**
     * Take screenshot
     * @param filename PNG file path
     * @return true on success
     */
    bool saveScreenshot(const std::string& filename);

    /**
     * Get render statistics
     */
    struct RenderStats {
        size_t visible_points;
        size_t total_points;
        size_t covered_cells;
        size_t total_cells;
        float frame_time_ms;
        float gpu_time_ms;
    };
    RenderStats getStats() const;

private:
    std::unique_ptr<SlamViewerImpl> impl_;
};

} // namespace viz
} // namespace slam
