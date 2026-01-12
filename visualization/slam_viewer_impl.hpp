/**
 * @file slam_viewer_impl.hpp
 * @brief Implementation details for SlamViewer (D3D11 + multithreading)
 */

#pragma once

#include "slam_viewer.hpp"

#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <queue>
#include <unordered_map>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")

namespace slam {
namespace viz {

using Microsoft::WRL::ComPtr;

//==============================================================================
// Shader Source Code (Embedded)
//==============================================================================

namespace shaders {

// Point cloud vertex shader - instanced rendering using SV_InstanceID
// Each point is rendered as a quad (4 vertices), using structured buffer for point data
constexpr const char* POINT_CLOUD_VS = R"(
cbuffer CameraBuffer : register(b0) {
    matrix viewProj;
    float3 cameraPos;
    float pointSize;
    float2 screenSize;
    float2 padding;
};

struct PointData {
    float3 position;
    float intensity;  // Stored as float for buffer alignment
};

StructuredBuffer<PointData> pointBuffer : register(t1);

struct VSInput {
    float2 corner : POSITION;  // Quad corner from vertex buffer
    uint instanceID : SV_InstanceID;
};

struct VSOutput {
    float4 position : SV_POSITION;
    float intensity : INTENSITY;
    float2 uv : TEXCOORD0;
};

VSOutput main(VSInput input) {
    VSOutput output;

    // Fetch point data from structured buffer
    PointData pt = pointBuffer[input.instanceID];

    // Transform to clip space
    float4 clipPos = mul(viewProj, float4(pt.position, 1.0));

    // Early out for points behind camera
    if (clipPos.w <= 0) {
        output.position = float4(0, 0, -1, 1);
        output.intensity = 0;
        output.uv = float2(0, 0);
        return output;
    }

    // Distance-based point size
    float dist = length(pt.position - cameraPos);
    float size = pointSize * saturate(10.0 / max(dist, 1.0));

    // Offset in clip space (multiply by W for perspective-correct)
    float2 offset = input.corner * size / screenSize * clipPos.w;
    clipPos.xy += offset;

    output.position = clipPos;
    output.intensity = pt.intensity;
    output.uv = input.corner * 0.5 + 0.5;

    return output;
}
)";

// Fallback: Point cloud vertex shader for geometry shader approach
constexpr const char* POINT_CLOUD_VS_GS = R"(
cbuffer CameraBuffer : register(b0) {
    matrix viewProj;
    float3 cameraPos;
    float pointSize;
    float2 screenSize;
    float2 padding;
};

struct VSInput {
    float3 position : POSITION;
    float intensity : TEXCOORD0;
};

struct VSOutput {
    float4 position : SV_POSITION;
    float intensity : INTENSITY;
    float size : PSIZE;
};

VSOutput main(VSInput input) {
    VSOutput output;

    // Transform to clip space
    output.position = mul(viewProj, float4(input.position, 1.0));

    // Pass through intensity
    output.intensity = input.intensity;

    // Calculate point size based on distance
    float dist = length(input.position - cameraPos);
    output.size = pointSize * saturate(10.0 / max(dist, 1.0));

    return output;
}
)";

// Point cloud geometry shader - expands points to screen-aligned quads
constexpr const char* POINT_CLOUD_GS = R"(
cbuffer CameraBuffer : register(b0) {
    matrix viewProj;
    float3 cameraPos;
    float pointSize;
    float2 screenSize;
    float2 padding;
};

struct GSInput {
    float4 position : SV_POSITION;
    float intensity : INTENSITY;
    float size : PSIZE;
};

struct GSOutput {
    float4 position : SV_POSITION;
    float intensity : INTENSITY;
    float2 uv : TEXCOORD0;
};

[maxvertexcount(4)]
void main(point GSInput input[1], inout TriangleStream<GSOutput> stream) {
    // Calculate half size in clip space (multiply by W for perspective-correct offset)
    float2 halfSize = input[0].size / screenSize * input[0].position.w;

    GSOutput output;
    output.intensity = input[0].intensity;

    // Expand point to quad in clip space
    output.position = input[0].position + float4(-halfSize.x, halfSize.y, 0, 0);
    output.uv = float2(0, 0);
    stream.Append(output);

    output.position = input[0].position + float4(halfSize.x, halfSize.y, 0, 0);
    output.uv = float2(1, 0);
    stream.Append(output);

    output.position = input[0].position + float4(-halfSize.x, -halfSize.y, 0, 0);
    output.uv = float2(0, 1);
    stream.Append(output);

    output.position = input[0].position + float4(halfSize.x, -halfSize.y, 0, 0);
    output.uv = float2(1, 1);
    stream.Append(output);
}
)";

// Point cloud pixel shader - applies colormap from lookup texture
constexpr const char* POINT_CLOUD_PS = R"(
Texture1D<float4> colormapTex : register(t0);
SamplerState colormapSampler : register(s0);

struct PSInput {
    float4 position : SV_POSITION;
    float intensity : INTENSITY;
    float2 uv : TEXCOORD0;
};

float4 main(PSInput input) : SV_TARGET {
    // Circular point shape (uv is 0-1)
    float2 centered = input.uv * 2.0 - 1.0;
    float dist = length(centered);
    if (dist > 1.0) discard;

    // Soft edge
    float alpha = 1.0 - smoothstep(0.7, 1.0, dist);

    // Sample colormap (intensity is 0-1 from R8_UNORM)
    float4 color = colormapTex.Sample(colormapSampler, input.intensity);
    color.a *= alpha;

    return color;
}
)";

// Compute shader for GPU frustum culling
constexpr const char* FRUSTUM_CULL_CS = R"(
cbuffer CullParams : register(b0) {
    float4 frustumPlanes[6];  // 6 frustum planes (xyz=normal, w=distance)
    float3 cameraPos;
    float lodDistance;
    uint totalPoints;
    uint maxOutputPoints;
    float2 padding;
};

struct PointData {
    float3 position;
    float intensity;
};

StructuredBuffer<PointData> inputPoints : register(t0);
RWStructuredBuffer<PointData> outputPoints : register(u0);
RWStructuredBuffer<uint> outputCounter : register(u1);  // Single uint for atomic count

// Test if point is inside frustum (all 6 planes)
bool isInsideFrustum(float3 pos) {
    [unroll]
    for (int i = 0; i < 6; i++) {
        float dist = dot(frustumPlanes[i].xyz, pos) + frustumPlanes[i].w;
        if (dist < 0) return false;
    }
    return true;
}

// Compute LOD decimation factor based on distance
uint getLODDecimation(float dist) {
    if (dist < lodDistance) return 1;
    if (dist < lodDistance * 3.0) return 2;
    if (dist < lodDistance * 6.0) return 4;
    return 8;
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint pointIdx = dispatchThreadID.x;
    if (pointIdx >= totalPoints) return;

    PointData pt = inputPoints[pointIdx];

    // Frustum culling
    if (!isInsideFrustum(pt.position)) return;

    // LOD check - only process every Nth point based on distance
    float dist = length(pt.position - cameraPos);
    uint decimation = getLODDecimation(dist);
    if ((pointIdx % decimation) != 0) return;

    // Atomically add to output
    uint outputIdx;
    InterlockedAdd(outputCounter[0], 1, outputIdx);

    if (outputIdx < maxOutputPoints) {
        outputPoints[outputIdx] = pt;
    }
}
)";

// Mesh vertex shader
constexpr const char* MESH_VS = R"(
cbuffer CameraBuffer : register(b0) {
    matrix viewProj;
    matrix world;
    float3 cameraPos;
    float padding;
};

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

struct VSOutput {
    float4 position : SV_POSITION;
    float3 worldPos : WORLDPOS;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

VSOutput main(VSInput input) {
    VSOutput output;
    float4 worldPos = mul(world, float4(input.position, 1.0));
    output.position = mul(viewProj, worldPos);
    output.worldPos = worldPos.xyz;
    output.normal = mul((float3x3)world, input.normal);
    output.uv = input.uv;
    return output;
}
)";

// Mesh pixel shader with coverage overlay
constexpr const char* MESH_PS = R"(
cbuffer MaterialBuffer : register(b1) {
    float4 baseColor;
    float4 coveredColor;
    float4 probeColor;
    float2 coverageTexSize;
    float2 padding;
};

Texture2D<float> coverageTex : register(t0);
SamplerState coverageSampler : register(s0);

struct PSInput {
    float4 position : SV_POSITION;
    float3 worldPos : WORLDPOS;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
};

float4 main(PSInput input) : SV_TARGET {
    // Simple directional lighting
    float3 lightDir = normalize(float3(0.5, 0.5, 1.0));
    float ndotl = max(dot(normalize(input.normal), lightDir), 0.0);
    float ambient = 0.3;
    float diffuse = ndotl * 0.7;

    // Sample coverage
    float coverage = coverageTex.Sample(coverageSampler, input.uv);

    // Blend base color with coverage color
    float4 color = lerp(baseColor, coveredColor, coverage);
    color.rgb *= (ambient + diffuse);

    return color;
}
)";

// Probe visualization vertex shader
constexpr const char* PROBE_VS = R"(
cbuffer CameraBuffer : register(b0) {
    matrix viewProj;
    matrix probeTransform;
    float probeWidth;
    float3 padding;
};

struct VSInput {
    float3 position : POSITION;
};

struct VSOutput {
    float4 position : SV_POSITION;
};

VSOutput main(VSInput input) {
    VSOutput output;
    float4 worldPos = mul(probeTransform, float4(input.position * probeWidth, 1.0));
    output.position = mul(viewProj, worldPos);
    return output;
}
)";

constexpr const char* PROBE_PS = R"(
cbuffer ProbeBuffer : register(b1) {
    float4 probeColor;
};

float4 main() : SV_TARGET {
    return probeColor;
}
)";

} // namespace shaders

//==============================================================================
// GPU Buffer Structures
//==============================================================================

using V2F = Eigen::Vector2f;

struct CameraConstants {
    M4F viewProj;
    V3F cameraPos;
    float pointSize;
    V2F screenSize;
    V2F padding;
};

struct MeshConstants {
    M4F viewProj;
    M4F world;
    V3F cameraPos;
    float padding;
};

struct MaterialConstants {
    V4F baseColor;
    V4F coveredColor;
    V4F probeColor;
    V2F coverageTexSize;
    V2F padding;
};

//==============================================================================
// Frustum Culling
//==============================================================================

class Frustum {
public:
    // Extract 6 frustum planes from view-projection matrix
    void update(const M4F& viewProj) {
        // Row-major extraction: plane = row3 +/- rowN
        // Each plane: Ax + By + Cz + D >= 0 for points inside

        // Left plane: row3 + row0
        planes_[0] = V4F(viewProj(3,0) + viewProj(0,0),
                         viewProj(3,1) + viewProj(0,1),
                         viewProj(3,2) + viewProj(0,2),
                         viewProj(3,3) + viewProj(0,3));

        // Right plane: row3 - row0
        planes_[1] = V4F(viewProj(3,0) - viewProj(0,0),
                         viewProj(3,1) - viewProj(0,1),
                         viewProj(3,2) - viewProj(0,2),
                         viewProj(3,3) - viewProj(0,3));

        // Bottom plane: row3 + row1
        planes_[2] = V4F(viewProj(3,0) + viewProj(1,0),
                         viewProj(3,1) + viewProj(1,1),
                         viewProj(3,2) + viewProj(1,2),
                         viewProj(3,3) + viewProj(1,3));

        // Top plane: row3 - row1
        planes_[3] = V4F(viewProj(3,0) - viewProj(1,0),
                         viewProj(3,1) - viewProj(1,1),
                         viewProj(3,2) - viewProj(1,2),
                         viewProj(3,3) - viewProj(1,3));

        // Near plane: row3 + row2
        planes_[4] = V4F(viewProj(3,0) + viewProj(2,0),
                         viewProj(3,1) + viewProj(2,1),
                         viewProj(3,2) + viewProj(2,2),
                         viewProj(3,3) + viewProj(2,3));

        // Far plane: row3 - row2
        planes_[5] = V4F(viewProj(3,0) - viewProj(2,0),
                         viewProj(3,1) - viewProj(2,1),
                         viewProj(3,2) - viewProj(2,2),
                         viewProj(3,3) - viewProj(2,3));

        // Normalize planes for correct distance calculations
        for (int i = 0; i < 6; i++) {
            float len = V3F(planes_[i].x(), planes_[i].y(), planes_[i].z()).norm();
            if (len > 0.0001f) {
                planes_[i] /= len;
            }
        }
    }

    // Test if AABB intersects frustum (returns true if potentially visible)
    bool testAABB(const V3F& minB, const V3F& maxB) const {
        for (int i = 0; i < 6; i++) {
            // Find the corner closest to the plane (p-vertex)
            V3F p;
            p.x() = (planes_[i].x() >= 0) ? maxB.x() : minB.x();
            p.y() = (planes_[i].y() >= 0) ? maxB.y() : minB.y();
            p.z() = (planes_[i].z() >= 0) ? maxB.z() : minB.z();

            // If p-vertex is outside, entire AABB is outside
            float dist = planes_[i].x() * p.x() + planes_[i].y() * p.y() +
                        planes_[i].z() * p.z() + planes_[i].w();
            if (dist < 0) {
                return false;  // Completely outside this plane
            }
        }
        return true;  // Intersects or inside frustum
    }

private:
    V4F planes_[6];  // left, right, bottom, top, near, far
};

//==============================================================================
// Point Chunk for Spatial Culling
//==============================================================================

struct PointChunk {
    V3F minBound;
    V3F maxBound;
    size_t startIndex;
    size_t count;
};

//==============================================================================
// Double-Buffered Point Cloud Data
//==============================================================================

struct PointBuffer {
    std::vector<PointData> points;
    size_t count = 0;
    std::atomic<bool> ready{false};

    // Chunked data for frustum culling
    std::vector<PointChunk> chunks;
    bool chunksValid = false;
};

//==============================================================================
// Coverage Grid
//==============================================================================

struct CoverageCell {
    V3F center;
    V3F normal;
    V2F uv;         // UV coordinates on mesh
    bool covered = false;
};

class CoverageGrid {
public:
    void initFromMesh(const std::vector<V3F>& vertices,
                      const std::vector<V3F>& normals,
                      const std::vector<uint32_t>& indices,
                      const std::vector<V2F>& uvs,
                      float cellSize);

    void markCovered(const V3F& probeCenter, const V3F& probeAxis, float probeWidth);
    void clear();

    float getCoveragePercent() const;
    const std::vector<CoverageCell>& getCells() const { return cells_; }

    // Get coverage as texture data (for GPU)
    void getCoverageTexture(std::vector<float>& data, int& width, int& height) const;

private:
    std::vector<CoverageCell> cells_;
    float cellSize_ = 0.02f;
    size_t coveredCount_ = 0;

    // Spatial index for fast queries
    struct SpatialHash {
        std::unordered_map<uint64_t, std::vector<size_t>> grid;
        float cellSize = 0.1f;

        uint64_t hash(const V3F& p) const {
            int x = static_cast<int>(std::floor(p.x() / cellSize));
            int y = static_cast<int>(std::floor(p.y() / cellSize));
            int z = static_cast<int>(std::floor(p.z() / cellSize));
            return (static_cast<uint64_t>(x) & 0x1FFFFF) |
                   ((static_cast<uint64_t>(y) & 0x1FFFFF) << 21) |
                   ((static_cast<uint64_t>(z) & 0x1FFFFF) << 42);
        }

        void insert(size_t idx, const V3F& p) {
            grid[hash(p)].push_back(idx);
        }

        std::vector<size_t> query(const V3F& center, float radius) const;
    };
    SpatialHash spatialIndex_;
};

//==============================================================================
// Arcball Camera
//==============================================================================

class ArcballCamera {
public:
    void setTarget(const V3F& target) { target_ = target; }
    void setDistance(float dist) { distance_ = dist; }
    void setRotation(float yaw, float pitch) { yaw_ = yaw; pitch_ = pitch; }

    void rotate(float dyaw, float dpitch);
    void pan(float dx, float dy);
    void zoom(float delta);

    M4F getViewMatrix() const;
    M4F getProjectionMatrix(float aspect, float fov, float near, float far) const;
    V3F getPosition() const;

    void fitToBounds(const V3F& min, const V3F& max);

private:
    V3F target_ = V3F::Zero();
    float distance_ = 5.0f;
    float yaw_ = 0.0f;
    float pitch_ = 0.5f;  // Start looking slightly down
};

//==============================================================================
// Main Implementation Class
//==============================================================================

class SlamViewerImpl {
public:
    SlamViewerImpl();
    ~SlamViewerImpl();

    // Initialization
    bool initWithDevice(void* device, void* context);
    bool initStandalone(int width, int height, const char* title);
    void shutdown();

    // Configuration
    void setConfig(const ViewerConfig& config);
    const ViewerConfig& getConfig() const { return config_; }

    // Mode
    void setMode(ViewMode mode) { mode_ = mode; }
    ViewMode getMode() const { return mode_; }

    // Point cloud updates (thread-safe)
    void updatePointCloud(const PointData* points, size_t count);
    void appendPointCloud(const PointData* points, size_t count);
    void clearPointCloud();

    // Pose updates
    void updatePose(const M4D& pose, uint64_t timestamp_ns);

    // Mesh
    bool loadHullMesh(const std::string& filename);
    void clearHullMesh();

    // Coverage
    void initCoverageGrid();
    void clearCoverage();
    float getCoveragePercent() const;

    // Rendering
    void renderWidget(float width, float height);
    bool renderStandalone();
    bool processMessages();

    // Camera
    void setCameraTarget(const V3F& target) { camera_.setTarget(target); }
    void setCameraDistance(float dist) { camera_.setDistance(dist); }
    void setCameraRotation(float yaw, float pitch) { camera_.setRotation(yaw, pitch); }
    void resetCamera();
    void fitCameraToContent();

    // Stats
    SlamViewer::RenderStats getStats() const;

private:
    // D3D11 initialization
    bool createDeviceAndSwapChain(HWND hwnd, int width, int height);
    bool createShaders();
    bool createBuffers();
    bool createColormapTexture();
    bool createRenderTargets(int width, int height);

    // Rendering
    void beginFrame();
    void endFrame();
    void renderPointCloud();
    void renderMesh();
    void renderProbe();
    void renderCoverage();

    // Buffer management
    void swapPointBuffers();
    void uploadPointsToGPU();
    void uploadCoverageToGPU();

    // Input handling
    void handleInput();

    // Window proc (standalone mode)
    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);

private:
    // Configuration
    ViewerConfig config_;
    ViewMode mode_ = ViewMode::SCANNING;

    // D3D11 objects
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<IDXGISwapChain> swapChain_;
    ComPtr<ID3D11RenderTargetView> renderTargetView_;
    ComPtr<ID3D11DepthStencilView> depthStencilView_;
    ComPtr<ID3D11DepthStencilState> depthStencilState_;
    ComPtr<ID3D11RasterizerState> rasterizerState_;
    ComPtr<ID3D11BlendState> blendState_;

    // Point cloud rendering - instanced approach (faster for large point counts)
    ComPtr<ID3D11VertexShader> pointVS_;           // Instanced vertex shader
    ComPtr<ID3D11VertexShader> pointVS_GS_;        // Fallback GS vertex shader
    ComPtr<ID3D11GeometryShader> pointGS_;         // Geometry shader (fallback)
    ComPtr<ID3D11PixelShader> pointPS_;
    ComPtr<ID3D11InputLayout> pointLayout_;        // Instanced input layout (quad only)
    ComPtr<ID3D11InputLayout> pointLayout_GS_;     // GS input layout (point data)
    ComPtr<ID3D11Buffer> quadVertexBuffer_;        // Quad corners for instancing
    ComPtr<ID3D11Buffer> quadIndexBuffer_;         // Quad indices
    ComPtr<ID3D11Buffer> pointStructuredBuffer_;   // Structured buffer for point data
    ComPtr<ID3D11ShaderResourceView> pointBufferSRV_;  // SRV for structured buffer
    ComPtr<ID3D11Buffer> pointVertexBuffer_;       // Fallback vertex buffer for GS path
    ComPtr<ID3D11Buffer> cameraConstantBuffer_;
    bool useInstancing_ = false;                   // TODO: Debug instanced rendering visibility issue

    // GPU frustum culling via compute shader
    ComPtr<ID3D11ComputeShader> frustumCullCS_;
    ComPtr<ID3D11Buffer> gpuInputBuffer_;          // All points (static after upload)
    ComPtr<ID3D11ShaderResourceView> gpuInputSRV_;
    ComPtr<ID3D11Buffer> gpuOutputBuffer_;         // Visible points (after culling)
    ComPtr<ID3D11UnorderedAccessView> gpuOutputUAV_;
    ComPtr<ID3D11ShaderResourceView> gpuOutputSRV_;
    ComPtr<ID3D11Buffer> gpuCounterBuffer_;        // Atomic counter for visible points
    ComPtr<ID3D11UnorderedAccessView> gpuCounterUAV_;
    ComPtr<ID3D11Buffer> gpuCounterStaging_;       // For CPU readback of count
    ComPtr<ID3D11Buffer> cullConstantBuffer_;      // Frustum planes + params
    bool useGPUCulling_ = false;
    bool gpuPointsUploaded_ = false;
    size_t gpuTotalPoints_ = 0;

    ComPtr<ID3D11Texture1D> colormapTexture_;
    ComPtr<ID3D11ShaderResourceView> colormapSRV_;
    ComPtr<ID3D11SamplerState> colormapSampler_;

    // Mesh rendering
    ComPtr<ID3D11VertexShader> meshVS_;
    ComPtr<ID3D11PixelShader> meshPS_;
    ComPtr<ID3D11InputLayout> meshLayout_;
    ComPtr<ID3D11Buffer> meshVertexBuffer_;
    ComPtr<ID3D11Buffer> meshIndexBuffer_;
    ComPtr<ID3D11Buffer> meshConstantBuffer_;
    ComPtr<ID3D11Buffer> materialConstantBuffer_;

    // Coverage texture
    ComPtr<ID3D11Texture2D> coverageTexture_;
    ComPtr<ID3D11ShaderResourceView> coverageSRV_;

    // Probe rendering
    ComPtr<ID3D11VertexShader> probeVS_;
    ComPtr<ID3D11PixelShader> probePS_;
    ComPtr<ID3D11Buffer> probeVertexBuffer_;
    ComPtr<ID3D11Buffer> probeConstantBuffer_;

    // Double-buffered point data
    PointBuffer pointBuffers_[2];
    std::atomic<int> writeBuffer_{0};
    std::atomic<int> readBuffer_{1};
    std::atomic<bool> pointsUpdated_{false};
    std::mutex pointMutex_;

    // Mesh data
    std::vector<V3F> meshVertices_;
    std::vector<V3F> meshNormals_;
    std::vector<V2F> meshUVs_;
    std::vector<uint32_t> meshIndices_;
    bool meshLoaded_ = false;

    // Coverage
    CoverageGrid coverageGrid_;
    bool coverageInitialized_ = false;
    std::atomic<bool> coverageUpdated_{false};

    // Current pose
    M4D currentPose_ = M4D::Identity();
    std::atomic<bool> poseUpdated_{false};
    std::mutex poseMutex_;

    // Camera
    ArcballCamera camera_;

    // Frustum culling and LOD
    Frustum frustum_;
    bool frustumCullingEnabled_ = true;
    static constexpr size_t CHUNK_SIZE = 50000;  // Points per chunk for culling
    V3F cameraPositionForLOD_;  // Camera position cached for LOD calculations

    // Viewport
    int viewportWidth_ = 800;
    int viewportHeight_ = 600;

    // Standalone window
    HWND hwnd_ = nullptr;
    bool standalone_ = false;
    bool ownsDevice_ = false;

    // Mouse state for standalone window
    bool mouseLeftDown_ = false;
    bool mouseRightDown_ = false;
    bool mouseMiddleDown_ = false;
    int lastMouseX_ = 0;
    int lastMouseY_ = 0;

    // Stats
    mutable SlamViewer::RenderStats stats_;

    // Point bounds (for camera fit)
    V3F boundsMin_ = V3F::Zero();
    V3F boundsMax_ = V3F::Zero();
};

} // namespace viz
} // namespace slam
