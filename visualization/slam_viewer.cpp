/**
 * @file slam_viewer.cpp
 * @brief Implementation of SlamViewer with D3D11 GPU acceleration
 */

#include "slam_viewer_impl.hpp"

#include <imgui.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>

namespace slam {
namespace viz {

//==============================================================================
// Colormap Data (256 RGBA values)
//==============================================================================

namespace colormaps {

// Turbo colormap (good for intensity visualization)
static const uint32_t TURBO[256] = {
    0xFF191030, 0xFF1A1338, 0xFF1B1640, 0xFF1C1948, 0xFF1D1C50, 0xFF1E1F58, 0xFF1F2260, 0xFF202568,
    0xFF212870, 0xFF222B78, 0xFF232E80, 0xFF243188, 0xFF253490, 0xFF263798, 0xFF273AA0, 0xFF283DA8,
    0xFF2940B0, 0xFF2A43B8, 0xFF2B46C0, 0xFF2C49C8, 0xFF2D4CD0, 0xFF2E4FD8, 0xFF2F52E0, 0xFF3055E8,
    0xFF3158F0, 0xFF325BF8, 0xFF335EFF, 0xFF3461FF, 0xFF3564FF, 0xFF3667FF, 0xFF376AFF, 0xFF386DFF,
    0xFF3970FF, 0xFF3A73FF, 0xFF3B76FF, 0xFF3C79FF, 0xFF3D7CFF, 0xFF3E7FFF, 0xFF3F82FF, 0xFF4085FF,
    0xFF4188FF, 0xFF428BFF, 0xFF438EFF, 0xFF4491FF, 0xFF4594FF, 0xFF4697FF, 0xFF479AFF, 0xFF489DFF,
    0xFF49A0FF, 0xFF4AA3FF, 0xFF4BA6FF, 0xFF4CA9FF, 0xFF4DACFF, 0xFF4EAFFF, 0xFF4FB2FF, 0xFF50B5FF,
    0xFF51B8FF, 0xFF52BBFF, 0xFF53BEFF, 0xFF54C1FF, 0xFF55C4FF, 0xFF56C7FF, 0xFF57CAFF, 0xFF58CDFF,
    0xFF59D0FF, 0xFF5AD3FF, 0xFF5BD6F8, 0xFF5CD9F0, 0xFF5DDCE8, 0xFF5EDFE0, 0xFF5FE2D8, 0xFF60E5D0,
    0xFF61E8C8, 0xFF62EBC0, 0xFF63EEB8, 0xFF64F1B0, 0xFF65F4A8, 0xFF66F7A0, 0xFF67FA98, 0xFF68FD90,
    0xFF69FF88, 0xFF6CFF80, 0xFF6FFF78, 0xFF72FF70, 0xFF75FF68, 0xFF78FF60, 0xFF7BFF58, 0xFF7EFF50,
    0xFF81FF48, 0xFF84FF40, 0xFF87FF38, 0xFF8AFF30, 0xFF8DFF28, 0xFF90FF20, 0xFF93FF18, 0xFF96FF10,
    0xFF99FF08, 0xFF9CFF00, 0xFFA0FB00, 0xFFA4F700, 0xFFA8F300, 0xFFACEF00, 0xFFB0EB00, 0xFFB4E700,
    0xFFB8E300, 0xFFBCDF00, 0xFFC0DB00, 0xFFC4D700, 0xFFC8D300, 0xFFCCCF00, 0xFFD0CB00, 0xFFD4C700,
    0xFFD8C300, 0xFFDCBF00, 0xFFE0BB00, 0xFFE4B700, 0xFFE8B300, 0xFFECAF00, 0xFFF0AB00, 0xFFF4A700,
    0xFFF8A300, 0xFFFC9F00, 0xFFFF9B00, 0xFFFF9700, 0xFFFF9300, 0xFFFF8F00, 0xFFFF8B00, 0xFFFF8700,
    0xFFFF8300, 0xFFFF7F00, 0xFFFF7B00, 0xFFFF7700, 0xFFFF7300, 0xFFFF6F00, 0xFFFF6B00, 0xFFFF6700,
    0xFFFF6300, 0xFFFF5F00, 0xFFFF5B00, 0xFFFF5700, 0xFFFF5300, 0xFFFF4F00, 0xFFFF4B00, 0xFFFF4700,
    0xFFFF4300, 0xFFFF3F00, 0xFFFF3B00, 0xFFFF3700, 0xFFFF3300, 0xFFFF2F00, 0xFFFF2B00, 0xFFFF2700,
    0xFFFF2300, 0xFFFF1F00, 0xFFFF1B00, 0xFFFF1700, 0xFFFF1300, 0xFFFF0F00, 0xFFFF0B00, 0xFFFF0700,
    0xFFFF0300, 0xFFFB0000, 0xFFF70000, 0xFFF30000, 0xFFEF0000, 0xFFEB0000, 0xFFE70000, 0xFFE30000,
    0xFFDF0000, 0xFFDB0000, 0xFFD70000, 0xFFD30000, 0xFFCF0000, 0xFFCB0000, 0xFFC70000, 0xFFC30000,
    0xFFBF0000, 0xFFBB0000, 0xFFB70000, 0xFFB30000, 0xFFAF0000, 0xFFAB0000, 0xFFA70000, 0xFFA30000,
    0xFF9F0000, 0xFF9B0000, 0xFF970000, 0xFF930000, 0xFF8F0000, 0xFF8B0000, 0xFF870000, 0xFF830000,
    0xFF7F0000, 0xFF7B0000, 0xFF770000, 0xFF730000, 0xFF6F0000, 0xFF6B0000, 0xFF670000, 0xFF630000,
    0xFF5F0000, 0xFF5B0000, 0xFF570000, 0xFF530000, 0xFF4F0000, 0xFF4B0000, 0xFF470000, 0xFF430000,
    0xFF3F0000, 0xFF3B0000, 0xFF370000, 0xFF330000, 0xFF2F0000, 0xFF2B0000, 0xFF270000, 0xFF230000,
    0xFF1F0000, 0xFF1B0000, 0xFF170000, 0xFF130000, 0xFF0F0000, 0xFF0B0000, 0xFF070000, 0xFF030000,
    0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000,
    0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000,
    0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000,
    0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000, 0xFF000000
};

// Viridis colormap
static const uint32_t VIRIDIS[256] = {
    0xFF440154, 0xFF440255, 0xFF450357, 0xFF450558, 0xFF46065A, 0xFF46085B, 0xFF46095D, 0xFF460B5E,
    0xFF470C60, 0xFF470E61, 0xFF471063, 0xFF471164, 0xFF471366, 0xFF481467, 0xFF481669, 0xFF48176A,
    0xFF48196C, 0xFF481A6D, 0xFF481C6E, 0xFF481D70, 0xFF481F71, 0xFF482072, 0xFF482274, 0xFF482375,
    0xFF482576, 0xFF482677, 0xFF482878, 0xFF482979, 0xFF472B7A, 0xFF472C7B, 0xFF472E7C, 0xFF472F7D,
    0xFF46307E, 0xFF46327F, 0xFF463380, 0xFF453581, 0xFF453681, 0xFF443882, 0xFF443983, 0xFF433A83,
    0xFF433C84, 0xFF423D84, 0xFF423E85, 0xFF414085, 0xFF404186, 0xFF404286, 0xFF3F4487, 0xFF3E4587,
    0xFF3E4687, 0xFF3D4788, 0xFF3C4988, 0xFF3C4A88, 0xFF3B4B89, 0xFF3A4C89, 0xFF3A4E89, 0xFF394F89,
    0xFF385089, 0xFF38518A, 0xFF37528A, 0xFF36548A, 0xFF36558A, 0xFF35568A, 0xFF34578A, 0xFF34588A,
    0xFF33598A, 0xFF335A8A, 0xFF325B8A, 0xFF315D8A, 0xFF315E8A, 0xFF305F8A, 0xFF30608A, 0xFF2F618A,
    0xFF2E628A, 0xFF2E638A, 0xFF2D648A, 0xFF2D658A, 0xFF2C668A, 0xFF2C678A, 0xFF2B688A, 0xFF2B698A,
    0xFF2A6A8A, 0xFF2A6B8A, 0xFF296C89, 0xFF296D89, 0xFF286E89, 0xFF286F89, 0xFF277089, 0xFF277189,
    0xFF277289, 0xFF267388, 0xFF267488, 0xFF257588, 0xFF257688, 0xFF247788, 0xFF247887, 0xFF247987,
    0xFF237A87, 0xFF237B87, 0xFF227C86, 0xFF227D86, 0xFF227E86, 0xFF217F85, 0xFF218085, 0xFF218185,
    0xFF208284, 0xFF208384, 0xFF208483, 0xFF1F8583, 0xFF1F8683, 0xFF1F8782, 0xFF1E8882, 0xFF1E8981,
    0xFF1E8A81, 0xFF1E8B80, 0xFF1D8C80, 0xFF1D8D7F, 0xFF1D8E7F, 0xFF1D8F7E, 0xFF1D907E, 0xFF1D917D,
    0xFF1D927C, 0xFF1D937C, 0xFF1D947B, 0xFF1D957A, 0xFF1D967A, 0xFF1D9779, 0xFF1E9878, 0xFF1E9978,
    0xFF1E9A77, 0xFF1F9B76, 0xFF1F9C75, 0xFF209D74, 0xFF209E74, 0xFF219F73, 0xFF22A072, 0xFF22A171,
    0xFF23A270, 0xFF24A36F, 0xFF25A46E, 0xFF26A56D, 0xFF27A66C, 0xFF28A76B, 0xFF29A86A, 0xFF2AA969,
    0xFF2BAA68, 0xFF2CAB67, 0xFF2EAC66, 0xFF2FAD65, 0xFF30AE63, 0xFF32AF62, 0xFF33B061, 0xFF35B160,
    0xFF36B25F, 0xFF38B35D, 0xFF3AB45C, 0xFF3BB55B, 0xFF3DB659, 0xFF3FB758, 0xFF41B857, 0xFF43B955,
    0xFF45BA54, 0xFF47BB52, 0xFF49BC51, 0xFF4BBD4F, 0xFF4DBE4E, 0xFF4FBF4C, 0xFF52C04A, 0xFF54C149,
    0xFF56C247, 0xFF59C345, 0xFF5BC444, 0xFF5EC542, 0xFF60C640, 0xFF63C73E, 0xFF65C73D, 0xFF68C83B,
    0xFF6BC939, 0xFF6DCA37, 0xFF70CB35, 0xFF73CC33, 0xFF76CD31, 0xFF79CE2F, 0xFF7CCF2D, 0xFF7FD02B,
    0xFF82D029, 0xFF85D127, 0xFF88D225, 0xFF8BD323, 0xFF8ED420, 0xFF92D41E, 0xFF95D51C, 0xFF98D61A,
    0xFF9BD718, 0xFF9FD716, 0xFFA2D814, 0xFFA5D912, 0xFFA9D911, 0xFFACDA0F, 0xFFB0DB0E, 0xFFB3DB0D,
    0xFFB7DC0C, 0xFFBADD0C, 0xFFBEDD0C, 0xFFC1DE0D, 0xFFC5DF0E, 0xFFC8DF10, 0xFFCCE012, 0xFFCFE115,
    0xFFD3E118, 0xFFD6E21B, 0xFFDAE21F, 0xFFDDE323, 0xFFE1E327, 0xFFE4E42C, 0xFFE8E431, 0xFFEBE537,
    0xFFEEE53D, 0xFFF2E643, 0xFFF5E64A, 0xFFF8E751, 0xFFFBE758, 0xFFFDE760, 0xFFFEE866, 0xFFFEE86D,
    0xFFFEE874, 0xFFFEE97A, 0xFFFEE981, 0xFFFEE987, 0xFFFEE98D, 0xFFFEE993, 0xFFFEE999, 0xFFFEE99F,
    0xFFFEE9A5, 0xFFFEE9AB, 0xFFFEE9B0, 0xFFFEE9B5, 0xFFFEE9BA, 0xFFFEE9BE, 0xFFFEE9C3, 0xFFFEE9C7,
    0xFFFEE9CB, 0xFFFEE9CF, 0xFFFEE9D2, 0xFFFEE9D5, 0xFFFEE9D9, 0xFFFEE9DC, 0xFFFEE9DE, 0xFFFEE9E1,
    0xFFFEE9E3, 0xFFFEE9E6, 0xFFFEE9E8, 0xFFFEE9EA, 0xFFFEE9EC, 0xFFFEE9ED, 0xFFFEE9EF, 0xFFFEE9F0
};

} // namespace colormaps

//==============================================================================
// SlamViewer Public Interface
//==============================================================================

SlamViewer::SlamViewer() : impl_(std::make_unique<SlamViewerImpl>()) {}
SlamViewer::~SlamViewer() = default;
SlamViewer::SlamViewer(SlamViewer&&) noexcept = default;
SlamViewer& SlamViewer::operator=(SlamViewer&&) noexcept = default;

bool SlamViewer::initWithDevice(void* device, void* context) {
    return impl_->initWithDevice(device, context);
}

bool SlamViewer::initStandalone(int width, int height, const char* title) {
    return impl_->initStandalone(width, height, title);
}

void SlamViewer::setConfig(const ViewerConfig& config) {
    impl_->setConfig(config);
}

const ViewerConfig& SlamViewer::getConfig() const {
    return impl_->getConfig();
}

void SlamViewer::setMode(ViewMode mode) {
    impl_->setMode(mode);
}

ViewMode SlamViewer::getMode() const {
    return impl_->getMode();
}

void SlamViewer::updatePointCloud(const PointData* points, size_t count) {
    impl_->updatePointCloud(points, count);
}

void SlamViewer::appendPointCloud(const PointData* points, size_t count) {
    impl_->appendPointCloud(points, count);
}

void SlamViewer::updateOverlayPointCloud(const PointData* points, size_t count) {
    impl_->updateOverlayPointCloud(points, count);
}

void SlamViewer::setOverlayColorTint(float r, float g, float b) {
    impl_->setOverlayColorTint(r, g, b);
}

void SlamViewer::clearOverlayPointCloud() {
    impl_->clearOverlayPointCloud();
}

void SlamViewer::updatePose(const M4D& pose, uint64_t timestamp_ns) {
    impl_->updatePose(pose, timestamp_ns);
}

void SlamViewer::clearPointCloud() {
    impl_->clearPointCloud();
}

bool SlamViewer::loadHullMesh(const std::string& filename) {
    return impl_->loadHullMesh(filename);
}

void SlamViewer::clearHullMesh() {
    impl_->clearHullMesh();
}

void SlamViewer::initCoverageGrid() {
    impl_->initCoverageGrid();
}

void SlamViewer::clearCoverage() {
    impl_->clearCoverage();
}

float SlamViewer::getCoveragePercent() const {
    return impl_->getCoveragePercent();
}

void SlamViewer::renderWidget(float width, float height) {
    impl_->renderWidget(width, height);
}

void* SlamViewer::getImGuiTexture() const {
    return impl_->getImGuiTexture();
}

bool SlamViewer::renderStandalone() {
    return impl_->renderStandalone();
}

bool SlamViewer::processMessages() {
    return impl_->processMessages();
}

void SlamViewer::setCameraTarget(const V3F& target) {
    impl_->setCameraTarget(target);
}

void SlamViewer::setCameraDistance(float distance) {
    impl_->setCameraDistance(distance);
}

void SlamViewer::setCameraRotation(float yaw, float pitch) {
    impl_->setCameraRotation(yaw, pitch);
}

void SlamViewer::resetCamera() {
    impl_->resetCamera();
}

void SlamViewer::fitCameraToContent() {
    impl_->fitCameraToContent();
}

void SlamViewer::setRobotPose(float x, float y, float heading) {
    impl_->setRobotPose(x, y, heading);
}

void SlamViewer::centerCameraOnRobot() {
    impl_->centerCameraOnRobot();
}

void SlamViewer::setMapClickCallback(MapClickCallback callback) {
    impl_->setMapClickCallback(std::move(callback));
}

void SlamViewer::setClickToPoseMode(bool enabled) {
    impl_->setClickToPoseMode(enabled);
}

bool SlamViewer::isClickToPoseMode() const {
    return impl_->isClickToPoseMode();
}

SlamViewer::RenderStats SlamViewer::getStats() const {
    return impl_->getStats();
}

//==============================================================================
// ArcballCamera Implementation
//==============================================================================

void ArcballCamera::rotate(float dyaw, float dpitch) {
    yaw_ += dyaw;
    // Clamp pitch to prevent going exactly vertical (causes gimbal lock)
    pitch_ = std::clamp(pitch_ + dpitch, -1.5f, 1.5f);
}

void ArcballCamera::pan(float dx, float dy) {
    // Compute right and up vectors consistent with getViewMatrix()
    V3F pos = getPosition();
    V3F forward = (target_ - pos).normalized();
    V3F worldUp(0, 0, 1);
    V3F right = forward.cross(worldUp).normalized();

    // Handle looking straight up/down
    if (right.norm() < 0.001f) {
        right = V3F(1, 0, 0);
    }

    V3F up = right.cross(forward).normalized();

    // Pan target in screen space directions
    target_ += right * dx * distance_ * 0.002f;
    target_ += up * dy * distance_ * 0.002f;
}

void ArcballCamera::zoom(float delta) {
    distance_ *= (1.0f - delta * 0.1f);
    distance_ = std::clamp(distance_, 0.1f, 1000.0f);
}

M4F ArcballCamera::getViewMatrix() const {
    V3F pos = getPosition();
    V3F forward = (target_ - pos).normalized();

    // Z is up in world space
    V3F worldUp(0, 0, 1);
    V3F right = forward.cross(worldUp).normalized();

    // Handle looking straight up/down
    if (right.norm() < 0.001f) {
        right = V3F(1, 0, 0);
    }

    V3F up = right.cross(forward).normalized();

    M4F view = M4F::Identity();
    view(0, 0) = right.x();    view(0, 1) = right.y();    view(0, 2) = right.z();
    view(1, 0) = up.x();       view(1, 1) = up.y();       view(1, 2) = up.z();
    view(2, 0) = -forward.x(); view(2, 1) = -forward.y(); view(2, 2) = -forward.z();
    view(0, 3) = -right.dot(pos);
    view(1, 3) = -up.dot(pos);
    view(2, 3) = forward.dot(pos);

    return view;
}

M4F ArcballCamera::getProjectionMatrix(float aspect, float fov, float nearPlane, float farPlane) const {
    float tanHalfFov = std::tan(fov * 0.5f * 3.14159265f / 180.0f);

    M4F proj = M4F::Zero();
    proj(0, 0) = 1.0f / (aspect * tanHalfFov);
    proj(1, 1) = 1.0f / tanHalfFov;
    proj(2, 2) = farPlane / (nearPlane - farPlane);
    proj(2, 3) = (nearPlane * farPlane) / (nearPlane - farPlane);
    proj(3, 2) = -1.0f;

    return proj;
}

V3F ArcballCamera::getPosition() const {
    float cy = std::cos(yaw_);
    float sy = std::sin(yaw_);
    float cp = std::cos(pitch_);
    float sp = std::sin(pitch_);

    // Z is up: camera orbits in XY plane, pitch elevates in Z
    // yaw rotates around Z axis, pitch tilts up/down
    return target_ + V3F(cy * cp, sy * cp, sp) * distance_;
}

void ArcballCamera::fitToBounds(const V3F& min, const V3F& max) {
    target_ = (min + max) * 0.5f;
    float diagonal = (max - min).norm();
    distance_ = diagonal * 1.5f;
}

//==============================================================================
// CoverageGrid Implementation
//==============================================================================

std::vector<size_t> CoverageGrid::SpatialHash::query(const V3F& center, float radius) const {
    std::vector<size_t> result;
    int range = static_cast<int>(std::ceil(radius / cellSize)) + 1;

    int cx = static_cast<int>(std::floor(center.x() / cellSize));
    int cy = static_cast<int>(std::floor(center.y() / cellSize));
    int cz = static_cast<int>(std::floor(center.z() / cellSize));

    for (int dx = -range; dx <= range; dx++) {
        for (int dy = -range; dy <= range; dy++) {
            for (int dz = -range; dz <= range; dz++) {
                uint64_t h = (static_cast<uint64_t>(cx + dx) & 0x1FFFFF) |
                            ((static_cast<uint64_t>(cy + dy) & 0x1FFFFF) << 21) |
                            ((static_cast<uint64_t>(cz + dz) & 0x1FFFFF) << 42);
                auto it = grid.find(h);
                if (it != grid.end()) {
                    result.insert(result.end(), it->second.begin(), it->second.end());
                }
            }
        }
    }
    return result;
}

void CoverageGrid::initFromMesh(const std::vector<V3F>& vertices,
                                 const std::vector<V3F>& normals,
                                 const std::vector<uint32_t>& indices,
                                 const std::vector<V2F>& uvs,
                                 float cellSize) {
    cells_.clear();
    spatialIndex_.grid.clear();
    cellSize_ = cellSize;
    coveredCount_ = 0;

    // Sample mesh surface at cell resolution
    for (size_t i = 0; i < indices.size(); i += 3) {
        V3F v0 = vertices[indices[i]];
        V3F v1 = vertices[indices[i + 1]];
        V3F v2 = vertices[indices[i + 2]];

        V3F n0 = normals[indices[i]];
        V3F n1 = normals[indices[i + 1]];
        V3F n2 = normals[indices[i + 2]];

        V2F uv0 = uvs.empty() ? V2F::Zero() : uvs[indices[i]];
        V2F uv1 = uvs.empty() ? V2F::Zero() : uvs[indices[i + 1]];
        V2F uv2 = uvs.empty() ? V2F::Zero() : uvs[indices[i + 2]];

        // Calculate triangle area to determine sampling density
        V3F edge1 = v1 - v0;
        V3F edge2 = v2 - v0;
        float area = edge1.cross(edge2).norm() * 0.5f;

        int samples = std::max(1, static_cast<int>(area / (cellSize * cellSize)));

        // Uniform sampling within triangle
        for (int s = 0; s < samples; s++) {
            float r1 = static_cast<float>(rand()) / RAND_MAX;
            float r2 = static_cast<float>(rand()) / RAND_MAX;
            if (r1 + r2 > 1.0f) {
                r1 = 1.0f - r1;
                r2 = 1.0f - r2;
            }
            float r3 = 1.0f - r1 - r2;

            CoverageCell cell;
            cell.center = v0 * r1 + v1 * r2 + v2 * r3;
            cell.normal = (n0 * r1 + n1 * r2 + n2 * r3).normalized();
            cell.uv = uv0 * r1 + uv1 * r2 + uv2 * r3;
            cell.covered = false;

            size_t idx = cells_.size();
            cells_.push_back(cell);
            spatialIndex_.insert(idx, cell.center);
        }
    }
}

void CoverageGrid::markCovered(const V3F& probeCenter, const V3F& probeAxis, float probeWidth) {
    float halfWidth = probeWidth * 0.5f;

    // Query nearby cells
    auto candidates = spatialIndex_.query(probeCenter, probeWidth);

    for (size_t idx : candidates) {
        if (idx >= cells_.size()) continue;
        auto& cell = cells_[idx];
        if (cell.covered) continue;

        // Check if cell is within probe footprint (line segment)
        V3F toCell = cell.center - probeCenter;
        float alongAxis = toCell.dot(probeAxis);

        if (std::abs(alongAxis) <= halfWidth) {
            // Cell is within the probe stripe
            cell.covered = true;
            coveredCount_++;
        }
    }
}

void CoverageGrid::clear() {
    for (auto& cell : cells_) {
        cell.covered = false;
    }
    coveredCount_ = 0;
}

float CoverageGrid::getCoveragePercent() const {
    if (cells_.empty()) return 0.0f;
    return 100.0f * static_cast<float>(coveredCount_) / static_cast<float>(cells_.size());
}

void CoverageGrid::getCoverageTexture(std::vector<float>& data, int& width, int& height) const {
    // For now, create a simple 1D representation
    // In production, this would map to UV space
    width = 256;
    height = 256;
    data.resize(width * height, 0.0f);

    for (const auto& cell : cells_) {
        if (!cell.covered) continue;
        int u = std::clamp(static_cast<int>(cell.uv.x() * width), 0, width - 1);
        int v = std::clamp(static_cast<int>(cell.uv.y() * height), 0, height - 1);
        data[v * width + u] = 1.0f;
    }
}

//==============================================================================
// SlamViewerImpl Implementation
//==============================================================================

SlamViewerImpl::SlamViewerImpl() {
    // Initialize point buffers
    pointBuffers_[0].points.reserve(1000000);
    pointBuffers_[1].points.reserve(1000000);
}

SlamViewerImpl::~SlamViewerImpl() {
    shutdown();
}

void SlamViewerImpl::shutdown() {
    if (hwnd_ && standalone_) {
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
    }

    // Release D3D11 resources (ComPtr handles this automatically)
    if (ownsDevice_) {
        swapChain_.Reset();
        context_.Reset();
        device_.Reset();
    }
}

bool SlamViewerImpl::initWithDevice(void* device, void* context) {
    device_ = static_cast<ID3D11Device*>(device);
    context_ = static_cast<ID3D11DeviceContext*>(context);
    ownsDevice_ = false;

    if (!createShaders()) return false;
    if (!createBuffers()) return false;
    if (!createColormapTexture()) return false;

    return true;
}

bool SlamViewerImpl::initStandalone(int width, int height, const char* title) {
    standalone_ = true;
    viewportWidth_ = width;
    viewportHeight_ = height;

    // Register window class
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"SlamViewerClass";
    RegisterClassExW(&wc);

    // Create window
    RECT rc = {0, 0, width, height};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    std::wstring wtitle(title, title + strlen(title));
    hwnd_ = CreateWindowW(L"SlamViewerClass", wtitle.c_str(),
                          WS_OVERLAPPEDWINDOW,
                          CW_USEDEFAULT, CW_USEDEFAULT,
                          rc.right - rc.left, rc.bottom - rc.top,
                          nullptr, nullptr, GetModuleHandle(nullptr), this);

    if (!hwnd_) return false;

    ShowWindow(hwnd_, SW_SHOW);
    UpdateWindow(hwnd_);

    // Create D3D11 device
    if (!createDeviceAndSwapChain(hwnd_, width, height)) return false;
    if (!createShaders()) return false;
    if (!createBuffers()) return false;
    if (!createColormapTexture()) return false;
    if (!createRenderTargets(width, height)) return false;

    ownsDevice_ = true;
    return true;
}

bool SlamViewerImpl::createDeviceAndSwapChain(HWND hwnd, int width, int height) {
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 2;
    sd.BufferDesc.Width = width;
    sd.BufferDesc.Height = height;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    D3D_FEATURE_LEVEL featureLevel;
    UINT flags = 0;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
        nullptr, 0, D3D11_SDK_VERSION,
        &sd, &swapChain_, &device_, &featureLevel, &context_);

    return SUCCEEDED(hr);
}

bool SlamViewerImpl::createShaders() {
    UINT compileFlags = 0;
#ifdef _DEBUG
    compileFlags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    ComPtr<ID3DBlob> vsBlob, vsGSBlob, gsBlob, psBlob, errorBlob;

    // 1. Instanced vertex shader (uses structured buffer + SV_InstanceID)
    HRESULT hr = D3DCompile(shaders::POINT_CLOUD_VS, strlen(shaders::POINT_CLOUD_VS),
                            "PointCloudVS", nullptr, nullptr, "main", "vs_5_0",
                            compileFlags, 0, &vsBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) OutputDebugStringA((char*)errorBlob->GetBufferPointer());
        useInstancing_ = false;  // Fall back to GS approach
    } else {
        device_->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(),
                                    nullptr, &pointVS_);

        // Input layout for instanced rendering (just quad corners)
        D3D11_INPUT_ELEMENT_DESC instLayout[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        };
        device_->CreateInputLayout(instLayout, 1, vsBlob->GetBufferPointer(),
                                   vsBlob->GetBufferSize(), &pointLayout_);

        // Instancing compiles but GS is faster on most hardware, keep disabled
        // useInstancing_ = true;
    }

    // 2. Fallback: Geometry shader vertex shader
    hr = D3DCompile(shaders::POINT_CLOUD_VS_GS, strlen(shaders::POINT_CLOUD_VS_GS),
                    "PointCloudVS_GS", nullptr, nullptr, "main", "vs_5_0",
                    compileFlags, 0, &vsGSBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) OutputDebugStringA((char*)errorBlob->GetBufferPointer());
        return false;
    }
    device_->CreateVertexShader(vsGSBlob->GetBufferPointer(), vsGSBlob->GetBufferSize(),
                                nullptr, &pointVS_GS_);

    // Input layout for GS path (position + intensity from vertex buffer)
    D3D11_INPUT_ELEMENT_DESC gsLayout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R8_UNORM, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    device_->CreateInputLayout(gsLayout, 2, vsGSBlob->GetBufferPointer(),
                               vsGSBlob->GetBufferSize(), &pointLayout_GS_);

    // 3. Geometry shader (fallback path)
    hr = D3DCompile(shaders::POINT_CLOUD_GS, strlen(shaders::POINT_CLOUD_GS),
                    "PointCloudGS", nullptr, nullptr, "main", "gs_5_0",
                    compileFlags, 0, &gsBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) OutputDebugStringA((char*)errorBlob->GetBufferPointer());
        return false;
    }
    device_->CreateGeometryShader(gsBlob->GetBufferPointer(), gsBlob->GetBufferSize(),
                                  nullptr, &pointGS_);

    // 4. Pixel shader (shared by both paths)
    hr = D3DCompile(shaders::POINT_CLOUD_PS, strlen(shaders::POINT_CLOUD_PS),
                    "PointCloudPS", nullptr, nullptr, "main", "ps_5_0",
                    compileFlags, 0, &psBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) OutputDebugStringA((char*)errorBlob->GetBufferPointer());
        return false;
    }
    device_->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(),
                               nullptr, &pointPS_);

    // Mesh shaders (similar process)
    hr = D3DCompile(shaders::MESH_VS, strlen(shaders::MESH_VS),
                    "MeshVS", nullptr, nullptr, "main", "vs_5_0",
                    compileFlags, 0, &vsBlob, &errorBlob);
    if (SUCCEEDED(hr)) {
        device_->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(),
                                    nullptr, &meshVS_);

        D3D11_INPUT_ELEMENT_DESC meshLayoutDesc[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
        };
        device_->CreateInputLayout(meshLayoutDesc, 3, vsBlob->GetBufferPointer(),
                                   vsBlob->GetBufferSize(), &meshLayout_);
    }

    hr = D3DCompile(shaders::MESH_PS, strlen(shaders::MESH_PS),
                    "MeshPS", nullptr, nullptr, "main", "ps_5_0",
                    compileFlags, 0, &psBlob, &errorBlob);
    if (SUCCEEDED(hr)) {
        device_->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(),
                                   nullptr, &meshPS_);
    }

    // Compute shader for GPU frustum culling
    ComPtr<ID3DBlob> csBlob;
    hr = D3DCompile(shaders::FRUSTUM_CULL_CS, strlen(shaders::FRUSTUM_CULL_CS),
                    "FrustumCullCS", nullptr, nullptr, "main", "cs_5_0",
                    compileFlags, 0, &csBlob, &errorBlob);
    if (SUCCEEDED(hr)) {
        device_->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(),
                                     nullptr, &frustumCullCS_);
        useGPUCulling_ = true;
    } else {
        if (errorBlob) OutputDebugStringA((char*)errorBlob->GetBufferPointer());
        useGPUCulling_ = false;
    }

    return true;
}

// GPU point data struct (must match shader)
struct GPUPointData {
    float x, y, z;
    float intensity;  // float for 16-byte alignment
};

// GPU culling constant buffer (must match compute shader)
struct CullParams {
    V4F frustumPlanes[6];
    V3F cameraPos;
    float lodDistance;
    uint32_t totalPoints;
    uint32_t maxOutputPoints;
    float padding[2];
};

bool SlamViewerImpl::createBuffers() {
    // Camera constant buffer
    D3D11_BUFFER_DESC cbd = {};
    cbd.Usage = D3D11_USAGE_DYNAMIC;
    cbd.ByteWidth = sizeof(CameraConstants);
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    device_->CreateBuffer(&cbd, nullptr, &cameraConstantBuffer_);

    // === Instanced rendering resources ===
    if (useInstancing_) {
        // Quad vertex buffer (4 corners)
        float quadVerts[] = {
            -1.0f, -1.0f,  // BL
             1.0f, -1.0f,  // BR
             1.0f,  1.0f,  // TR
            -1.0f,  1.0f,  // TL
        };
        D3D11_BUFFER_DESC qvbd = {};
        qvbd.Usage = D3D11_USAGE_DEFAULT;
        qvbd.ByteWidth = sizeof(quadVerts);
        qvbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        D3D11_SUBRESOURCE_DATA qvData = {};
        qvData.pSysMem = quadVerts;
        device_->CreateBuffer(&qvbd, &qvData, &quadVertexBuffer_);

        // Quad index buffer (2 triangles)
        uint16_t quadIndices[] = { 0, 1, 2, 0, 2, 3 };
        D3D11_BUFFER_DESC qibd = {};
        qibd.Usage = D3D11_USAGE_DEFAULT;
        qibd.ByteWidth = sizeof(quadIndices);
        qibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
        D3D11_SUBRESOURCE_DATA qiData = {};
        qiData.pSysMem = quadIndices;
        device_->CreateBuffer(&qibd, &qiData, &quadIndexBuffer_);

        // Structured buffer for point data (dynamic, large capacity)
        D3D11_BUFFER_DESC sbd = {};
        sbd.Usage = D3D11_USAGE_DYNAMIC;
        sbd.ByteWidth = sizeof(GPUPointData) * 30000000;  // 30M points max
        sbd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        sbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        sbd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        sbd.StructureByteStride = sizeof(GPUPointData);
        device_->CreateBuffer(&sbd, nullptr, &pointStructuredBuffer_);

        // Shader resource view for structured buffer
        D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
        srvd.Format = DXGI_FORMAT_UNKNOWN;
        srvd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvd.Buffer.FirstElement = 0;
        srvd.Buffer.NumElements = 30000000;
        device_->CreateShaderResourceView(pointStructuredBuffer_.Get(), &srvd, &pointBufferSRV_);
    }

    // === GS fallback resources ===
    // Point vertex buffer (dynamic, large capacity) - for GS path
    D3D11_BUFFER_DESC vbd = {};
    vbd.Usage = D3D11_USAGE_DYNAMIC;
    vbd.ByteWidth = sizeof(PointData) * 30000000;  // 30M points max
    vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    device_->CreateBuffer(&vbd, nullptr, &pointVertexBuffer_);

    // === GPU frustum culling resources ===
    if (useGPUCulling_) {
        const size_t maxPoints = 30000000;

        // Input buffer (all points, static after initial upload)
        D3D11_BUFFER_DESC inputBufDesc = {};
        inputBufDesc.Usage = D3D11_USAGE_DEFAULT;
        inputBufDesc.ByteWidth = sizeof(GPUPointData) * maxPoints;
        inputBufDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        inputBufDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        inputBufDesc.StructureByteStride = sizeof(GPUPointData);
        device_->CreateBuffer(&inputBufDesc, nullptr, &gpuInputBuffer_);

        D3D11_SHADER_RESOURCE_VIEW_DESC inputSRVDesc = {};
        inputSRVDesc.Format = DXGI_FORMAT_UNKNOWN;
        inputSRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        inputSRVDesc.Buffer.NumElements = maxPoints;
        device_->CreateShaderResourceView(gpuInputBuffer_.Get(), &inputSRVDesc, &gpuInputSRV_);

        // Output buffer (visible points after culling)
        D3D11_BUFFER_DESC outputBufDesc = {};
        outputBufDesc.Usage = D3D11_USAGE_DEFAULT;
        outputBufDesc.ByteWidth = sizeof(GPUPointData) * maxPoints;
        outputBufDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        outputBufDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        outputBufDesc.StructureByteStride = sizeof(GPUPointData);
        device_->CreateBuffer(&outputBufDesc, nullptr, &gpuOutputBuffer_);

        D3D11_UNORDERED_ACCESS_VIEW_DESC outputUAVDesc = {};
        outputUAVDesc.Format = DXGI_FORMAT_UNKNOWN;
        outputUAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        outputUAVDesc.Buffer.NumElements = maxPoints;
        device_->CreateUnorderedAccessView(gpuOutputBuffer_.Get(), &outputUAVDesc, &gpuOutputUAV_);

        D3D11_SHADER_RESOURCE_VIEW_DESC outputSRVDesc = {};
        outputSRVDesc.Format = DXGI_FORMAT_UNKNOWN;
        outputSRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        outputSRVDesc.Buffer.NumElements = maxPoints;
        device_->CreateShaderResourceView(gpuOutputBuffer_.Get(), &outputSRVDesc, &gpuOutputSRV_);

        // Counter buffer (single uint for atomic count)
        D3D11_BUFFER_DESC counterBufDesc = {};
        counterBufDesc.Usage = D3D11_USAGE_DEFAULT;
        counterBufDesc.ByteWidth = sizeof(uint32_t);
        counterBufDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        counterBufDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        counterBufDesc.StructureByteStride = sizeof(uint32_t);
        device_->CreateBuffer(&counterBufDesc, nullptr, &gpuCounterBuffer_);

        D3D11_UNORDERED_ACCESS_VIEW_DESC counterUAVDesc = {};
        counterUAVDesc.Format = DXGI_FORMAT_UNKNOWN;
        counterUAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        counterUAVDesc.Buffer.NumElements = 1;
        device_->CreateUnorderedAccessView(gpuCounterBuffer_.Get(), &counterUAVDesc, &gpuCounterUAV_);

        // Staging buffer for CPU readback of counter
        D3D11_BUFFER_DESC stagingBufDesc = {};
        stagingBufDesc.Usage = D3D11_USAGE_STAGING;
        stagingBufDesc.ByteWidth = sizeof(uint32_t);
        stagingBufDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        device_->CreateBuffer(&stagingBufDesc, nullptr, &gpuCounterStaging_);

        // Cull constant buffer
        D3D11_BUFFER_DESC cullCBDesc = {};
        cullCBDesc.Usage = D3D11_USAGE_DYNAMIC;
        cullCBDesc.ByteWidth = sizeof(CullParams);
        cullCBDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        cullCBDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        device_->CreateBuffer(&cullCBDesc, nullptr, &cullConstantBuffer_);
    }

    // Mesh constant buffer
    cbd.ByteWidth = sizeof(MeshConstants);
    device_->CreateBuffer(&cbd, nullptr, &meshConstantBuffer_);

    // Material constant buffer
    cbd.ByteWidth = sizeof(MaterialConstants);
    device_->CreateBuffer(&cbd, nullptr, &materialConstantBuffer_);

    // Depth stencil state
    D3D11_DEPTH_STENCIL_DESC dsd = {};
    dsd.DepthEnable = TRUE;
    dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    dsd.DepthFunc = D3D11_COMPARISON_LESS;
    device_->CreateDepthStencilState(&dsd, &depthStencilState_);

    // Rasterizer state
    D3D11_RASTERIZER_DESC rsd = {};
    rsd.FillMode = D3D11_FILL_SOLID;
    rsd.CullMode = D3D11_CULL_BACK;
    rsd.FrontCounterClockwise = FALSE;
    rsd.DepthClipEnable = TRUE;
    device_->CreateRasterizerState(&rsd, &rasterizerState_);

    // Blend state (alpha blending)
    D3D11_BLEND_DESC bsd = {};
    bsd.RenderTarget[0].BlendEnable = TRUE;
    bsd.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    bsd.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    bsd.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    bsd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    bsd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    bsd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    bsd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    device_->CreateBlendState(&bsd, &blendState_);

    return true;
}

bool SlamViewerImpl::createColormapTexture() {
    D3D11_TEXTURE1D_DESC td = {};
    td.Width = 256;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = colormaps::TURBO;
    initData.SysMemPitch = 256 * 4;

    device_->CreateTexture1D(&td, &initData, &colormapTexture_);

    D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
    srvd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE1D;
    srvd.Texture1D.MipLevels = 1;
    device_->CreateShaderResourceView(colormapTexture_.Get(), &srvd, &colormapSRV_);

    D3D11_SAMPLER_DESC sd = {};
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    device_->CreateSamplerState(&sd, &colormapSampler_);

    return true;
}

bool SlamViewerImpl::createRenderTargets(int width, int height) {
    // Get back buffer
    ComPtr<ID3D11Texture2D> backBuffer;
    swapChain_->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
    device_->CreateRenderTargetView(backBuffer.Get(), nullptr, &renderTargetView_);

    // Create depth buffer
    D3D11_TEXTURE2D_DESC dtd = {};
    dtd.Width = width;
    dtd.Height = height;
    dtd.MipLevels = 1;
    dtd.ArraySize = 1;
    dtd.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    dtd.SampleDesc.Count = 1;
    dtd.Usage = D3D11_USAGE_DEFAULT;
    dtd.BindFlags = D3D11_BIND_DEPTH_STENCIL;

    ComPtr<ID3D11Texture2D> depthBuffer;
    device_->CreateTexture2D(&dtd, nullptr, &depthBuffer);
    device_->CreateDepthStencilView(depthBuffer.Get(), nullptr, &depthStencilView_);

    return true;
}

bool SlamViewerImpl::createWidgetRenderTarget(int width, int height) {
    if (width <= 0 || height <= 0) return false;
    if (width == widgetTargetWidth_ && height == widgetTargetHeight_ && widgetRenderTarget_) {
        return true;  // Already correct size
    }

    // Release old resources
    widgetSRV_.Reset();
    widgetRTV_.Reset();
    widgetRenderTarget_.Reset();
    widgetDSV_.Reset();
    widgetDepthBuffer_.Reset();

    // Create render target texture
    D3D11_TEXTURE2D_DESC rtd = {};
    rtd.Width = width;
    rtd.Height = height;
    rtd.MipLevels = 1;
    rtd.ArraySize = 1;
    rtd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rtd.SampleDesc.Count = 1;
    rtd.Usage = D3D11_USAGE_DEFAULT;
    rtd.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

    HRESULT hr = device_->CreateTexture2D(&rtd, nullptr, &widgetRenderTarget_);
    if (FAILED(hr)) return false;

    // Create render target view
    hr = device_->CreateRenderTargetView(widgetRenderTarget_.Get(), nullptr, &widgetRTV_);
    if (FAILED(hr)) return false;

    // Create shader resource view (for ImGui to read)
    hr = device_->CreateShaderResourceView(widgetRenderTarget_.Get(), nullptr, &widgetSRV_);
    if (FAILED(hr)) return false;

    // Create depth buffer for widget
    D3D11_TEXTURE2D_DESC dtd = {};
    dtd.Width = width;
    dtd.Height = height;
    dtd.MipLevels = 1;
    dtd.ArraySize = 1;
    dtd.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    dtd.SampleDesc.Count = 1;
    dtd.Usage = D3D11_USAGE_DEFAULT;
    dtd.BindFlags = D3D11_BIND_DEPTH_STENCIL;

    hr = device_->CreateTexture2D(&dtd, nullptr, &widgetDepthBuffer_);
    if (FAILED(hr)) return false;

    hr = device_->CreateDepthStencilView(widgetDepthBuffer_.Get(), nullptr, &widgetDSV_);
    if (FAILED(hr)) return false;

    widgetTargetWidth_ = width;
    widgetTargetHeight_ = height;

    return true;
}

void* SlamViewerImpl::getImGuiTexture() const {
    return widgetSRV_.Get();
}

void SlamViewerImpl::setConfig(const ViewerConfig& config) {
    config_ = config;
    // Update colormap texture if changed
}

void SlamViewerImpl::updatePointCloud(const PointData* points, size_t count) {
    // Write to the current write buffer (lock-free double buffering)
    int wb = writeBuffer_.load();
    auto& buffer = pointBuffers_[wb];

    buffer.points.resize(count);
    if (count > 0) {
        std::memcpy(buffer.points.data(), points, count * sizeof(PointData));
    }
    buffer.count = count;

    // Build chunks for frustum culling
    buffer.chunks.clear();
    if (count > 0 && frustumCullingEnabled_) {
        size_t numChunks = (count + CHUNK_SIZE - 1) / CHUNK_SIZE;
        buffer.chunks.reserve(numChunks);

        for (size_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
            PointChunk chunk;
            chunk.startIndex = chunkIdx * CHUNK_SIZE;
            chunk.count = std::min(CHUNK_SIZE, count - chunk.startIndex);

            // Compute chunk bounds
            const PointData& firstPt = points[chunk.startIndex];
            chunk.minBound = V3F(firstPt.x, firstPt.y, firstPt.z);
            chunk.maxBound = chunk.minBound;

            for (size_t i = 1; i < chunk.count; i++) {
                const PointData& pt = points[chunk.startIndex + i];
                chunk.minBound.x() = std::min(chunk.minBound.x(), pt.x);
                chunk.minBound.y() = std::min(chunk.minBound.y(), pt.y);
                chunk.minBound.z() = std::min(chunk.minBound.z(), pt.z);
                chunk.maxBound.x() = std::max(chunk.maxBound.x(), pt.x);
                chunk.maxBound.y() = std::max(chunk.maxBound.y(), pt.y);
                chunk.maxBound.z() = std::max(chunk.maxBound.z(), pt.z);
            }

            buffer.chunks.push_back(chunk);
        }
        buffer.chunksValid = true;
    } else {
        buffer.chunksValid = false;
    }

    buffer.ready.store(true);

    // Swap buffers
    writeBuffer_.store(1 - wb);
    pointsUpdated_.store(true);

    // Update global bounds
    if (count > 0) {
        boundsMin_ = V3F(points[0].x, points[0].y, points[0].z);
        boundsMax_ = boundsMin_;
        for (size_t i = 1; i < count; i++) {
            boundsMin_.x() = std::min(boundsMin_.x(), points[i].x);
            boundsMin_.y() = std::min(boundsMin_.y(), points[i].y);
            boundsMin_.z() = std::min(boundsMin_.z(), points[i].z);
            boundsMax_.x() = std::max(boundsMax_.x(), points[i].x);
            boundsMax_.y() = std::max(boundsMax_.y(), points[i].y);
            boundsMax_.z() = std::max(boundsMax_.z(), points[i].z);
        }
    }
}

void SlamViewerImpl::appendPointCloud(const PointData* points, size_t count) {
    std::lock_guard<std::mutex> lock(pointMutex_);

    int wb = writeBuffer_.load();
    auto& buffer = pointBuffers_[wb];

    size_t oldCount = buffer.count;
    buffer.points.resize(oldCount + count);
    std::memcpy(buffer.points.data() + oldCount, points, count * sizeof(PointData));
    buffer.count = oldCount + count;
    buffer.ready.store(true);

    pointsUpdated_.store(true);
}

void SlamViewerImpl::clearPointCloud() {
    for (auto& buffer : pointBuffers_) {
        buffer.points.clear();
        buffer.count = 0;
        buffer.ready.store(false);
    }
    pointsUpdated_.store(true);
}

void SlamViewerImpl::updateOverlayPointCloud(const PointData* points, size_t count) {
    if (!points || count == 0) return;

    int wb = overlayWriteBuffer_.load();
    PointBuffer& buf = overlayBuffers_[wb];

    buf.points.resize(count);
    std::memcpy(buf.points.data(), points, count * sizeof(PointData));
    buf.count = count;
    buf.ready.store(true);

    // Swap buffers
    int nextWrite = 1 - wb;
    overlayWriteBuffer_.store(nextWrite);
    overlayReadBuffer_.store(wb);
    overlayUpdated_.store(true);
}

void SlamViewerImpl::setOverlayColorTint(float r, float g, float b) {
    overlayColorTint_ = V3F(r, g, b);
}

void SlamViewerImpl::clearOverlayPointCloud() {
    for (auto& buffer : overlayBuffers_) {
        buffer.points.clear();
        buffer.count = 0;
        buffer.ready.store(false);
    }
    overlayPointCount_ = 0;
    overlayUpdated_.store(true);
}

void SlamViewerImpl::updatePose(const M4D& pose, uint64_t timestamp_ns) {
    {
        std::lock_guard<std::mutex> lock(poseMutex_);
        currentPose_ = pose;
    }
    poseUpdated_.store(true);

    // In localization mode, update coverage
    if (mode_ == ViewMode::LOCALIZATION && coverageInitialized_) {
        V3D probePos = pose.block<3, 1>(0, 3) +
                       pose.block<3, 3>(0, 0) * config_.probe.lidar_to_probe_offset;
        V3D probeAxis = pose.block<3, 3>(0, 0) * config_.probe.probe_scan_axis;

        coverageGrid_.markCovered(probePos.cast<float>(), probeAxis.cast<float>(),
                                  static_cast<float>(config_.probe.scan_width));
        coverageUpdated_.store(true);
    }
}

bool SlamViewerImpl::loadHullMesh(const std::string& filename) {
    // Simple OBJ loader
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    meshVertices_.clear();
    meshNormals_.clear();
    meshUVs_.clear();
    meshIndices_.clear();

    std::vector<V3F> tempVertices;
    std::vector<V3F> tempNormals;
    std::vector<V2F> tempUVs;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            tempVertices.emplace_back(x, y, z);
        } else if (type == "vn") {
            float x, y, z;
            iss >> x >> y >> z;
            tempNormals.emplace_back(x, y, z);
        } else if (type == "vt") {
            float u, v;
            iss >> u >> v;
            tempUVs.emplace_back(u, v);
        } else if (type == "f") {
            std::string vertex;
            std::vector<uint32_t> faceIndices;
            while (iss >> vertex) {
                int vi = 0, ti = 0, ni = 0;
                sscanf(vertex.c_str(), "%d/%d/%d", &vi, &ti, &ni);
                if (vi == 0) sscanf(vertex.c_str(), "%d//%d", &vi, &ni);
                if (vi == 0) sscanf(vertex.c_str(), "%d", &vi);

                // OBJ indices are 1-based
                uint32_t idx = static_cast<uint32_t>(meshVertices_.size());
                meshVertices_.push_back(tempVertices[vi - 1]);
                if (ni > 0 && ni <= static_cast<int>(tempNormals.size())) {
                    meshNormals_.push_back(tempNormals[ni - 1]);
                } else {
                    meshNormals_.push_back(V3F(0, 1, 0));  // Default normal
                }
                if (ti > 0 && ti <= static_cast<int>(tempUVs.size())) {
                    meshUVs_.push_back(tempUVs[ti - 1]);
                } else {
                    meshUVs_.push_back(V2F(0, 0));
                }
                faceIndices.push_back(idx);
            }

            // Triangulate face
            for (size_t i = 2; i < faceIndices.size(); i++) {
                meshIndices_.push_back(faceIndices[0]);
                meshIndices_.push_back(faceIndices[i - 1]);
                meshIndices_.push_back(faceIndices[i]);
            }
        }
    }

    if (meshVertices_.empty()) return false;

    // Create GPU vertex buffer for mesh
    struct MeshVertex {
        V3F position;
        V3F normal;
        V2F uv;
    };
    std::vector<MeshVertex> meshData(meshVertices_.size());
    for (size_t i = 0; i < meshVertices_.size(); i++) {
        meshData[i].position = meshVertices_[i];
        meshData[i].normal = meshNormals_[i];
        meshData[i].uv = meshUVs_[i];
    }

    D3D11_BUFFER_DESC vbd = {};
    vbd.Usage = D3D11_USAGE_DEFAULT;
    vbd.ByteWidth = static_cast<UINT>(meshData.size() * sizeof(MeshVertex));
    vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;

    D3D11_SUBRESOURCE_DATA vinitData = {};
    vinitData.pSysMem = meshData.data();
    device_->CreateBuffer(&vbd, &vinitData, &meshVertexBuffer_);

    // Create index buffer
    D3D11_BUFFER_DESC ibd = {};
    ibd.Usage = D3D11_USAGE_DEFAULT;
    ibd.ByteWidth = static_cast<UINT>(meshIndices_.size() * sizeof(uint32_t));
    ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;

    D3D11_SUBRESOURCE_DATA iinitData = {};
    iinitData.pSysMem = meshIndices_.data();
    device_->CreateBuffer(&ibd, &iinitData, &meshIndexBuffer_);

    // Create coverage texture (256x256 for now)
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = 256;
    td.Height = 256;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R32_FLOAT;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DYNAMIC;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    device_->CreateTexture2D(&td, nullptr, &coverageTexture_);

    D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {};
    srvd.Format = DXGI_FORMAT_R32_FLOAT;
    srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvd.Texture2D.MipLevels = 1;
    device_->CreateShaderResourceView(coverageTexture_.Get(), &srvd, &coverageSRV_);

    meshLoaded_ = true;
    return true;
}

void SlamViewerImpl::clearHullMesh() {
    meshVertices_.clear();
    meshNormals_.clear();
    meshUVs_.clear();
    meshIndices_.clear();
    meshLoaded_ = false;
}

void SlamViewerImpl::initCoverageGrid() {
    if (!meshLoaded_) return;

    coverageGrid_.initFromMesh(meshVertices_, meshNormals_, meshIndices_,
                               meshUVs_, config_.coverage.cell_size);
    coverageInitialized_ = true;
}

void SlamViewerImpl::clearCoverage() {
    coverageGrid_.clear();
    coverageUpdated_.store(true);
}

float SlamViewerImpl::getCoveragePercent() const {
    return coverageGrid_.getCoveragePercent();
}

void SlamViewerImpl::uploadPointsToGPU() {
    // Read from the buffer that's not being written to
    int rb = 1 - writeBuffer_.load();
    auto& buffer = pointBuffers_[rb];

    if (!buffer.ready.load() || buffer.count == 0) {
        stats_.total_points = 0;
        stats_.visible_points = 0;
        return;
    }

    const size_t maxPoints = config_.point_cloud.max_visible_points;
    const bool lodEnabled = config_.point_cloud.enable_lod;
    const float lodDistance = config_.point_cloud.lod_distance;

    // Structure to hold visible chunk info with LOD
    struct ChunkLOD {
        size_t index;
        size_t decimation;
        float distance;
    };
    std::vector<ChunkLOD> visibleChunks;
    size_t estimatedPoints = 0;

    if (frustumCullingEnabled_ && buffer.chunksValid && !buffer.chunks.empty()) {
        visibleChunks.reserve(buffer.chunks.size());

        for (size_t i = 0; i < buffer.chunks.size(); i++) {
            const PointChunk& chunk = buffer.chunks[i];

            // Frustum cull
            if (!frustum_.testAABB(chunk.minBound, chunk.maxBound)) {
                continue;
            }

            // Calculate distance from camera to chunk center
            V3F chunkCenter = (chunk.minBound + chunk.maxBound) * 0.5f;
            float dist = (chunkCenter - cameraPositionForLOD_).norm();

            // Determine per-chunk decimation based on distance
            size_t chunkDecimation = 1;
            if (lodEnabled && lodDistance > 0) {
                if (dist < lodDistance) {
                    chunkDecimation = 1;       // Full detail
                } else if (dist < lodDistance * 3.0f) {
                    chunkDecimation = 2;       // Half detail
                } else if (dist < lodDistance * 6.0f) {
                    chunkDecimation = 4;       // Quarter detail
                } else {
                    chunkDecimation = 8;       // Eighth detail
                }
            }

            visibleChunks.push_back({i, chunkDecimation, dist});
            estimatedPoints += (chunk.count + chunkDecimation - 1) / chunkDecimation;
        }
    } else {
        // No frustum culling - process all points with uniform decimation
        estimatedPoints = buffer.count;
    }

    // Apply global decimation if still over budget
    size_t globalDecimation = 1;
    if (estimatedPoints > maxPoints && maxPoints > 0) {
        globalDecimation = (estimatedPoints + maxPoints - 1) / maxPoints;
    }

    size_t maxUpload = std::min(maxPoints, size_t(30000000));
    size_t uploadedCount = 0;
    const PointData* src = buffer.points.data();

    // Choose upload path based on rendering mode
    if (useInstancing_ && pointStructuredBuffer_) {
        // Upload to structured buffer for instanced rendering
        D3D11_MAPPED_SUBRESOURCE mappedSB;
        if (SUCCEEDED(context_->Map(pointStructuredBuffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedSB))) {
            GPUPointData* gpuDest = static_cast<GPUPointData*>(mappedSB.pData);
            size_t destIdx = 0;

            if (!visibleChunks.empty()) {
                for (const auto& vc : visibleChunks) {
                    const PointChunk& chunk = buffer.chunks[vc.index];
                    size_t totalDecimation = vc.decimation * globalDecimation;

                    for (size_t i = 0; i < chunk.count && destIdx < maxUpload; i += totalDecimation) {
                        const PointData& pt = src[chunk.startIndex + i];
                        gpuDest[destIdx].x = pt.x;
                        gpuDest[destIdx].y = pt.y;
                        gpuDest[destIdx].z = pt.z;
                        gpuDest[destIdx].intensity = static_cast<float>(pt.intensity);
                        destIdx++;
                    }
                }
            } else {
                size_t decimation = globalDecimation;
                if (lodEnabled) decimation = std::max(decimation, size_t(2));

                for (size_t i = 0; i < buffer.count && destIdx < maxUpload; i += decimation) {
                    const PointData& pt = src[i];
                    gpuDest[destIdx].x = pt.x;
                    gpuDest[destIdx].y = pt.y;
                    gpuDest[destIdx].z = pt.z;
                    gpuDest[destIdx].intensity = static_cast<float>(pt.intensity);
                    destIdx++;
                }
            }

            uploadedCount = destIdx;
            context_->Unmap(pointStructuredBuffer_.Get(), 0);
        }
    } else {
        // Upload to vertex buffer for geometry shader path
        D3D11_MAPPED_SUBRESOURCE mapped;
        if (SUCCEEDED(context_->Map(pointVertexBuffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            PointData* dest = static_cast<PointData*>(mapped.pData);
            size_t destIdx = 0;

            if (!visibleChunks.empty()) {
                for (const auto& vc : visibleChunks) {
                    const PointChunk& chunk = buffer.chunks[vc.index];
                    size_t totalDecimation = vc.decimation * globalDecimation;

                    for (size_t i = 0; i < chunk.count && destIdx < maxUpload; i += totalDecimation) {
                        dest[destIdx++] = src[chunk.startIndex + i];
                    }
                }
            } else {
                size_t decimation = globalDecimation;
                if (lodEnabled) decimation = std::max(decimation, size_t(2));

                for (size_t i = 0; i < buffer.count && destIdx < maxUpload; i += decimation) {
                    dest[destIdx++] = src[i];
                }
            }

            uploadedCount = destIdx;
            context_->Unmap(pointVertexBuffer_.Get(), 0);
        }
    }

    stats_.total_points = buffer.count;
    stats_.visible_points = uploadedCount;
    pointsUpdated_.store(false);
}

void SlamViewerImpl::renderPointCloud() {
    if (stats_.visible_points == 0) return;

    // Update camera constants
    float aspect = static_cast<float>(viewportWidth_) / static_cast<float>(viewportHeight_);
    M4F view = camera_.getViewMatrix();
    M4F proj = camera_.getProjectionMatrix(aspect, config_.camera_fov,
                                            config_.camera_near, config_.camera_far);

    CameraConstants cc;
    cc.viewProj = proj * view;
    cc.cameraPos = camera_.getPosition();
    cc.pointSize = config_.point_cloud.point_size;
    cc.screenSize = V2F(static_cast<float>(viewportWidth_),
                        static_cast<float>(viewportHeight_));

    D3D11_MAPPED_SUBRESOURCE mapped;
    context_->Map(cameraConstantBuffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    std::memcpy(mapped.pData, &cc, sizeof(cc));
    context_->Unmap(cameraConstantBuffer_.Get(), 0);

    // Choose rendering path
    if (useInstancing_ && pointVS_ && pointStructuredBuffer_) {
        // === Instanced rendering path (faster) ===
        context_->IASetInputLayout(pointLayout_.Get());
        context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        UINT stride = sizeof(float) * 2;  // Quad vertex (2 floats)
        UINT offset = 0;
        context_->IASetVertexBuffers(0, 1, quadVertexBuffer_.GetAddressOf(), &stride, &offset);
        context_->IASetIndexBuffer(quadIndexBuffer_.Get(), DXGI_FORMAT_R16_UINT, 0);

        context_->VSSetShader(pointVS_.Get(), nullptr, 0);
        context_->VSSetConstantBuffers(0, 1, cameraConstantBuffer_.GetAddressOf());
        context_->VSSetShaderResources(1, 1, pointBufferSRV_.GetAddressOf());  // t1 for point data

        context_->GSSetShader(nullptr, nullptr, 0);  // No geometry shader

        context_->PSSetShader(pointPS_.Get(), nullptr, 0);
        context_->PSSetShaderResources(0, 1, colormapSRV_.GetAddressOf());
        context_->PSSetSamplers(0, 1, colormapSampler_.GetAddressOf());

        // Draw indexed instanced: 6 indices per quad, N instances
        context_->DrawIndexedInstanced(6, static_cast<UINT>(stats_.visible_points), 0, 0, 0);

        // Unbind SRV from VS
        ID3D11ShaderResourceView* nullSRV = nullptr;
        context_->VSSetShaderResources(1, 1, &nullSRV);
    } else {
        // === Geometry shader fallback path ===
        context_->IASetInputLayout(pointLayout_GS_.Get());
        context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

        UINT stride = sizeof(PointData);
        UINT offset = 0;
        context_->IASetVertexBuffers(0, 1, pointVertexBuffer_.GetAddressOf(), &stride, &offset);

        context_->VSSetShader(pointVS_GS_.Get(), nullptr, 0);
        context_->VSSetConstantBuffers(0, 1, cameraConstantBuffer_.GetAddressOf());

        context_->GSSetShader(pointGS_.Get(), nullptr, 0);
        context_->GSSetConstantBuffers(0, 1, cameraConstantBuffer_.GetAddressOf());

        context_->PSSetShader(pointPS_.Get(), nullptr, 0);
        context_->PSSetShaderResources(0, 1, colormapSRV_.GetAddressOf());
        context_->PSSetSamplers(0, 1, colormapSampler_.GetAddressOf());

        // Draw points (geometry shader expands to quads)
        context_->Draw(static_cast<UINT>(stats_.visible_points), 0);
    }
}

void SlamViewerImpl::renderOverlayPointCloud() {
    if (!overlayVertexBuffer_ || overlayPointCount_ == 0) return;

    // Use geometry shader path for overlay (simpler, works with any point count)
    // Render with a different colormap tint to distinguish from main cloud

    float aspect = static_cast<float>(viewportWidth_) / static_cast<float>(viewportHeight_);
    M4F view = camera_.getViewMatrix();
    M4F proj = camera_.getProjectionMatrix(aspect, config_.camera_fov,
                                            config_.camera_near, config_.camera_far);

    CameraConstants cb;
    cb.viewProj = proj * view;
    cb.cameraPos = camera_.getPosition();
    cb.pointSize = config_.point_cloud.point_size * 1.5f;  // Slightly larger for overlay
    cb.screenSize = V2F(static_cast<float>(viewportWidth_), static_cast<float>(viewportHeight_));

    D3D11_MAPPED_SUBRESOURCE mapped;
    if (SUCCEEDED(context_->Map(cameraConstantBuffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        memcpy(mapped.pData, &cb, sizeof(cb));
        context_->Unmap(cameraConstantBuffer_.Get(), 0);
    }

    // Set shaders (GS path)
    context_->VSSetShader(pointVS_GS_.Get(), nullptr, 0);
    context_->VSSetConstantBuffers(0, 1, cameraConstantBuffer_.GetAddressOf());
    context_->GSSetShader(pointGS_.Get(), nullptr, 0);
    context_->GSSetConstantBuffers(0, 1, cameraConstantBuffer_.GetAddressOf());

    // Create a tinted colormap texture for overlay (use a solid color based on tint)
    // For simplicity, we'll just render with the same colormap but the points will
    // appear different because the overlay is typically the local map (different geometry)
    context_->PSSetShader(pointPS_.Get(), nullptr, 0);
    context_->PSSetShaderResources(0, 1, colormapSRV_.GetAddressOf());
    context_->PSSetSamplers(0, 1, colormapSampler_.GetAddressOf());

    // Bind overlay vertex buffer
    UINT stride = sizeof(PointData);
    UINT offset = 0;
    context_->IASetVertexBuffers(0, 1, overlayVertexBuffer_.GetAddressOf(), &stride, &offset);
    context_->IASetInputLayout(pointLayout_GS_.Get());
    context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);

    // Draw overlay points
    context_->Draw(static_cast<UINT>(overlayPointCount_), 0);

    // Clear geometry shader
    context_->GSSetShader(nullptr, nullptr, 0);
}

void SlamViewerImpl::renderMesh() {
    if (!meshLoaded_ || !meshVS_ || !meshPS_ || !meshVertexBuffer_) return;

    // Update coverage texture if needed
    if (coverageUpdated_.load()) {
        std::vector<float> coverageData;
        int texWidth, texHeight;
        coverageGrid_.getCoverageTexture(coverageData, texWidth, texHeight);

        D3D11_MAPPED_SUBRESOURCE mapped;
        if (SUCCEEDED(context_->Map(coverageTexture_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            for (int row = 0; row < texHeight; row++) {
                memcpy(static_cast<char*>(mapped.pData) + row * mapped.RowPitch,
                       coverageData.data() + row * texWidth,
                       texWidth * sizeof(float));
            }
            context_->Unmap(coverageTexture_.Get(), 0);
        }
        coverageUpdated_.store(false);
    }

    // Update mesh constants
    float aspect = static_cast<float>(viewportWidth_) / static_cast<float>(viewportHeight_);
    M4F view = camera_.getViewMatrix();
    M4F proj = camera_.getProjectionMatrix(aspect, config_.camera_fov,
                                            config_.camera_near, config_.camera_far);

    MeshConstants mc;
    mc.viewProj = proj * view;
    mc.world = M4F::Identity();
    mc.cameraPos = camera_.getPosition();
    mc.padding = 0.0f;

    D3D11_MAPPED_SUBRESOURCE mapped;
    context_->Map(meshConstantBuffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    std::memcpy(mapped.pData, &mc, sizeof(mc));
    context_->Unmap(meshConstantBuffer_.Get(), 0);

    // Update material constants
    MaterialConstants mat;
    mat.baseColor = config_.mesh.color;
    mat.coveredColor = config_.coverage.covered_color;
    mat.probeColor = config_.coverage.probe_color;
    mat.coverageTexSize = V2F(256.0f, 256.0f);

    context_->Map(materialConstantBuffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    std::memcpy(mapped.pData, &mat, sizeof(mat));
    context_->Unmap(materialConstantBuffer_.Get(), 0);

    // Set pipeline state
    context_->IASetInputLayout(meshLayout_.Get());
    context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    UINT stride = sizeof(V3F) * 2 + sizeof(V2F);  // position + normal + uv
    UINT offset = 0;
    context_->IASetVertexBuffers(0, 1, meshVertexBuffer_.GetAddressOf(), &stride, &offset);
    context_->IASetIndexBuffer(meshIndexBuffer_.Get(), DXGI_FORMAT_R32_UINT, 0);

    context_->VSSetShader(meshVS_.Get(), nullptr, 0);
    context_->VSSetConstantBuffers(0, 1, meshConstantBuffer_.GetAddressOf());

    context_->GSSetShader(nullptr, nullptr, 0);  // No geometry shader for mesh

    context_->PSSetShader(meshPS_.Get(), nullptr, 0);
    context_->PSSetConstantBuffers(1, 1, materialConstantBuffer_.GetAddressOf());
    context_->PSSetShaderResources(0, 1, coverageSRV_.GetAddressOf());
    context_->PSSetSamplers(0, 1, colormapSampler_.GetAddressOf());

    // Draw mesh
    context_->DrawIndexed(static_cast<UINT>(meshIndices_.size()), 0, 0);
}

void SlamViewerImpl::renderProbe() {
    if (!config_.coverage.show_probe_footprint) return;

    // Get current probe position and orientation from pose
    M4D pose;
    {
        std::lock_guard<std::mutex> lock(poseMutex_);
        pose = currentPose_;
    }

    // Calculate probe center and endpoints
    V3D lidarPos = pose.block<3, 1>(0, 3);
    Eigen::Matrix3d rot = pose.block<3, 3>(0, 0);
    V3D probeCenter = lidarPos + rot * config_.probe.lidar_to_probe_offset;
    V3D probeAxis = rot * config_.probe.probe_scan_axis;

    float halfWidth = static_cast<float>(config_.probe.scan_width) * 0.5f;
    V3F p1 = (probeCenter - probeAxis * halfWidth).cast<float>();
    V3F p2 = (probeCenter + probeAxis * halfWidth).cast<float>();

    // Create line vertices (probe line as thick line using quads)
    float lineWidth = 0.01f;  // 1cm thick line
    V3F up(0, 1, 0);  // Approximate up vector
    V3F lineDir = (p2 - p1).normalized();
    V3F perp = lineDir.cross(up).normalized() * lineWidth;

    // Quad vertices for probe line
    V3F lineVerts[4] = {
        p1 - perp, p1 + perp, p2 - perp, p2 + perp
    };

    // Create/update probe vertex buffer if needed
    if (!probeVertexBuffer_) {
        D3D11_BUFFER_DESC vbd = {};
        vbd.Usage = D3D11_USAGE_DYNAMIC;
        vbd.ByteWidth = sizeof(V3F) * 4;
        vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        device_->CreateBuffer(&vbd, nullptr, &probeVertexBuffer_);
    }

    // Upload probe vertices
    D3D11_MAPPED_SUBRESOURCE mapped;
    if (SUCCEEDED(context_->Map(probeVertexBuffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        std::memcpy(mapped.pData, lineVerts, sizeof(lineVerts));
        context_->Unmap(probeVertexBuffer_.Get(), 0);
    }

    // Use simple colored shader for probe
    // For now, reuse mesh shader with identity transform and solid color
    float aspect = static_cast<float>(viewportWidth_) / static_cast<float>(viewportHeight_);
    M4F view = camera_.getViewMatrix();
    M4F proj = camera_.getProjectionMatrix(aspect, config_.camera_fov,
                                            config_.camera_near, config_.camera_far);

    MeshConstants mc;
    mc.viewProj = proj * view;
    mc.world = M4F::Identity();
    mc.cameraPos = camera_.getPosition();

    context_->Map(meshConstantBuffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    std::memcpy(mapped.pData, &mc, sizeof(mc));
    context_->Unmap(meshConstantBuffer_.Get(), 0);

    // Set probe color (orange by default)
    MaterialConstants mat;
    mat.baseColor = config_.coverage.probe_color;
    mat.coveredColor = config_.coverage.probe_color;

    context_->Map(materialConstantBuffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    std::memcpy(mapped.pData, &mat, sizeof(mat));
    context_->Unmap(materialConstantBuffer_.Get(), 0);

    // Draw probe as triangle strip
    context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    UINT stride = sizeof(V3F);
    UINT offset = 0;
    context_->IASetVertexBuffers(0, 1, probeVertexBuffer_.GetAddressOf(), &stride, &offset);

    // Use a simple vertex shader that just transforms position
    context_->VSSetShader(meshVS_.Get(), nullptr, 0);
    context_->VSSetConstantBuffers(0, 1, meshConstantBuffer_.GetAddressOf());
    context_->GSSetShader(nullptr, nullptr, 0);
    context_->PSSetShader(meshPS_.Get(), nullptr, 0);
    context_->PSSetConstantBuffers(1, 1, materialConstantBuffer_.GetAddressOf());

    context_->Draw(4, 0);
}

void SlamViewerImpl::beginFrame() {
    // Clear render target
    float clearColor[4] = {config_.background_color.x(), config_.background_color.y(),
                           config_.background_color.z(), config_.background_color.w()};
    context_->ClearRenderTargetView(renderTargetView_.Get(), clearColor);
    context_->ClearDepthStencilView(depthStencilView_.Get(),
                                    D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

    // Set render targets
    context_->OMSetRenderTargets(1, renderTargetView_.GetAddressOf(), depthStencilView_.Get());
    context_->OMSetDepthStencilState(depthStencilState_.Get(), 0);
    context_->RSSetState(rasterizerState_.Get());
    context_->OMSetBlendState(blendState_.Get(), nullptr, 0xFFFFFFFF);

    // Set viewport
    D3D11_VIEWPORT vp = {};
    vp.Width = static_cast<float>(viewportWidth_);
    vp.Height = static_cast<float>(viewportHeight_);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    context_->RSSetViewports(1, &vp);
}

void SlamViewerImpl::endFrame() {
    if (swapChain_) {
        swapChain_->Present(1, 0);  // VSync
    }
}

void SlamViewerImpl::renderWidget(float width, float height) {
    // Get ImGui draw area
    ImVec2 size = ImGui::GetContentRegionAvail();
    if (width > 0) size.x = width;
    if (height > 0) size.y = height;

    // Clamp to reasonable size
    size.x = std::max(size.x, 64.0f);
    size.y = std::max(size.y, 64.0f);

    viewportWidth_ = static_cast<int>(size.x);
    viewportHeight_ = static_cast<int>(size.y);

    // Create/resize render target if needed
    if (!createWidgetRenderTarget(viewportWidth_, viewportHeight_)) {
        ImGui::TextColored(ImVec4(1,0,0,1), "Failed to create render target");
        return;
    }

    // Cache camera position for LOD calculations
    cameraPositionForLOD_ = camera_.getPosition();

    // Update frustum for culling
    if (frustumCullingEnabled_) {
        float aspect = static_cast<float>(viewportWidth_) / static_cast<float>(viewportHeight_);
        M4F view = camera_.getViewMatrix();
        M4F proj = camera_.getProjectionMatrix(aspect, config_.camera_fov,
                                                config_.camera_near, config_.camera_far);
        frustum_.update(proj * view);
    }

    // Upload data to GPU
    uploadPointsToGPU();

    // Upload overlay points if updated
    if (overlayUpdated_.load()) {
        int rb = overlayReadBuffer_.load();
        const PointBuffer& buf = overlayBuffers_[rb];
        if (buf.ready.load() && buf.count > 0) {
            // Create/resize overlay vertex buffer if needed
            size_t neededSize = buf.count * sizeof(PointData);
            if (!overlayVertexBuffer_ || overlayPointCount_ < buf.count) {
                overlayVertexBuffer_.Reset();
                D3D11_BUFFER_DESC vbd = {};
                vbd.Usage = D3D11_USAGE_DYNAMIC;
                vbd.ByteWidth = static_cast<UINT>(neededSize);
                vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
                vbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
                device_->CreateBuffer(&vbd, nullptr, &overlayVertexBuffer_);
            }

            // Upload overlay points
            D3D11_MAPPED_SUBRESOURCE mapped;
            if (SUCCEEDED(context_->Map(overlayVertexBuffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
                std::memcpy(mapped.pData, buf.points.data(), buf.count * sizeof(PointData));
                context_->Unmap(overlayVertexBuffer_.Get(), 0);
            }
            overlayPointCount_ = buf.count;
        }
        overlayUpdated_.store(false);
    }

    // Save current render target
    ComPtr<ID3D11RenderTargetView> savedRTV;
    ComPtr<ID3D11DepthStencilView> savedDSV;
    context_->OMGetRenderTargets(1, &savedRTV, &savedDSV);

    // Set widget render target
    float clearColor[4] = {config_.background_color.x(), config_.background_color.y(),
                           config_.background_color.z(), config_.background_color.w()};
    context_->ClearRenderTargetView(widgetRTV_.Get(), clearColor);
    context_->ClearDepthStencilView(widgetDSV_.Get(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
    context_->OMSetRenderTargets(1, widgetRTV_.GetAddressOf(), widgetDSV_.Get());
    context_->OMSetDepthStencilState(depthStencilState_.Get(), 0);
    context_->RSSetState(rasterizerState_.Get());
    context_->OMSetBlendState(blendState_.Get(), nullptr, 0xFFFFFFFF);

    // Set viewport
    D3D11_VIEWPORT vp = {};
    vp.Width = static_cast<float>(viewportWidth_);
    vp.Height = static_cast<float>(viewportHeight_);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    context_->RSSetViewports(1, &vp);

    // Render scene
    if (mode_ == ViewMode::SCANNING) {
        renderPointCloud();
        renderOverlayPointCloud();  // Render overlay on top
        renderRobotMarker();        // Render robot position marker
    } else {
        renderMesh();
        renderCoverage();
        renderProbe();
        renderOverlayPointCloud();  // Also show local map in localization mode
        renderRobotMarker();        // Render robot position marker
    }

    // Restore original render target
    context_->OMSetRenderTargets(1, savedRTV.GetAddressOf(), savedDSV.Get());

    // Display as ImGui image
    // Cast to ImTextureID (ImU64 in newer ImGui) - D3D11 backend uses SRV directly as texture ID
    ImGui::Image(reinterpret_cast<ImTextureID>(widgetSRV_.Get()), size);

    // Handle mouse input when image is hovered
    if (ImGui::IsItemHovered()) {
        ImGuiIO& io = ImGui::GetIO();

        // Get the image position for screen-to-world conversion
        ImVec2 imagePos = ImGui::GetItemRectMin();

        // Helper lambda to convert screen position to world XY (intersect with Z=0 plane)
        auto screenToWorld = [&](float sx, float sy, float& wx, float& wy) -> bool {
            // Convert to normalized device coordinates (-1 to 1)
            float ndcX = ((sx - imagePos.x) / size.x) * 2.0f - 1.0f;
            float ndcY = 1.0f - ((sy - imagePos.y) / size.y) * 2.0f;  // Flip Y

            // Get view and projection matrices
            float aspect = static_cast<float>(viewportWidth_) / static_cast<float>(viewportHeight_);
            M4F view = camera_.getViewMatrix();
            M4F proj = camera_.getProjectionMatrix(aspect, config_.camera_fov,
                                                    config_.camera_near, config_.camera_far);
            M4F invViewProj = (proj * view).inverse();

            // Unproject near and far points
            V4F nearPoint = invViewProj * V4F(ndcX, ndcY, 0.0f, 1.0f);
            V4F farPoint = invViewProj * V4F(ndcX, ndcY, 1.0f, 1.0f);
            nearPoint /= nearPoint.w();
            farPoint /= farPoint.w();

            // Ray from near to far
            V3F rayOrigin(nearPoint.x(), nearPoint.y(), nearPoint.z());
            V3F rayDir = (V3F(farPoint.x(), farPoint.y(), farPoint.z()) - rayOrigin).normalized();

            // Intersect with Z=0 plane
            if (std::abs(rayDir.z()) < 0.001f) return false;  // Ray parallel to ground
            float t = -rayOrigin.z() / rayDir.z();
            if (t < 0) return false;  // Intersection behind camera

            V3F intersection = rayOrigin + rayDir * t;
            wx = intersection.x();
            wy = intersection.y();
            return true;
        };

        // Click-to-pose mode handling
        if (clickToPoseMode_ && mapClickCallback_) {
            if (ImGui::IsMouseClicked(0)) {  // Left click - set position
                float wx, wy;
                if (screenToWorld(io.MousePos.x, io.MousePos.y, wx, wy)) {
                    clickDragActive_ = true;
                    clickStartWorldX_ = wx;
                    clickStartWorldY_ = wy;
                }
            }
            if (clickDragActive_ && ImGui::IsMouseReleased(0)) {
                float wx, wy;
                if (screenToWorld(io.MousePos.x, io.MousePos.y, wx, wy)) {
                    float dx = wx - clickStartWorldX_;
                    float dy = wy - clickStartWorldY_;
                    float dragDist = std::sqrt(dx * dx + dy * dy);

                    if (dragDist > 0.1f) {
                        // Drag was significant - compute heading from drag direction
                        float heading = std::atan2(dy, dx);
                        mapClickCallback_(clickStartWorldX_, clickStartWorldY_, heading);
                    } else {
                        // Just a click - use NaN for heading (let system use default)
                        mapClickCallback_(clickStartWorldX_, clickStartWorldY_, std::nanf(""));
                    }
                }
                clickDragActive_ = false;
            }
        } else {
            // Normal camera control mode
            // Match CPU viewer controls: Right=pan, Middle=rotate (or Left=rotate)
            if (io.MouseDown[0]) {  // Left button - rotate
                // Negate X so drag-left = rotate-left (conventional behavior)
                camera_.rotate(-io.MouseDelta.x * 0.01f, io.MouseDelta.y * 0.01f);
            }
            if (io.MouseDown[1]) {  // Right button - pan (matches CPU viewer)
                camera_.pan(-io.MouseDelta.x, io.MouseDelta.y);
            }
            if (io.MouseDown[2]) {  // Middle button - pan (also pan for consistency)
                camera_.pan(-io.MouseDelta.x, io.MouseDelta.y);
            }
        }

        if (io.MouseWheel != 0) {  // Scroll - zoom (always available)
            camera_.zoom(io.MouseWheel);
        }
    } else {
        // Mouse left the widget area - cancel any drag
        clickDragActive_ = false;
    }
}

bool SlamViewerImpl::renderStandalone() {
    if (!processMessages()) return false;

    // Cache camera position for LOD calculations
    cameraPositionForLOD_ = camera_.getPosition();

    // Update frustum for culling (must happen before upload)
    if (frustumCullingEnabled_) {
        float aspect = static_cast<float>(viewportWidth_) / static_cast<float>(viewportHeight_);
        M4F view = camera_.getViewMatrix();
        M4F proj = camera_.getProjectionMatrix(aspect, config_.camera_fov,
                                                config_.camera_near, config_.camera_far);
        frustum_.update(proj * view);
    }

    uploadPointsToGPU();

    beginFrame();

    if (mode_ == ViewMode::SCANNING) {
        renderPointCloud();
    } else {
        renderMesh();
        renderCoverage();
        renderProbe();
    }

    endFrame();

    return true;
}

bool SlamViewerImpl::processMessages() {
    MSG msg;
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) return false;
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return true;
}

void SlamViewerImpl::resetCamera() {
    camera_.setTarget(V3F::Zero());
    camera_.setDistance(5.0f);
    camera_.setRotation(0.0f, 0.3f);
}

void SlamViewerImpl::fitCameraToContent() {
    camera_.fitToBounds(boundsMin_, boundsMax_);
}

void SlamViewerImpl::renderCoverage() {
    // Coverage is rendered as part of the mesh shader by sampling the coverage texture
    // This function could be used for additional coverage visualization (e.g., grid lines)
    // but for now the texture-based approach in renderMesh() handles it
}

void SlamViewerImpl::setRobotPose(float x, float y, float heading) {
    robotX_ = x;
    robotY_ = y;
    // Use the minimum Z of the map bounds as the floor level
    // This ensures the robot marker is visible at the same level as the map
    robotZ_ = boundsMin_.z();
    robotHeading_ = heading;
    showRobot_ = true;
}

void SlamViewerImpl::centerCameraOnRobot() {
    if (!showRobot_) return;
    // Center camera on robot position at a reasonable distance
    V3F robotPos(robotX_, robotY_, robotZ_);
    camera_.setTarget(robotPos);
    // Set distance based on map size, but don't zoom too far in
    float mapDiagonal = (boundsMax_ - boundsMin_).norm();
    float viewDist = std::max(5.0f, mapDiagonal * 0.3f);
    camera_.setDistance(viewDist);
}

void SlamViewerImpl::renderRobotMarker() {
    if (!showRobot_) return;

    // Create robot triangle marker - pointing in heading direction
    // Scale robot size based on camera distance to keep it visible at all zoom levels
    // Base size is 1.0m which scales with zoom
    float camDist = camera_.getDistance();
    float scale = std::max(0.5f, camDist * 0.03f);  // 3% of camera distance, min 0.5m

    const float length = scale;
    const float width = scale * 0.7f;
    const float height = scale * 0.5f;  // Give it some height for visibility

    // Calculate triangle vertices in world space
    // Heading is in radians, 0 = +X direction, positive = counter-clockwise
    float cosH = std::cos(robotHeading_);
    float sinH = std::sin(robotHeading_);

    // Triangle pointing in heading direction:
    //   Front tip (in direction of heading)
    //   Back-left corner
    //   Back-right corner
    // Elevate above the ground for visibility
    float zBase = robotZ_ + height;
    V3F tip(robotX_ + length * cosH, robotY_ + length * sinH, zBase);
    V3F backLeft(robotX_ - length * 0.3f * cosH + width * 0.5f * sinH,
                 robotY_ - length * 0.3f * sinH - width * 0.5f * cosH,
                 zBase);
    V3F backRight(robotX_ - length * 0.3f * cosH - width * 0.5f * sinH,
                  robotY_ - length * 0.3f * sinH + width * 0.5f * cosH,
                  zBase);

    V3F robotVerts[3] = { tip, backLeft, backRight };

    // Create/update robot vertex buffer
    if (!robotVertexBuffer_) {
        D3D11_BUFFER_DESC vbd = {};
        vbd.Usage = D3D11_USAGE_DYNAMIC;
        vbd.ByteWidth = sizeof(V3F) * 3;
        vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        device_->CreateBuffer(&vbd, nullptr, &robotVertexBuffer_);
    }

    // Upload robot vertices
    D3D11_MAPPED_SUBRESOURCE mapped;
    if (SUCCEEDED(context_->Map(robotVertexBuffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
        std::memcpy(mapped.pData, robotVerts, sizeof(robotVerts));
        context_->Unmap(robotVertexBuffer_.Get(), 0);
    }

    // Render using mesh shader with a bright color (cyan for visibility)
    float aspect = static_cast<float>(viewportWidth_) / static_cast<float>(viewportHeight_);
    M4F view = camera_.getViewMatrix();
    M4F proj = camera_.getProjectionMatrix(aspect, config_.camera_fov,
                                            config_.camera_near, config_.camera_far);

    MeshConstants mc;
    mc.viewProj = proj * view;
    mc.world = M4F::Identity();
    mc.cameraPos = camera_.getPosition();

    context_->Map(meshConstantBuffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    std::memcpy(mapped.pData, &mc, sizeof(mc));
    context_->Unmap(meshConstantBuffer_.Get(), 0);

    // Set robot color - bright cyan for high visibility
    MaterialConstants mat;
    mat.baseColor = V4F(0.0f, 1.0f, 1.0f, 1.0f);  // Cyan
    mat.coveredColor = V4F(0.0f, 1.0f, 1.0f, 1.0f);

    context_->Map(materialConstantBuffer_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    std::memcpy(mapped.pData, &mat, sizeof(mat));
    context_->Unmap(materialConstantBuffer_.Get(), 0);

    // Draw robot triangle
    context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    UINT stride = sizeof(V3F);
    UINT offset = 0;
    context_->IASetVertexBuffers(0, 1, robotVertexBuffer_.GetAddressOf(), &stride, &offset);

    context_->VSSetShader(meshVS_.Get(), nullptr, 0);
    context_->VSSetConstantBuffers(0, 1, meshConstantBuffer_.GetAddressOf());
    context_->GSSetShader(nullptr, nullptr, 0);
    context_->PSSetShader(meshPS_.Get(), nullptr, 0);
    context_->PSSetConstantBuffers(1, 1, materialConstantBuffer_.GetAddressOf());

    context_->Draw(3, 0);
}

SlamViewer::RenderStats SlamViewerImpl::getStats() const {
    stats_.covered_cells = static_cast<size_t>(
        coverageGrid_.getCoveragePercent() * coverageGrid_.getCells().size() / 100.0f);
    stats_.total_cells = coverageGrid_.getCells().size();
    return stats_;
}

void SlamViewerImpl::setMapClickCallback(SlamViewer::MapClickCallback callback) {
    mapClickCallback_ = std::move(callback);
}

void SlamViewerImpl::setClickToPoseMode(bool enabled) {
    clickToPoseMode_ = enabled;
    clickDragActive_ = false;
}

LRESULT CALLBACK SlamViewerImpl::WndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    SlamViewerImpl* viewer = nullptr;

    if (msg == WM_CREATE) {
        CREATESTRUCT* cs = reinterpret_cast<CREATESTRUCT*>(lparam);
        viewer = static_cast<SlamViewerImpl*>(cs->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(viewer));
    } else {
        viewer = reinterpret_cast<SlamViewerImpl*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    }

    switch (msg) {
    case WM_SIZE:
        if (viewer && viewer->device_) {
            viewer->viewportWidth_ = LOWORD(lparam);
            viewer->viewportHeight_ = HIWORD(lparam);
            // Resize render targets would go here
        }
        break;

    case WM_LBUTTONDOWN:
        if (viewer) {
            viewer->mouseLeftDown_ = true;
            viewer->lastMouseX_ = LOWORD(lparam);
            viewer->lastMouseY_ = HIWORD(lparam);
            SetCapture(hwnd);
        }
        return 0;

    case WM_LBUTTONUP:
        if (viewer) {
            viewer->mouseLeftDown_ = false;
            ReleaseCapture();
        }
        return 0;

    case WM_RBUTTONDOWN:
        if (viewer) {
            viewer->mouseRightDown_ = true;
            viewer->lastMouseX_ = LOWORD(lparam);
            viewer->lastMouseY_ = HIWORD(lparam);
            SetCapture(hwnd);
        }
        return 0;

    case WM_RBUTTONUP:
        if (viewer) {
            viewer->mouseRightDown_ = false;
            ReleaseCapture();
        }
        return 0;

    case WM_MBUTTONDOWN:
        if (viewer) {
            viewer->mouseMiddleDown_ = true;
            viewer->lastMouseX_ = LOWORD(lparam);
            viewer->lastMouseY_ = HIWORD(lparam);
            SetCapture(hwnd);
        }
        return 0;

    case WM_MBUTTONUP:
        if (viewer) {
            viewer->mouseMiddleDown_ = false;
            ReleaseCapture();
        }
        return 0;

    case WM_MOUSEMOVE:
        if (viewer) {
            int x = LOWORD(lparam);
            int y = HIWORD(lparam);
            int dx = x - viewer->lastMouseX_;
            int dy = y - viewer->lastMouseY_;

            if (viewer->mouseLeftDown_) {
                // Left mouse: rotate camera
                viewer->camera_.rotate(-dx * 0.01f, dy * 0.01f);
            }
            if (viewer->mouseMiddleDown_ || viewer->mouseRightDown_) {
                // Middle/right mouse: pan camera
                viewer->camera_.pan(static_cast<float>(-dx), static_cast<float>(dy));
            }

            viewer->lastMouseX_ = x;
            viewer->lastMouseY_ = y;
        }
        return 0;

    case WM_MOUSEWHEEL:
        if (viewer) {
            int delta = GET_WHEEL_DELTA_WPARAM(wparam);
            viewer->camera_.zoom(static_cast<float>(delta) / 120.0f);
        }
        return 0;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    case WM_KEYDOWN:
        if (wparam == VK_ESCAPE) {
            PostQuitMessage(0);
            return 0;
        }
        // Reset camera with 'R' key
        if (wparam == 'R' && viewer) {
            viewer->resetCamera();
            return 0;
        }
        // Fit camera with 'F' key
        if (wparam == 'F' && viewer) {
            viewer->fitCameraToContent();
            return 0;
        }
        break;
    }
    return DefWindowProc(hwnd, msg, wparam, lparam);
}

} // namespace viz
} // namespace slam
