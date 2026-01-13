#pragma once
/**
 * @file probe_coverage.hpp
 * @brief PAUT probe coverage tracking on hull mesh
 *
 * Tracks which areas of the hull have been inspected by the PAUT probe
 * as the robot moves across the surface.
 */

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>

namespace slam_gui {

// Triangle vertex
struct Vertex {
    float x, y, z;

    Vertex operator-(const Vertex& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    Vertex cross(const Vertex& other) const {
        return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        };
    }

    float dot(const Vertex& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    float length() const {
        return std::sqrt(x*x + y*y + z*z);
    }

    Vertex normalized() const {
        float len = length();
        if (len < 1e-6f) return {0, 0, 0};
        return {x/len, y/len, z/len};
    }
};

// Triangle in hull mesh
struct Triangle {
    uint32_t v0, v1, v2;  // Vertex indices
    float area;           // Pre-computed area
    bool covered;         // Has probe passed over this
    float coverage_time;  // When it was covered (for visualization)
};

// PAUT probe configuration
struct PAUTProbeConfig {
    // Enable coverage tracking
    bool enabled = true;

    // Offset from robot center (in robot frame)
    float offset_x = -0.15f;  // Behind robot (negative = behind)
    float offset_y = 0.0f;    // Lateral offset (positive = left)
    float offset_z = -0.02f;  // Below robot center

    // Probe dimensions
    float width = 0.08f;      // Width of probe swath (meters)
    float length = 0.02f;     // Length in travel direction (meters)

    // Visualization
    bool show_probe = true;
    bool show_coverage = true;
    float coverage_opacity = 0.7f;
};

/**
 * Hull mesh with coverage tracking
 */
class HullCoverage {
public:
    HullCoverage() = default;

    /**
     * Load hull mesh from OBJ file
     */
    bool loadOBJ(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return false;

        vertices_.clear();
        triangles_.clear();

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;

            if (prefix == "v") {
                Vertex v;
                iss >> v.x >> v.y >> v.z;
                vertices_.push_back(v);
            } else if (prefix == "f") {
                // Parse face (handles v, v/vt, v/vt/vn, v//vn formats)
                std::vector<uint32_t> indices;
                std::string token;
                while (iss >> token) {
                    uint32_t idx = std::stoul(token.substr(0, token.find('/'))) - 1;
                    indices.push_back(idx);
                }

                // Triangulate if needed (fan triangulation)
                for (size_t i = 2; i < indices.size(); i++) {
                    Triangle tri;
                    tri.v0 = indices[0];
                    tri.v1 = indices[i-1];
                    tri.v2 = indices[i];
                    tri.covered = false;
                    tri.coverage_time = 0.0f;

                    // Compute area
                    const Vertex& a = vertices_[tri.v0];
                    const Vertex& b = vertices_[tri.v1];
                    const Vertex& c = vertices_[tri.v2];
                    Vertex ab = b - a;
                    Vertex ac = c - a;
                    tri.area = ab.cross(ac).length() * 0.5f;

                    triangles_.push_back(tri);
                }
            }
        }

        // Compute total area
        total_area_ = 0.0f;
        for (const auto& tri : triangles_) {
            total_area_ += tri.area;
        }

        filename_ = path;
        loaded_ = !vertices_.empty() && !triangles_.empty();
        return loaded_;
    }

    /**
     * Load hull mesh from STL file
     */
    bool loadSTL(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return false;

        vertices_.clear();
        triangles_.clear();

        // Read header (80 bytes)
        char header[80];
        file.read(header, 80);

        // Check if ASCII or binary
        std::string headerStr(header, 80);
        if (headerStr.find("solid") == 0) {
            // ASCII STL - reopen as text
            file.close();
            return loadSTLAscii(path);
        }

        // Binary STL
        uint32_t numTriangles;
        file.read(reinterpret_cast<char*>(&numTriangles), 4);

        for (uint32_t i = 0; i < numTriangles; i++) {
            float normal[3], v1[3], v2[3], v3[3];
            uint16_t attrib;

            file.read(reinterpret_cast<char*>(normal), 12);
            file.read(reinterpret_cast<char*>(v1), 12);
            file.read(reinterpret_cast<char*>(v2), 12);
            file.read(reinterpret_cast<char*>(v3), 12);
            file.read(reinterpret_cast<char*>(&attrib), 2);

            uint32_t idx = static_cast<uint32_t>(vertices_.size());
            vertices_.push_back({v1[0], v1[1], v1[2]});
            vertices_.push_back({v2[0], v2[1], v2[2]});
            vertices_.push_back({v3[0], v3[1], v3[2]});

            Triangle tri;
            tri.v0 = idx;
            tri.v1 = idx + 1;
            tri.v2 = idx + 2;
            tri.covered = false;
            tri.coverage_time = 0.0f;

            // Compute area
            Vertex ab = vertices_[tri.v1] - vertices_[tri.v0];
            Vertex ac = vertices_[tri.v2] - vertices_[tri.v0];
            tri.area = ab.cross(ac).length() * 0.5f;

            triangles_.push_back(tri);
        }

        total_area_ = 0.0f;
        for (const auto& tri : triangles_) {
            total_area_ += tri.area;
        }

        filename_ = path;
        loaded_ = !vertices_.empty();
        return loaded_;
    }

    bool loadSTLAscii(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return false;

        vertices_.clear();
        triangles_.clear();

        std::string line;
        Vertex v[3];
        int vcount = 0;

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string keyword;
            iss >> keyword;

            if (keyword == "vertex") {
                iss >> v[vcount].x >> v[vcount].y >> v[vcount].z;
                vcount++;

                if (vcount == 3) {
                    uint32_t idx = static_cast<uint32_t>(vertices_.size());
                    vertices_.push_back(v[0]);
                    vertices_.push_back(v[1]);
                    vertices_.push_back(v[2]);

                    Triangle tri;
                    tri.v0 = idx;
                    tri.v1 = idx + 1;
                    tri.v2 = idx + 2;
                    tri.covered = false;
                    tri.coverage_time = 0.0f;

                    Vertex ab = v[1] - v[0];
                    Vertex ac = v[2] - v[0];
                    tri.area = ab.cross(ac).length() * 0.5f;

                    triangles_.push_back(tri);
                    vcount = 0;
                }
            }
        }

        total_area_ = 0.0f;
        for (const auto& tri : triangles_) {
            total_area_ += tri.area;
        }

        filename_ = path;
        loaded_ = !vertices_.empty();
        return loaded_;
    }

    /**
     * Update coverage based on robot pose and probe config
     *
     * @param robot_x Robot X position (world frame)
     * @param robot_y Robot Y position (world frame)
     * @param robot_yaw Robot heading (radians)
     * @param time Current time for visualization
     */
    void updateCoverage(float robot_x, float robot_y, float robot_yaw,
                        const PAUTProbeConfig& probe, float time) {
        if (!loaded_) return;

        // Calculate probe center in world frame
        float cos_yaw = std::cos(robot_yaw);
        float sin_yaw = std::sin(robot_yaw);

        float probe_x = robot_x + probe.offset_x * cos_yaw - probe.offset_y * sin_yaw;
        float probe_y = robot_y + probe.offset_x * sin_yaw + probe.offset_y * cos_yaw;

        // Probe rectangle corners (in world frame)
        float half_w = probe.width * 0.5f;
        float half_l = probe.length * 0.5f;

        // Direction vectors
        float forward_x = cos_yaw;
        float forward_y = sin_yaw;
        float right_x = sin_yaw;
        float right_y = -cos_yaw;

        // Four corners of probe rectangle
        float corners[4][2] = {
            {probe_x - half_l * forward_x - half_w * right_x,
             probe_y - half_l * forward_y - half_w * right_y},
            {probe_x + half_l * forward_x - half_w * right_x,
             probe_y + half_l * forward_y - half_w * right_y},
            {probe_x + half_l * forward_x + half_w * right_x,
             probe_y + half_l * forward_y + half_w * right_y},
            {probe_x - half_l * forward_x + half_w * right_x,
             probe_y - half_l * forward_y + half_w * right_y}
        };

        // Check each triangle for intersection with probe rectangle
        for (auto& tri : triangles_) {
            if (tri.covered) continue;

            // Get triangle center (in XY plane)
            const Vertex& v0 = vertices_[tri.v0];
            const Vertex& v1 = vertices_[tri.v1];
            const Vertex& v2 = vertices_[tri.v2];

            float cx = (v0.x + v1.x + v2.x) / 3.0f;
            float cy = (v0.y + v1.y + v2.y) / 3.0f;

            // Simple check: is triangle center inside probe rectangle?
            // Using cross product method for point-in-polygon
            if (pointInQuad(cx, cy, corners)) {
                tri.covered = true;
                tri.coverage_time = time;
                covered_area_ += tri.area;
            }
        }
    }

    /**
     * Clear all coverage data
     */
    void clearCoverage() {
        for (auto& tri : triangles_) {
            tri.covered = false;
            tri.coverage_time = 0.0f;
        }
        covered_area_ = 0.0f;
    }

    /**
     * Export coverage map to file
     */
    bool exportCoverage(const std::string& path) const {
        std::ofstream file(path);
        if (!file.is_open()) return false;

        file << "# Hull Coverage Export\n";
        file << "# Total triangles: " << triangles_.size() << "\n";
        file << "# Covered triangles: " << getCoveredTriangleCount() << "\n";
        file << "# Coverage: " << (getCoveragePercent() * 100.0f) << "%\n";
        file << "# Total area: " << total_area_ << " m²\n";
        file << "# Covered area: " << covered_area_ << " m²\n\n";

        file << "# Format: triangle_index, covered (0/1), coverage_time\n";
        for (size_t i = 0; i < triangles_.size(); i++) {
            file << i << "," << (triangles_[i].covered ? 1 : 0)
                 << "," << triangles_[i].coverage_time << "\n";
        }

        return true;
    }

    // Getters
    bool isLoaded() const { return loaded_; }
    const std::string& getFilename() const { return filename_; }
    size_t getVertexCount() const { return vertices_.size(); }
    size_t getTriangleCount() const { return triangles_.size(); }
    float getTotalArea() const { return total_area_; }
    float getCoveredArea() const { return covered_area_; }
    float getCoveragePercent() const {
        return total_area_ > 0 ? covered_area_ / total_area_ : 0.0f;
    }

    size_t getCoveredTriangleCount() const {
        return std::count_if(triangles_.begin(), triangles_.end(),
                            [](const Triangle& t) { return t.covered; });
    }

    const std::vector<Vertex>& getVertices() const { return vertices_; }
    const std::vector<Triangle>& getTriangles() const { return triangles_; }

private:
    bool pointInQuad(float px, float py, float corners[4][2]) const {
        // Check if point is inside quad using cross product signs
        for (int i = 0; i < 4; i++) {
            int j = (i + 1) % 4;
            float ex = corners[j][0] - corners[i][0];
            float ey = corners[j][1] - corners[i][1];
            float px_rel = px - corners[i][0];
            float py_rel = py - corners[i][1];
            float cross = ex * py_rel - ey * px_rel;
            if (cross < 0) return false;  // Point is outside this edge
        }
        return true;
    }

    bool loaded_ = false;
    std::string filename_;
    std::vector<Vertex> vertices_;
    std::vector<Triangle> triangles_;
    float total_area_ = 0.0f;
    float covered_area_ = 0.0f;
};

} // namespace slam_gui
