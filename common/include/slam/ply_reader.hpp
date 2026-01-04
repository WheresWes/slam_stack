/**
 * @file ply_reader.hpp
 * @brief PLY point cloud reader for loading existing point clouds
 */

#ifndef SLAM_PLY_READER_HPP
#define SLAM_PLY_READER_HPP

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

#include "slam/types.hpp"

namespace slam {

/**
 * @brief Load point cloud from PLY file
 * @param filename Input filename
 * @param points Output vector of 3D points
 * @return true on success
 */
inline bool loadFromPly(const std::string& filename, std::vector<V3D>& points) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[PLY Load] Failed to open file: " << filename << std::endl;
        return false;
    }

    points.clear();

    // Parse header
    std::string line;
    int num_vertices = 0;
    bool header_done = false;
    int x_idx = -1, y_idx = -1, z_idx = -1;
    int prop_count = 0;
    bool is_binary = false;

    while (std::getline(file, line)) {
        // Remove carriage returns if present (Windows line endings)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.find("format binary") != std::string::npos) {
            is_binary = true;
        }

        if (line.find("element vertex") != std::string::npos) {
            std::istringstream iss(line);
            std::string elem, vert;
            iss >> elem >> vert >> num_vertices;
        }

        if (line.find("property float x") != std::string::npos ||
            line.find("property float32 x") != std::string::npos) {
            x_idx = prop_count;
        }
        if (line.find("property float y") != std::string::npos ||
            line.find("property float32 y") != std::string::npos) {
            y_idx = prop_count;
        }
        if (line.find("property float z") != std::string::npos ||
            line.find("property float32 z") != std::string::npos) {
            z_idx = prop_count;
        }

        if (line.find("property") != std::string::npos) {
            prop_count++;
        }

        if (line == "end_header") {
            header_done = true;
            break;
        }
    }

    if (!header_done || num_vertices == 0) {
        std::cerr << "[PLY Load] Invalid PLY header" << std::endl;
        return false;
    }

    // For simplicity, assume x, y, z are first 3 properties (common convention)
    if (x_idx < 0) x_idx = 0;
    if (y_idx < 0) y_idx = 1;
    if (z_idx < 0) z_idx = 2;

    points.reserve(num_vertices);

    if (is_binary) {
        // Binary PLY - read raw floats
        for (int i = 0; i < num_vertices; i++) {
            std::vector<float> props(prop_count);
            file.read(reinterpret_cast<char*>(props.data()), prop_count * sizeof(float));
            if (!file) break;
            points.emplace_back(props[x_idx], props[y_idx], props[z_idx]);
        }
    } else {
        // ASCII PLY
        for (int i = 0; i < num_vertices && std::getline(file, line); i++) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            std::istringstream iss(line);
            std::vector<float> props;
            float val;
            while (iss >> val) {
                props.push_back(val);
            }
            if (props.size() >= 3) {
                points.emplace_back(
                    static_cast<double>(props[x_idx]),
                    static_cast<double>(props[y_idx]),
                    static_cast<double>(props[z_idx])
                );
            }
        }
    }

    file.close();
    std::cout << "[PLY Load] Loaded " << points.size() << " points from: " << filename << std::endl;
    return !points.empty();
}

} // namespace slam

#endif // SLAM_PLY_READER_HPP
