/**
 * @file test_slam_viewer.cpp
 * @brief Test application for the SLAM visualization system
 *
 * Demonstrates:
 * - Point cloud visualization with intensity colormap
 * - Camera controls (orbit, pan, zoom)
 * - Standalone window rendering
 *
 * Usage:
 *   test_slam_viewer [--input recording.bin] [--map map.ply]
 */

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <thread>

#include "slam_viewer.hpp"

using namespace slam::viz;

// Generate synthetic point cloud for testing
void generateTestPointCloud(std::vector<PointData>& points, size_t count) {
    points.resize(count);

    // Create a simple scene: floor + walls
    for (size_t i = 0; i < count; i++) {
        float t = static_cast<float>(i) / static_cast<float>(count);

        if (t < 0.4f) {
            // Floor points
            float x = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
            float z = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
            points[i].x = x;
            points[i].y = 0.0f;
            points[i].z = z;
            points[i].intensity = static_cast<uint8_t>(50 + rand() % 50);
        } else if (t < 0.7f) {
            // Wall 1
            float x = -5.0f;
            float y = static_cast<float>(rand()) / RAND_MAX * 3.0f;
            float z = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
            points[i].x = x;
            points[i].y = y;
            points[i].z = z;
            points[i].intensity = static_cast<uint8_t>(100 + rand() % 100);
        } else {
            // Random scatter
            float theta = static_cast<float>(rand()) / RAND_MAX * 6.28f;
            float phi = static_cast<float>(rand()) / RAND_MAX * 3.14f * 0.5f;
            float r = 2.0f + static_cast<float>(rand()) / RAND_MAX * 3.0f;

            points[i].x = r * std::sin(phi) * std::cos(theta);
            points[i].y = r * std::cos(phi);
            points[i].z = r * std::sin(phi) * std::sin(theta);
            points[i].intensity = static_cast<uint8_t>(150 + rand() % 105);
        }
    }
}

// Load point cloud from PLY file
bool loadPLY(const std::string& filename, std::vector<PointData>& points) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filename << std::endl;
        return false;
    }

    std::string line;
    int vertexCount = 0;
    bool hasIntensity = false;
    bool intensityIsFloat = false;
    bool headerEnded = false;

    // Parse header
    while (std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos) {
            sscanf(line.c_str(), "element vertex %d", &vertexCount);
        }
        if (line.find("property") != std::string::npos &&
            line.find("intensity") != std::string::npos) {
            hasIntensity = true;
            // Check if intensity is float or uchar
            if (line.find("float") != std::string::npos) {
                intensityIsFloat = true;
            }
        }
        if (line == "end_header") {
            headerEnded = true;
            break;
        }
    }

    if (!headerEnded || vertexCount == 0) {
        std::cerr << "Invalid PLY header" << std::endl;
        return false;
    }

    std::cout << "PLY format: " << vertexCount << " vertices, intensity="
              << (hasIntensity ? (intensityIsFloat ? "float" : "uchar") : "none") << std::endl;

    points.resize(vertexCount);

    // Read binary data
    for (int i = 0; i < vertexCount; i++) {
        float x, y, z;
        file.read(reinterpret_cast<char*>(&x), sizeof(float));
        file.read(reinterpret_cast<char*>(&y), sizeof(float));
        file.read(reinterpret_cast<char*>(&z), sizeof(float));

        points[i].x = x;
        points[i].y = y;
        points[i].z = z;

        if (hasIntensity) {
            if (intensityIsFloat) {
                float intensity;
                file.read(reinterpret_cast<char*>(&intensity), sizeof(float));
                // Normalize float intensity to 0-255
                points[i].intensity = static_cast<uint8_t>(std::clamp(intensity, 0.0f, 255.0f));
            } else {
                uint8_t intensity;
                file.read(reinterpret_cast<char*>(&intensity), sizeof(uint8_t));
                points[i].intensity = intensity;
            }
        } else {
            // Generate intensity from height
            points[i].intensity = static_cast<uint8_t>(
                std::clamp((y + 2.0f) / 5.0f * 255.0f, 0.0f, 255.0f));
        }
    }

    std::cout << "Loaded " << vertexCount << " points from " << filename << std::endl;
    return true;
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\nOptions:\n"
              << "  --map <file.ply>      Load point cloud from PLY file\n"
              << "  --test                Generate synthetic test data\n"
              << "  --points <count>      Number of test points (default: 100000)\n"
              << "\nControls:\n"
              << "  Left mouse + drag     Rotate camera\n"
              << "  Middle mouse + drag   Pan camera\n"
              << "  Scroll wheel          Zoom\n"
              << "  ESC                   Exit\n"
              << std::endl;
}

int main(int argc, char** argv) {
    std::string map_file;
    bool use_test_data = false;
    size_t test_point_count = 100000;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--map") == 0 && i + 1 < argc) {
            map_file = argv[++i];
        } else if (strcmp(argv[i], "--test") == 0) {
            use_test_data = true;
        } else if (strcmp(argv[i], "--points") == 0 && i + 1 < argc) {
            test_point_count = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (map_file.empty() && !use_test_data) {
        std::cout << "No input specified, using test data\n" << std::endl;
        use_test_data = true;
    }

    std::cout << "============================================" << std::endl;
    std::cout << "    SLAM Viewer Test Application" << std::endl;
    std::cout << "============================================" << std::endl;

    // Load or generate point cloud
    std::vector<PointData> points;

    if (!map_file.empty()) {
        if (!loadPLY(map_file, points)) {
            std::cerr << "Failed to load map file" << std::endl;
            return 1;
        }
    } else {
        std::cout << "Generating " << test_point_count << " test points..." << std::endl;
        generateTestPointCloud(points, test_point_count);
    }

    std::cout << "Point cloud: " << points.size() << " points" << std::endl;

    // Create viewer
    SlamViewer viewer;

    // Initialize standalone window
    if (!viewer.initStandalone(1280, 720, "SLAM Viewer Test")) {
        std::cerr << "Failed to initialize viewer" << std::endl;
        return 1;
    }

    // Configure viewer
    ViewerConfig config;
    config.point_cloud.point_size = 8.0f;  // Larger points for visibility
    config.point_cloud.colormap = Colormap::TURBO;
    config.background_color = V4F(0.1f, 0.1f, 0.15f, 1.0f);  // Slightly lighter background
    viewer.setConfig(config);

    // Set mode
    viewer.setMode(ViewMode::SCANNING);

    // Upload point cloud
    viewer.updatePointCloud(points.data(), points.size());

    // Fit camera to content
    viewer.fitCameraToContent();

    // Debug: print point bounds
    float minX = 1e10f, maxX = -1e10f;
    float minY = 1e10f, maxY = -1e10f;
    float minZ = 1e10f, maxZ = -1e10f;
    for (const auto& p : points) {
        minX = std::min(minX, p.x); maxX = std::max(maxX, p.x);
        minY = std::min(minY, p.y); maxY = std::max(maxY, p.y);
        minZ = std::min(minZ, p.z); maxZ = std::max(maxZ, p.z);
    }
    std::cout << "Point bounds: X[" << minX << ", " << maxX << "] "
              << "Y[" << minY << ", " << maxY << "] "
              << "Z[" << minZ << ", " << maxZ << "]" << std::endl;

    std::cout << "\nViewer running. Press ESC to exit.\n" << std::endl;

    // Main loop
    auto lastTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;

    while (viewer.renderStandalone()) {
        frameCount++;

        // Print FPS every second
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - lastTime).count();

        if (elapsed >= 1.0) {
            auto stats = viewer.getStats();
            std::cout << "FPS: " << frameCount << " | Points: "
                      << stats.visible_points << "/" << stats.total_points << "\r" << std::flush;
            frameCount = 0;
            lastTime = now;
        }

        // Small sleep to avoid maxing out CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cout << "\n\nViewer closed." << std::endl;
    return 0;
}
