/**
 * @file test_global_relocalization.cpp
 * @brief Global relocalization with exhaustive search for large environments
 *
 * Designed for:
 * - 130m+ boat hulls
 * - Unknown initial position AND orientation
 * - Up to 20 second search time budget
 *
 * Strategy:
 * 1. Build voxel occupancy map for fast hypothesis scoring
 * 2. Grid search: sample positions across map bounds
 * 3. Rotation search: try multiple yaw angles at each position
 * 4. Quick score with voxel occupancy (O(n) per hypothesis)
 * 5. Coarse ICP on top candidates
 * 6. Fine ICP on best candidates
 * 7. Disambiguation if multiple good matches
 */

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
#endif

#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>
#include <queue>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "slam/types.hpp"
#include "slam/icp.hpp"
#include "slam/icp_accelerated.hpp"

using namespace slam;

//=============================================================================
// Voxel Occupancy Map for Fast Scoring
//=============================================================================

class VoxelMap {
public:
    VoxelMap(double voxel_size = 0.5) : voxel_size_(voxel_size) {}

    void build(const std::vector<V3D>& points) {
        occupied_.clear();
        if (points.empty()) return;

        min_bound_ = max_bound_ = points[0];
        for (const auto& p : points) {
            min_bound_ = min_bound_.cwiseMin(p);
            max_bound_ = max_bound_.cwiseMax(p);

            int64_t key = toKey(p);
            occupied_.insert(key);
        }
        std::cout << "  Voxel map: " << occupied_.size() << " occupied voxels\n";
        std::cout << "  Bounds: [" << min_bound_.x() << "," << min_bound_.y() << "," << min_bound_.z()
                  << "] to [" << max_bound_.x() << "," << max_bound_.y() << "," << max_bound_.z() << "]\n";
    }

    // Score a pose by counting how many transformed query points hit occupied voxels
    double scorePose(const std::vector<V3D>& query, const M4D& pose, int max_samples = 500) const {
        if (query.empty() || occupied_.empty()) return 0.0;

        M3D R = pose.block<3,3>(0,0);
        V3D t = pose.block<3,1>(0,3);

        int hits = 0;
        int step = std::max(1, static_cast<int>(query.size()) / max_samples);
        int count = 0;

        for (size_t i = 0; i < query.size(); i += step) {
            V3D world_pt = R * query[i] + t;
            if (occupied_.count(toKey(world_pt)) > 0) {
                hits++;
            }
            count++;
        }

        return static_cast<double>(hits) / count;
    }

    V3D minBound() const { return min_bound_; }
    V3D maxBound() const { return max_bound_; }

private:
    double voxel_size_;
    std::unordered_set<int64_t> occupied_;
    V3D min_bound_, max_bound_;

    int64_t toKey(const V3D& p) const {
        int64_t ix = static_cast<int64_t>(std::floor(p.x() / voxel_size_));
        int64_t iy = static_cast<int64_t>(std::floor(p.y() / voxel_size_));
        int64_t iz = static_cast<int64_t>(std::floor(p.z() / voxel_size_));
        return (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791);
    }
};

//=============================================================================
// Pose Hypothesis
//=============================================================================

struct PoseHypothesis {
    M4D pose = M4D::Identity();
    double score = 0.0;
    double fitness = 0.0;
    double rmse = 0.0;
    std::string source;
    int refinement_level = 0;  // 0=voxel, 1=coarse, 2=medium, 3=fine

    bool operator<(const PoseHypothesis& other) const {
        return score < other.score;  // For max-heap
    }
};

//=============================================================================
// Global Localizer
//=============================================================================

struct GlobalLocConfig {
    // Search parameters
    double position_step = 3.0;       // Grid step in meters
    double yaw_step_deg = 30.0;       // Rotation step in degrees
    double z_search_range = 2.0;      // Vertical search range

    // Scoring
    int voxel_samples = 500;          // Points for voxel scoring
    double min_voxel_score = 0.15;    // Minimum to consider

    // Candidate counts
    int coarse_candidates = 100;
    int medium_candidates = 20;
    int fine_candidates = 5;

    // ICP parameters
    double coarse_voxel = 0.5;
    double medium_voxel = 0.2;
    double fine_voxel = 0.1;

    // Time budget
    double max_time_seconds = 20.0;
};

struct GlobalLocResult {
    M4D pose = M4D::Identity();
    double fitness = 0.0;
    double confidence = 0.0;  // Gap between best and second best
    bool success = false;
    double elapsed_seconds = 0.0;
    int hypotheses_tested = 0;
    std::string message;
    std::vector<PoseHypothesis> top_candidates;
};

M4D makePose(double x, double y, double z, double yaw) {
    M3D R = Eigen::AngleAxisd(yaw, V3D::UnitZ()).toRotationMatrix();
    M4D pose = M4D::Identity();
    pose.block<3,3>(0,0) = R;
    pose(0, 3) = x;
    pose(1, 3) = y;
    pose(2, 3) = z;
    return pose;
}

GlobalLocResult globalLocalize(
    const std::vector<V3D>& query_scan,
    const std::vector<V3D>& map_points,
    const GlobalLocConfig& config = GlobalLocConfig()) {

    GlobalLocResult result;
    auto start_time = std::chrono::steady_clock::now();

    auto elapsed = [&]() {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
    };

    std::cout << "\n=== Global Localization ===\n";
    std::cout << "  Query points: " << query_scan.size() << "\n";
    std::cout << "  Map points: " << map_points.size() << "\n";
    std::cout << "  Time budget: " << config.max_time_seconds << "s\n\n";

    //=========================================================================
    // Phase 1: Build voxel occupancy map
    //=========================================================================
    std::cout << "[Phase 1] Building voxel map...\n";
    VoxelMap voxel_map(config.coarse_voxel);
    voxel_map.build(map_points);

    V3D map_min = voxel_map.minBound();
    V3D map_max = voxel_map.maxBound();
    V3D map_size = map_max - map_min;
    std::cout << "  Map size: " << map_size.x() << "m x " << map_size.y() << "m x " << map_size.z() << "m\n";

    //=========================================================================
    // Phase 2: Generate candidate poses (grid + rotation search)
    //=========================================================================
    std::cout << "\n[Phase 2] Generating candidate poses...\n";

    std::vector<PoseHypothesis> candidates;
    double yaw_step = config.yaw_step_deg * M_PI / 180.0;
    int num_yaw_steps = static_cast<int>(std::ceil(2.0 * M_PI / yaw_step));

    // Calculate grid dimensions
    int nx = static_cast<int>(std::ceil(map_size.x() / config.position_step)) + 1;
    int ny = static_cast<int>(std::ceil(map_size.y() / config.position_step)) + 1;
    int nz = static_cast<int>(std::ceil(config.z_search_range / config.position_step)) + 1;

    int total_hypotheses = nx * ny * nz * num_yaw_steps;
    std::cout << "  Grid: " << nx << " x " << ny << " x " << nz << " positions\n";
    std::cout << "  Yaw steps: " << num_yaw_steps << " (" << config.yaw_step_deg << " deg each)\n";
    std::cout << "  Total hypotheses: " << total_hypotheses << "\n";

    candidates.reserve(total_hypotheses);

    for (int ix = 0; ix < nx; ix++) {
        double x = map_min.x() + ix * config.position_step;
        for (int iy = 0; iy < ny; iy++) {
            double y = map_min.y() + iy * config.position_step;
            for (int iz = 0; iz < nz; iz++) {
                double z = map_min.z() + iz * config.position_step;
                for (int iyaw = 0; iyaw < num_yaw_steps; iyaw++) {
                    double yaw = iyaw * yaw_step;

                    PoseHypothesis h;
                    h.pose = makePose(x, y, z, yaw);
                    h.source = "grid";
                    candidates.push_back(h);
                }
            }
        }
    }

    //=========================================================================
    // Phase 3: Quick scoring with voxel occupancy
    //=========================================================================
    std::cout << "\n[Phase 3] Voxel scoring " << candidates.size() << " hypotheses...\n";
    auto phase3_start = std::chrono::steady_clock::now();

    int num_candidates = static_cast<int>(candidates.size());
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_candidates; i++) {
        candidates[i].score = voxel_map.scorePose(query_scan, candidates[i].pose, config.voxel_samples);
    }

    // Sort by score (descending)
    std::sort(candidates.begin(), candidates.end(),
              [](const PoseHypothesis& a, const PoseHypothesis& b) {
                  return a.score > b.score;
              });

    auto phase3_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase3_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << phase3_time << "s\n";
    std::cout << "  Best voxel score: " << candidates[0].score << "\n";
    std::cout << "  Top 10 scores: ";
    for (int i = 0; i < std::min(10, static_cast<int>(candidates.size())); i++) {
        std::cout << std::fixed << std::setprecision(3) << candidates[i].score << " ";
    }
    std::cout << "\n";

    // Filter to candidates above threshold
    std::vector<PoseHypothesis> good_candidates;
    for (const auto& c : candidates) {
        if (c.score >= config.min_voxel_score) {
            good_candidates.push_back(c);
        }
    }
    std::cout << "  Candidates above threshold: " << good_candidates.size() << "\n";

    if (good_candidates.empty()) {
        result.message = "No candidates passed voxel scoring threshold";
        result.elapsed_seconds = elapsed();
        return result;
    }

    // Limit to top N for coarse ICP
    if (good_candidates.size() > static_cast<size_t>(config.coarse_candidates)) {
        good_candidates.resize(config.coarse_candidates);
    }

    result.hypotheses_tested = static_cast<int>(candidates.size());

    //=========================================================================
    // Phase 4: Coarse ICP refinement
    //=========================================================================
    std::cout << "\n[Phase 4] Coarse ICP on " << good_candidates.size() << " candidates...\n";
    auto phase4_start = std::chrono::steady_clock::now();

    auto query_coarse = voxelDownsample(query_scan, config.coarse_voxel);
    auto map_coarse = voxelDownsample(map_points, config.coarse_voxel);
    std::cout << "  Downsampled: query=" << query_coarse.size() << ", map=" << map_coarse.size() << "\n";

    ICPConfig coarse_icp_config;
    coarse_icp_config.max_iterations = 15;
    coarse_icp_config.max_correspondence_dist = 3.0;
    coarse_icp_config.convergence_threshold = 1e-4;

    for (auto& c : good_candidates) {
#ifdef HAS_NANOFLANN
        ICPAccelerated icp(coarse_icp_config);
#else
        ICP icp(coarse_icp_config);
#endif
        auto icp_result = icp.align(query_coarse, map_coarse, c.pose);
        c.pose = icp_result.transformation;
        c.fitness = icp_result.fitness_score;
        c.rmse = icp_result.rmse;
        c.score = c.fitness;  // Use fitness as new score
        c.refinement_level = 1;

        // Time check
        if (elapsed() > config.max_time_seconds * 0.6) {
            std::cout << "  (time limit approaching, stopping coarse ICP early)\n";
            break;
        }
    }

    std::sort(good_candidates.begin(), good_candidates.end(),
              [](const PoseHypothesis& a, const PoseHypothesis& b) {
                  return a.fitness > b.fitness;
              });

    auto phase4_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase4_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << phase4_time << "s\n";
    std::cout << "  Best coarse fitness: " << good_candidates[0].fitness << "\n";

    // Keep top candidates for medium refinement
    if (good_candidates.size() > static_cast<size_t>(config.medium_candidates)) {
        good_candidates.resize(config.medium_candidates);
    }

    //=========================================================================
    // Phase 5: Medium ICP refinement
    //=========================================================================
    std::cout << "\n[Phase 5] Medium ICP on " << good_candidates.size() << " candidates...\n";
    auto phase5_start = std::chrono::steady_clock::now();

    auto query_medium = voxelDownsample(query_scan, config.medium_voxel);
    auto map_medium = voxelDownsample(map_points, config.medium_voxel);

    ICPConfig medium_icp_config;
    medium_icp_config.max_iterations = 30;
    medium_icp_config.max_correspondence_dist = 1.0;
    medium_icp_config.convergence_threshold = 1e-5;

    for (auto& c : good_candidates) {
#ifdef HAS_NANOFLANN
        ICPAccelerated icp(medium_icp_config);
#else
        ICP icp(medium_icp_config);
#endif
        auto icp_result = icp.align(query_medium, map_medium, c.pose);
        c.pose = icp_result.transformation;
        c.fitness = icp_result.fitness_score;
        c.rmse = icp_result.rmse;
        c.refinement_level = 2;
    }

    std::sort(good_candidates.begin(), good_candidates.end(),
              [](const PoseHypothesis& a, const PoseHypothesis& b) {
                  return a.fitness > b.fitness;
              });

    auto phase5_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase5_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << phase5_time << "s\n";
    std::cout << "  Best medium fitness: " << good_candidates[0].fitness << "\n";

    // Keep top candidates for fine refinement
    if (good_candidates.size() > static_cast<size_t>(config.fine_candidates)) {
        good_candidates.resize(config.fine_candidates);
    }

    //=========================================================================
    // Phase 6: Fine ICP refinement
    //=========================================================================
    std::cout << "\n[Phase 6] Fine ICP on " << good_candidates.size() << " candidates...\n";
    auto phase6_start = std::chrono::steady_clock::now();

    auto query_fine = voxelDownsample(query_scan, config.fine_voxel);
    auto map_fine = voxelDownsample(map_points, config.fine_voxel);

    ICPConfig fine_icp_config;
    fine_icp_config.max_iterations = 50;
    fine_icp_config.max_correspondence_dist = 0.5;
    fine_icp_config.convergence_threshold = 1e-6;

    for (auto& c : good_candidates) {
#ifdef HAS_NANOFLANN
        ICPAccelerated icp(fine_icp_config);
#else
        ICP icp(fine_icp_config);
#endif
        auto icp_result = icp.align(query_fine, map_fine, c.pose);
        c.pose = icp_result.transformation;
        c.fitness = icp_result.fitness_score;
        c.rmse = icp_result.rmse;
        c.refinement_level = 3;
    }

    std::sort(good_candidates.begin(), good_candidates.end(),
              [](const PoseHypothesis& a, const PoseHypothesis& b) {
                  return a.fitness > b.fitness;
              });

    auto phase6_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase6_start).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << phase6_time << "s\n";

    //=========================================================================
    // Phase 7: Result and confidence
    //=========================================================================
    result.elapsed_seconds = elapsed();
    result.top_candidates = good_candidates;

    if (!good_candidates.empty()) {
        result.pose = good_candidates[0].pose;
        result.fitness = good_candidates[0].fitness;

        // Calculate confidence as gap between best and second best
        if (good_candidates.size() >= 2) {
            result.confidence = (good_candidates[0].fitness - good_candidates[1].fitness)
                               / std::max(0.01, good_candidates[0].fitness);
        } else {
            result.confidence = 1.0;
        }

        // Success criteria
        if (result.fitness >= 0.4 && result.confidence >= 0.05) {
            result.success = true;
            result.message = "Localization successful";
        } else if (result.fitness >= 0.3) {
            result.success = true;
            result.message = "Localization likely (low confidence)";
        } else {
            result.message = "Localization uncertain";
        }
    } else {
        result.message = "No valid candidates found";
    }

    std::cout << "\n[Result]\n";
    std::cout << "  Best fitness: " << result.fitness << "\n";
    std::cout << "  Confidence: " << result.confidence << "\n";
    std::cout << "  Total time: " << result.elapsed_seconds << "s\n";
    std::cout << "  Status: " << result.message << "\n";

    return result;
}

//=============================================================================
// File Loading (same as before)
//=============================================================================

std::vector<V3D> loadPlyFile(const std::string& filename) {
    std::vector<V3D> points;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return points;

    std::string line;
    int vertex_count = 0;
    bool is_binary = false;

    while (std::getline(file, line)) {
        if (line.find("element vertex") != std::string::npos) {
            sscanf(line.c_str(), "element vertex %d", &vertex_count);
        }
        if (line.find("format binary") != std::string::npos) {
            is_binary = true;
        }
        if (line == "end_header") break;
    }

    points.reserve(vertex_count);
    if (is_binary) {
        for (int i = 0; i < vertex_count; i++) {
            float x, y, z, intensity;
            file.read(reinterpret_cast<char*>(&x), sizeof(float));
            file.read(reinterpret_cast<char*>(&y), sizeof(float));
            file.read(reinterpret_cast<char*>(&z), sizeof(float));
            file.read(reinterpret_cast<char*>(&intensity), sizeof(float));
            if (file.good()) points.emplace_back(x, y, z);
        }
    }
    return points;
}

bool loadRecordedScans(const std::string& filename, std::vector<V3D>& all_points, int max_frames = 50) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != 0x534C414D) return false;

    const double SCAN_PERIOD_NS = 100e6;
    uint64_t scan_start_time = 0;
    int frame_count = 0;
    std::vector<V3D> current_scan;

    while (file.good() && !file.eof()) {
        uint8_t msg_type;
        file.read(reinterpret_cast<char*>(&msg_type), 1);
        if (file.eof() || !file.good()) break;

        if (msg_type == 1) {
            uint64_t timestamp_ns;
            file.read(reinterpret_cast<char*>(&timestamp_ns), sizeof(timestamp_ns));

            uint32_t num_points;
            file.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));

            if (scan_start_time == 0) scan_start_time = timestamp_ns;

            for (uint32_t i = 0; i < num_points; i++) {
                float x, y, z;
                uint8_t refl;
                file.read(reinterpret_cast<char*>(&x), sizeof(x));
                file.read(reinterpret_cast<char*>(&y), sizeof(y));
                file.read(reinterpret_cast<char*>(&z), sizeof(z));
                file.read(reinterpret_cast<char*>(&refl), sizeof(refl));

                double dist = std::sqrt(x*x + y*y + z*z);
                if (dist > 0.5 && dist < 50.0) {
                    current_scan.emplace_back(x, y, z);
                }
            }

            if ((timestamp_ns - scan_start_time) >= SCAN_PERIOD_NS && current_scan.size() > 1000) {
                all_points.insert(all_points.end(), current_scan.begin(), current_scan.end());
                current_scan.clear();
                scan_start_time = timestamp_ns;
                frame_count++;
                if (frame_count >= max_frames) break;
            }
        } else if (msg_type == 2) {
            file.seekg(8 + 6*4, std::ios::cur);  // Skip IMU data
        } else {
            break;
        }
    }
    return !all_points.empty();
}

std::vector<V3D> transformPoints(const std::vector<V3D>& points, const M4D& T) {
    std::vector<V3D> result;
    result.reserve(points.size());
    M3D R = T.block<3,3>(0,0);
    V3D t = T.block<3,1>(0,3);
    for (const auto& p : points) {
        result.push_back(R * p + t);
    }
    return result;
}

//=============================================================================
// Main
//=============================================================================

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <map.ply> <recording.bin> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --offset <x> <y> <yaw>  Apply test offset (meters, degrees)\n";
    std::cout << "  --random                Random offset\n";
    std::cout << "  --grid <step>           Grid search step in meters (default: 3.0)\n";
    std::cout << "  --yaw <step>            Yaw search step in degrees (default: 30)\n";
    std::cout << "  --time <seconds>        Time budget (default: 20)\n";
    std::cout << "  --help                  Show this help\n";
}

int main(int argc, char** argv) {
    std::cout << "============================================\n";
    std::cout << "  Global Relocalization Test\n";
    std::cout << "  (Exhaustive Search for Large Environments)\n";
    std::cout << "============================================\n\n";

    if (argc < 3 || std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return argc < 3 ? 1 : 0;
    }

    std::string map_file = argv[1];
    std::string recording_file = argv[2];
    double offset_x = 5.0, offset_y = 3.0, offset_yaw = 120.0;  // Extreme default
    bool random_offset = false;
    GlobalLocConfig config;

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--offset" && i + 3 < argc) {
            offset_x = std::atof(argv[++i]);
            offset_y = std::atof(argv[++i]);
            offset_yaw = std::atof(argv[++i]);
        } else if (arg == "--random") {
            random_offset = true;
        } else if (arg == "--grid" && i + 1 < argc) {
            config.position_step = std::atof(argv[++i]);
        } else if (arg == "--yaw" && i + 1 < argc) {
            config.yaw_step_deg = std::atof(argv[++i]);
        } else if (arg == "--time" && i + 1 < argc) {
            config.max_time_seconds = std::atof(argv[++i]);
        }
    }

    if (random_offset) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist_pos(-5.0, 5.0);
        std::uniform_real_distribution<> dist_yaw(0, 360);
        offset_x = dist_pos(gen);
        offset_y = dist_pos(gen);
        offset_yaw = dist_yaw(gen);
    }

    // Load map
    std::cout << "Loading map: " << map_file << std::endl;
    auto map_points = loadPlyFile(map_file);
    if (map_points.empty()) {
        std::cerr << "Failed to load map!\n";
        return 1;
    }
    std::cout << "  Map points: " << map_points.size() << "\n";

    // Load query scan
    std::cout << "\nLoading query scan: " << recording_file << std::endl;
    std::vector<V3D> query_raw;
    if (!loadRecordedScans(recording_file, query_raw, 10)) {
        std::cerr << "Failed to load recording!\n";
        return 1;
    }
    auto query_scan = voxelDownsample(query_raw, 0.05);
    std::cout << "  Query points: " << query_scan.size() << "\n";

    // Apply offset to simulate unknown position
    std::cout << "\nTest offset:\n";
    std::cout << "  X: " << offset_x << " m\n";
    std::cout << "  Y: " << offset_y << " m\n";
    std::cout << "  Yaw: " << offset_yaw << " deg\n";

    double offset_yaw_rad = offset_yaw * M_PI / 180.0;
    M4D true_offset = makePose(offset_x, offset_y, 0, offset_yaw_rad);
    auto query_transformed = transformPoints(query_scan, true_offset.inverse());

    // Run global localization
    auto result = globalLocalize(query_transformed, map_points, config);

    // Evaluate result
    std::cout << "\n============================================\n";
    std::cout << "  Evaluation\n";
    std::cout << "============================================\n";

    V3D recovered_pos = result.pose.block<3,1>(0,3);
    double recovered_yaw = std::atan2(result.pose(1,0), result.pose(0,0)) * 180.0 / M_PI;

    double pos_error = (recovered_pos - V3D(offset_x, offset_y, 0)).norm();
    double yaw_error = std::abs(recovered_yaw - offset_yaw);
    if (yaw_error > 180) yaw_error = 360 - yaw_error;

    std::cout << "\nRecovered pose:\n";
    std::cout << "  Position: [" << std::fixed << std::setprecision(2)
              << recovered_pos.x() << ", " << recovered_pos.y() << ", " << recovered_pos.z() << "]\n";
    std::cout << "  Yaw: " << recovered_yaw << " deg\n";

    std::cout << "\nTrue offset:\n";
    std::cout << "  Position: [" << offset_x << ", " << offset_y << ", 0.0]\n";
    std::cout << "  Yaw: " << offset_yaw << " deg\n";

    std::cout << "\nError:\n";
    std::cout << "  Position: " << (pos_error * 100) << " cm\n";
    std::cout << "  Yaw: " << yaw_error << " deg\n";

    bool position_ok = pos_error < 0.5;
    bool yaw_ok = yaw_error < 20;
    bool success = position_ok && yaw_ok && result.fitness > 0.3;

    std::cout << "\n============================================\n";
    if (success) {
        std::cout << "  GLOBAL LOCALIZATION SUCCESSFUL!\n";
    } else {
        std::cout << "  GLOBAL LOCALIZATION FAILED\n";
        if (!position_ok) std::cout << "  - Position error: " << (pos_error*100) << "cm (>50cm)\n";
        if (!yaw_ok) std::cout << "  - Yaw error: " << yaw_error << " deg (>20 deg)\n";
        if (result.fitness <= 0.3) std::cout << "  - Low fitness: " << result.fitness << "\n";
    }
    std::cout << "============================================\n";

    return success ? 0 : 1;
}
