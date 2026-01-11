/**
 * @file analyze_recording.cpp
 * @brief Analyze recorded LiDAR/IMU data to diagnose tracking issues
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Core>

using V3D = Eigen::Vector3d;

struct RecordedPointCloud {
    uint64_t timestamp_ns;
    std::vector<V3D> points;
    size_t point_count;
};

struct RecordedIMU {
    uint64_t timestamp_ns;
    V3D accel;
    V3D gyro;
    double accel_mag;
    double gyro_mag;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <recording.bin>\n";
        return 1;
    }

    std::ifstream file(argv[1], std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << argv[1] << std::endl;
        return 1;
    }

    // Read header
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x534C414D) {
        std::cerr << "Invalid file format\n";
        return 1;
    }

    std::vector<RecordedPointCloud> clouds;
    std::vector<RecordedIMU> imus;

    // Read all data
    while (file.good() && !file.eof()) {
        uint8_t msg_type;
        file.read(reinterpret_cast<char*>(&msg_type), 1);
        if (file.eof()) break;

        if (msg_type == 1) {  // Point cloud
            RecordedPointCloud cloud;
            file.read(reinterpret_cast<char*>(&cloud.timestamp_ns), sizeof(cloud.timestamp_ns));

            uint32_t num_points;
            file.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
            cloud.point_count = num_points;

            for (uint32_t i = 0; i < num_points; i++) {
                float x, y, z;
                uint8_t refl;
                file.read(reinterpret_cast<char*>(&x), sizeof(x));
                file.read(reinterpret_cast<char*>(&y), sizeof(y));
                file.read(reinterpret_cast<char*>(&z), sizeof(z));
                file.read(reinterpret_cast<char*>(&refl), sizeof(refl));
                cloud.points.emplace_back(x, y, z);
            }
            clouds.push_back(std::move(cloud));

        } else if (msg_type == 2) {  // IMU
            RecordedIMU imu;
            file.read(reinterpret_cast<char*>(&imu.timestamp_ns), sizeof(imu.timestamp_ns));

            float ax, ay, az, gx, gy, gz;
            file.read(reinterpret_cast<char*>(&ax), sizeof(ax));
            file.read(reinterpret_cast<char*>(&ay), sizeof(ay));
            file.read(reinterpret_cast<char*>(&az), sizeof(az));
            file.read(reinterpret_cast<char*>(&gx), sizeof(gx));
            file.read(reinterpret_cast<char*>(&gy), sizeof(gy));
            file.read(reinterpret_cast<char*>(&gz), sizeof(gz));

            imu.accel = V3D(ax, ay, az);
            imu.gyro = V3D(gx, gy, gz);
            imu.accel_mag = imu.accel.norm();
            imu.gyro_mag = imu.gyro.norm();
            imus.push_back(imu);
        }
    }

    file.close();

    std::cout << "============================================\n";
    std::cout << "  Recording Analysis\n";
    std::cout << "============================================\n\n";

    std::cout << "Total point cloud frames: " << clouds.size() << "\n";
    std::cout << "Total IMU samples: " << imus.size() << "\n\n";

    if (clouds.empty() || imus.empty()) {
        std::cerr << "No data!\n";
        return 1;
    }

    // Time range
    uint64_t start_time = std::min(clouds[0].timestamp_ns, imus[0].timestamp_ns);
    uint64_t end_time = std::max(clouds.back().timestamp_ns, imus.back().timestamp_ns);
    double duration = (end_time - start_time) / 1e9;
    std::cout << "Duration: " << duration << " seconds\n\n";

    // Analyze point clouds
    std::cout << "=== Point Cloud Analysis ===\n";
    size_t min_pts = SIZE_MAX, max_pts = 0;
    double avg_pts = 0;
    std::vector<size_t> low_point_frames;

    for (size_t i = 0; i < clouds.size(); i++) {
        size_t pts = clouds[i].point_count;
        min_pts = std::min(min_pts, pts);
        max_pts = std::max(max_pts, pts);
        avg_pts += pts;
        if (pts < 50) {  // Very low point count
            low_point_frames.push_back(i);
        }
    }
    avg_pts /= clouds.size();

    std::cout << "  Points per frame: min=" << min_pts << ", max=" << max_pts
              << ", avg=" << (int)avg_pts << "\n";

    if (!low_point_frames.empty()) {
        std::cout << "  WARNING: " << low_point_frames.size()
                  << " frames with <50 points (indices: ";
        for (size_t i = 0; i < std::min(low_point_frames.size(), (size_t)10); i++) {
            std::cout << low_point_frames[i];
            if (i < std::min(low_point_frames.size(), (size_t)10) - 1) std::cout << ", ";
        }
        if (low_point_frames.size() > 10) std::cout << "...";
        std::cout << ")\n";
    }

    // Analyze IMU - find extreme motion events
    std::cout << "\n=== IMU Analysis ===\n";

    double max_accel = 0, max_gyro = 0;
    double avg_accel = 0, avg_gyro = 0;
    std::vector<std::pair<size_t, double>> high_accel_events;
    std::vector<std::pair<size_t, double>> high_gyro_events;

    for (size_t i = 0; i < imus.size(); i++) {
        // Remove gravity from accel magnitude (approximate)
        double accel_no_g = std::abs(imus[i].accel_mag - 9.81);
        max_accel = std::max(max_accel, accel_no_g);
        avg_accel += accel_no_g;

        max_gyro = std::max(max_gyro, imus[i].gyro_mag);
        avg_gyro += imus[i].gyro_mag;

        // Detect high motion events
        if (accel_no_g > 2.0) {  // >2 m/s² acceleration
            high_accel_events.push_back({i, accel_no_g});
        }
        if (imus[i].gyro_mag > 1.0) {  // >1 rad/s (~57 deg/s)
            high_gyro_events.push_back({i, imus[i].gyro_mag});
        }
    }
    avg_accel /= imus.size();
    avg_gyro /= imus.size();

    std::cout << "  Acceleration (excluding gravity):\n";
    std::cout << "    max=" << max_accel << " m/s², avg=" << avg_accel << " m/s²\n";
    std::cout << "  Angular velocity:\n";
    std::cout << "    max=" << max_gyro << " rad/s (" << max_gyro * 180/M_PI << " deg/s)\n";
    std::cout << "    avg=" << avg_gyro << " rad/s (" << avg_gyro * 180/M_PI << " deg/s)\n";

    if (!high_accel_events.empty()) {
        std::cout << "\n  High acceleration events (>2 m/s²): " << high_accel_events.size() << "\n";
        // Find time ranges
        double event_start = (imus[high_accel_events[0].first].timestamp_ns - start_time) / 1e9;
        double event_end = (imus[high_accel_events.back().first].timestamp_ns - start_time) / 1e9;
        std::cout << "    Time range: " << event_start << "s - " << event_end << "s\n";

        // Find max event
        auto max_event = std::max_element(high_accel_events.begin(), high_accel_events.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        double max_time = (imus[max_event->first].timestamp_ns - start_time) / 1e9;
        std::cout << "    Peak: " << max_event->second << " m/s² at t=" << max_time << "s\n";
    }

    if (!high_gyro_events.empty()) {
        std::cout << "\n  High rotation events (>1 rad/s): " << high_gyro_events.size() << "\n";
        double event_start = (imus[high_gyro_events[0].first].timestamp_ns - start_time) / 1e9;
        double event_end = (imus[high_gyro_events.back().first].timestamp_ns - start_time) / 1e9;
        std::cout << "    Time range: " << event_start << "s - " << event_end << "s\n";

        auto max_event = std::max_element(high_gyro_events.begin(), high_gyro_events.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        double max_time = (imus[max_event->first].timestamp_ns - start_time) / 1e9;
        std::cout << "    Peak: " << max_event->second << " rad/s ("
                  << max_event->second * 180/M_PI << " deg/s) at t=" << max_time << "s\n";
    }

    // Analyze gaps in data
    std::cout << "\n=== Timing Analysis ===\n";

    std::vector<std::pair<size_t, double>> large_gaps;
    for (size_t i = 1; i < clouds.size(); i++) {
        double gap_ms = (clouds[i].timestamp_ns - clouds[i-1].timestamp_ns) / 1e6;
        if (gap_ms > 150) {  // >150ms gap (normal is ~5ms between packets)
            large_gaps.push_back({i, gap_ms});
        }
    }

    if (!large_gaps.empty()) {
        std::cout << "  Large gaps in point cloud data (>150ms): " << large_gaps.size() << "\n";
        for (const auto& gap : large_gaps) {
            double gap_time = (clouds[gap.first].timestamp_ns - start_time) / 1e9;
            std::cout << "    Frame " << gap.first << " at t=" << gap_time
                      << "s: gap=" << gap.second << "ms\n";
        }
    } else {
        std::cout << "  No large gaps in point cloud data\n";
    }

    // Timeline of critical period (if tracking likely lost)
    std::cout << "\n=== Critical Period Analysis (t=10-20s) ===\n";

    int critical_high_accel = 0, critical_high_gyro = 0;
    for (const auto& imu : imus) {
        double t = (imu.timestamp_ns - start_time) / 1e9;
        if (t >= 10.0 && t <= 20.0) {
            double accel_no_g = std::abs(imu.accel_mag - 9.81);
            if (accel_no_g > 2.0) critical_high_accel++;
            if (imu.gyro_mag > 1.0) critical_high_gyro++;
        }
    }

    std::cout << "  High accel events (>2 m/s²): " << critical_high_accel << "\n";
    std::cout << "  High rotation events (>1 rad/s): " << critical_high_gyro << "\n";

    std::cout << "\n============================================\n";
    std::cout << "  Diagnosis\n";
    std::cout << "============================================\n";

    if (!high_gyro_events.empty() && high_gyro_events.size() > 50) {
        std::cout << "LIKELY CAUSE: Fast rotation (" << high_gyro_events.size()
                  << " events >57 deg/s)\n";
        std::cout << "  - IEKF may not converge during rapid rotation\n";
        std::cout << "  - Try: slower rotation, or increase iterations\n";
    }
    if (!high_accel_events.empty() && high_accel_events.size() > 50) {
        std::cout << "LIKELY CAUSE: Fast acceleration (" << high_accel_events.size()
                  << " events >2 m/s²)\n";
        std::cout << "  - Motion blur in point cloud\n";
        std::cout << "  - Try: slower movement\n";
    }
    if (!large_gaps.empty()) {
        std::cout << "LIKELY CAUSE: Data gaps (" << large_gaps.size() << " gaps >150ms)\n";
        std::cout << "  - Missing point cloud data\n";
        std::cout << "  - Check: network/USB stability\n";
    }
    if (high_gyro_events.empty() && high_accel_events.empty() && large_gaps.empty()) {
        std::cout << "No obvious motion/timing issues detected.\n";
        std::cout << "Possible causes:\n";
        std::cout << "  - Feature-poor environment\n";
        std::cout << "  - Incorrect IMU/LiDAR calibration\n";
        std::cout << "  - Insufficient IEKF iterations\n";
    }

    return 0;
}
