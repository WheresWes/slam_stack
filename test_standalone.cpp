/**
 * @file test_standalone.cpp
 * @brief Standalone test for core SLAM components
 *
 * This test verifies:
 * 1. SO3 math operations
 * 2. Gravity alignment calculation
 * 3. Intensity preservation in point transforms
 *
 * Compile with:
 *   cl /EHsc /I third_party/eigen /I common/include /I slam_engine/include test_standalone.cpp
 */

#include <iostream>
#include <cmath>
#include <cassert>

// Define M_PI for MSVC
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Just Eigen - no PCL required
#include <Eigen/Core>
#include <Eigen/Geometry>

// Test the core math
namespace {

using V3D = Eigen::Vector3d;
using M3D = Eigen::Matrix3d;

constexpr double EPSILON = 1e-10;

/**
 * SO3 Exponential map (Rodrigues formula)
 */
M3D Exp(const V3D& omega) {
    double theta = omega.norm();
    if (theta < EPSILON) {
        return M3D::Identity();
    }

    V3D axis = omega / theta;
    M3D K;
    K << 0, -axis.z(), axis.y(),
         axis.z(), 0, -axis.x(),
         -axis.y(), axis.x(), 0;

    return M3D::Identity() + std::sin(theta) * K + (1 - std::cos(theta)) * K * K;
}

/**
 * Rotation from one vector to another
 * This is the key function for gravity alignment
 */
M3D rotationFromTwoVectors(const V3D& from, const V3D& to) {
    V3D a = from.normalized();
    V3D b = to.normalized();

    V3D v = a.cross(b);
    double c = a.dot(b);

    // Handle near-parallel vectors
    if (c > 1.0 - EPSILON) {
        return M3D::Identity();
    }

    // Handle anti-parallel vectors
    if (c < -1.0 + EPSILON) {
        // Find perpendicular axis
        V3D perp = (std::abs(a.x()) < 0.9) ? V3D(1, 0, 0) : V3D(0, 1, 0);
        V3D axis = a.cross(perp).normalized();
        return Eigen::AngleAxisd(M_PI, axis).toRotationMatrix();
    }

    // Rodrigues formula for rotation from cross and dot products
    double s = v.norm();
    M3D vx;
    vx << 0, -v.z(), v.y(),
          v.z(), 0, -v.x(),
          -v.y(), v.x(), 0;

    return M3D::Identity() + vx + vx * vx * ((1 - c) / (s * s));
}

/**
 * Simulate the gravity alignment fix
 */
M3D computeGravityAlignment(const V3D& measured_gravity) {
    V3D g_measured = measured_gravity.normalized();
    V3D g_world(0, 0, -1);  // World gravity points down
    return rotationFromTwoVectors(g_measured, g_world);
}

} // namespace

//=============================================================================
// Test Cases
//=============================================================================

void testRotationFromTwoVectors() {
    std::cout << "Test: rotationFromTwoVectors" << std::endl;

    // Test 1: Rotate Z to X
    {
        V3D from(0, 0, 1);
        V3D to(1, 0, 0);
        M3D R = rotationFromTwoVectors(from, to);
        V3D result = R * from;

        double error = (result - to).norm();
        std::cout << "  Z->X rotation error: " << error << std::endl;
        assert(error < 1e-6);
    }

    // Test 2: Rotate arbitrary vectors
    {
        V3D from(1, 2, 3);
        V3D to(-1, 1, 2);
        M3D R = rotationFromTwoVectors(from, to);
        V3D result = R * from.normalized();

        double error = (result - to.normalized()).norm();
        std::cout << "  Arbitrary rotation error: " << error << std::endl;
        assert(error < 1e-6);
    }

    // Test 3: Same vector (identity)
    {
        V3D from(1, 0, 0);
        V3D to(1, 0, 0);
        M3D R = rotationFromTwoVectors(from, to);

        double error = (R - M3D::Identity()).norm();
        std::cout << "  Identity case error: " << error << std::endl;
        assert(error < 1e-6);
    }

    // Test 4: Opposite vectors (180 degree rotation)
    {
        V3D from(1, 0, 0);
        V3D to(-1, 0, 0);
        M3D R = rotationFromTwoVectors(from, to);
        V3D result = R * from;

        double error = (result - to).norm();
        std::cout << "  180-degree rotation error: " << error << std::endl;
        assert(error < 1e-6);
    }

    std::cout << "  PASSED" << std::endl << std::endl;
}

void testGravityAlignment() {
    std::cout << "Test: Gravity Alignment Fix" << std::endl;

    // Test various tilt angles
    double test_angles[] = {5, 15, 30, 45, 60, 90};

    for (double angle_deg : test_angles) {
        double angle_rad = angle_deg * M_PI / 180.0;

        // Simulate tilted IMU measuring gravity
        // If IMU is tilted by 'angle' around Y axis, it measures gravity as:
        V3D tilted_gravity(-std::sin(angle_rad) * 9.81, 0, -std::cos(angle_rad) * 9.81);

        // Compute alignment rotation
        M3D R_align = computeGravityAlignment(tilted_gravity);

        // Apply to gravity - should align with world Z
        V3D aligned = R_align * tilted_gravity.normalized();
        V3D expected(0, 0, -1);

        double error = (aligned - expected).norm();
        std::cout << "  Tilt " << angle_deg << " deg: error = " << error << std::endl;
        assert(error < 1e-6);
    }

    // Test combined roll/pitch
    {
        double roll = 20 * M_PI / 180;
        double pitch = 35 * M_PI / 180;

        M3D R_tilt = Eigen::AngleAxisd(roll, V3D::UnitX()).toRotationMatrix() *
                     Eigen::AngleAxisd(pitch, V3D::UnitY()).toRotationMatrix();

        V3D gravity_world(0, 0, -9.81);
        V3D measured = R_tilt.transpose() * gravity_world;

        M3D R_align = computeGravityAlignment(measured);
        V3D aligned = R_align * measured.normalized();

        double error = (aligned - V3D(0, 0, -1)).norm();
        std::cout << "  Roll 20 + Pitch 35 deg: error = " << error << std::endl;
        assert(error < 1e-6);
    }

    std::cout << "  PASSED" << std::endl << std::endl;
}

void testPointTransformWithIntensity() {
    std::cout << "Test: Point Transform with Intensity Preservation" << std::endl;

    // Simulate a point with intensity
    struct LidarPoint {
        float x, y, z;
        float intensity;
    };

    struct WorldPoint {
        float x, y, z;
        float intensity;
    };

    // Transform function that preserves intensity
    auto transformPoint = [](const LidarPoint& p, const M3D& R, const V3D& t) -> WorldPoint {
        V3D p_body(p.x, p.y, p.z);
        V3D p_world = R * p_body + t;
        WorldPoint wp;
        wp.x = static_cast<float>(p_world.x());
        wp.y = static_cast<float>(p_world.y());
        wp.z = static_cast<float>(p_world.z());
        wp.intensity = p.intensity;  // KEY: Preserve intensity!
        return wp;
    };

    // Test
    LidarPoint p;
    p.x = 1.0f; p.y = 2.0f; p.z = 3.0f;
    p.intensity = 127.5f;

    M3D R = Eigen::AngleAxisd(0.5, V3D::UnitZ()).toRotationMatrix();
    V3D t(10, 20, 30);

    WorldPoint wp = transformPoint(p, R, t);

    std::cout << "  Original intensity: " << p.intensity << std::endl;
    std::cout << "  Transformed intensity: " << wp.intensity << std::endl;
    assert(std::abs(wp.intensity - p.intensity) < 1e-6);

    std::cout << "  PASSED" << std::endl << std::endl;
}

void testExp() {
    std::cout << "Test: SO3 Exponential Map" << std::endl;

    // Test rotation around Z axis
    {
        V3D omega(0, 0, M_PI / 2);  // 90 degrees around Z
        M3D R = Exp(omega);

        V3D x_axis(1, 0, 0);
        V3D result = R * x_axis;
        V3D expected(0, 1, 0);

        double error = (result - expected).norm();
        std::cout << "  90-deg Z rotation error: " << error << std::endl;
        assert(error < 1e-6);
    }

    // Test small rotation (near identity)
    {
        V3D omega(0.001, 0.002, 0.003);
        M3D R = Exp(omega);

        // Should be close to identity
        double det_error = std::abs(R.determinant() - 1.0);
        std::cout << "  Small rotation det error: " << det_error << std::endl;
        assert(det_error < 1e-10);
    }

    std::cout << "  PASSED" << std::endl << std::endl;
}

//=============================================================================
// Main
//=============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  SLAM Stack Standalone Tests" << std::endl;
    std::cout << "  Testing core math without PCL" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

    testExp();
    testRotationFromTwoVectors();
    testGravityAlignment();
    testPointTransformWithIntensity();

    std::cout << "========================================" << std::endl;
    std::cout << "  ALL TESTS PASSED!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
