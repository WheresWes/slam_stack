/**
 * @file test_gravity_alignment.cpp
 * @brief Test the gravity alignment fix
 *
 * This example demonstrates how the gravity alignment works.
 * It simulates starting the IMU at various tilt angles and shows
 * how the initial rotation is computed to level the map.
 */

#include <iostream>
#include <iomanip>
#include <cmath>

#include "slam/types.hpp"
#include "slam/so3_math.hpp"

using namespace slam;

void testGravityAlignment(const V3D& measured_gravity, const std::string& description) {
    std::cout << "\n=== " << description << " ===" << std::endl;

    // Normalize measured gravity
    V3D g_measured = measured_gravity.normalized();

    // Expected world gravity direction (down = -Z)
    V3D g_world(0, 0, -1);

    // Compute alignment rotation
    M3D R_align = rotationFromTwoVectors<double>(g_measured, g_world);

    // Apply rotation to measured gravity - should give world gravity
    V3D g_aligned = R_align * g_measured;

    // Get Euler angles (roll, pitch, yaw)
    V3D euler = RotMtoEuler(R_align);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Measured gravity (body frame): [" << g_measured.transpose() << "]" << std::endl;
    std::cout << "  After alignment:               [" << g_aligned.transpose() << "]" << std::endl;
    std::cout << "  Expected (world -Z):           [0.0000 0.0000 -1.0000]" << std::endl;
    std::cout << "  Correction rotation (RPY deg): ["
              << rad2deg(euler(0)) << ", "
              << rad2deg(euler(1)) << ", "
              << rad2deg(euler(2)) << "]" << std::endl;

    // Verify alignment worked
    double error = (g_aligned - g_world).norm();
    if (error < 1e-6) {
        std::cout << "  Status: PASS (error = " << error << ")" << std::endl;
    } else {
        std::cout << "  Status: FAIL (error = " << error << ")" << std::endl;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Gravity Alignment Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // Test 1: Level start (gravity in -Z)
    testGravityAlignment(
        V3D(0, 0, -9.81),
        "Level start (no tilt needed)"
    );

    // Test 2: 30 degree pitch forward
    double pitch_30 = deg2rad(30.0);
    V3D g_pitched = V3D(
        -9.81 * std::sin(pitch_30),
        0,
        -9.81 * std::cos(pitch_30)
    );
    testGravityAlignment(g_pitched, "30 degree pitch forward");

    // Test 3: 15 degree roll right
    double roll_15 = deg2rad(15.0);
    V3D g_rolled = V3D(
        9.81 * std::sin(roll_15),
        0,
        -9.81 * std::cos(roll_15)
    );
    testGravityAlignment(g_rolled, "15 degree roll right");

    // Test 4: Combined 20 degree pitch + 10 degree roll
    double pitch_20 = deg2rad(20.0);
    double roll_10 = deg2rad(10.0);
    M3D R_combined = Exp(V3D(roll_10, pitch_20, 0));
    V3D g_combined = R_combined.transpose() * V3D(0, 0, -9.81);
    testGravityAlignment(g_combined, "20 deg pitch + 10 deg roll");

    // Test 5: Extreme 45 degree tilt
    testGravityAlignment(
        V3D(-6.94, 0, -6.94),  // 45 degree forward pitch
        "45 degree forward pitch"
    );

    // Test 6: Nearly upside down (170 degrees)
    double pitch_170 = deg2rad(170.0);
    V3D g_nearly_inverted = V3D(
        -9.81 * std::sin(pitch_170),
        0,
        -9.81 * std::cos(pitch_170)
    );
    testGravityAlignment(g_nearly_inverted, "Nearly inverted (170 deg pitch)");

    std::cout << "\n========================================" << std::endl;
    std::cout << "  All tests completed!" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}
