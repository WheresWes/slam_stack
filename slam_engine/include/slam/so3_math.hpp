/**
 * @file so3_math.hpp
 * @brief SO(3) math utilities - exponential map, logarithm, skew-symmetric matrices
 *
 * Provides Rodrigues formula implementations for SO(3) operations.
 */

#ifndef SLAM_SO3_MATH_HPP
#define SLAM_SO3_MATH_HPP

#include <cmath>
#include <Eigen/Core>

namespace slam {

/**
 * @brief Create skew-symmetric matrix from vector
 */
template<typename T>
inline Eigen::Matrix<T, 3, 3> skew_sym_mat(const Eigen::Matrix<T, 3, 1>& v) {
    Eigen::Matrix<T, 3, 3> S;
    S <<    T(0), -v[2],  v[1],
           v[2],  T(0), -v[0],
          -v[1],  v[0],  T(0);
    return S;
}

/**
 * @brief Exponential map: so(3) -> SO(3) (Rodrigues formula)
 * @param ang Angle-axis representation (rotation vector)
 * @return Rotation matrix
 */
template<typename T>
inline Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1>& ang) {
    T ang_norm = ang.norm();
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();

    if (ang_norm > T(1e-7)) {
        Eigen::Matrix<T, 3, 1> axis = ang / ang_norm;
        Eigen::Matrix<T, 3, 3> K = skew_sym_mat(axis);
        // Rodrigues formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
        return Eye3 + std::sin(ang_norm) * K + (T(1) - std::cos(ang_norm)) * K * K;
    } else {
        return Eye3;
    }
}

/**
 * @brief Exponential map with angular velocity and time step
 * @param ang_vel Angular velocity (rad/s)
 * @param dt Time step (seconds)
 * @return Rotation matrix
 */
template<typename T, typename Ts>
inline Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1>& ang_vel, const Ts& dt) {
    T ang_vel_norm = ang_vel.norm();
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();

    if (ang_vel_norm > T(1e-7)) {
        Eigen::Matrix<T, 3, 1> axis = ang_vel / ang_vel_norm;
        Eigen::Matrix<T, 3, 3> K = skew_sym_mat(axis);
        T angle = ang_vel_norm * static_cast<T>(dt);
        return Eye3 + std::sin(angle) * K + (T(1) - std::cos(angle)) * K * K;
    } else {
        return Eye3;
    }
}

/**
 * @brief Exponential map from individual components
 */
template<typename T>
inline Eigen::Matrix<T, 3, 3> Exp(const T& v1, const T& v2, const T& v3) {
    T norm = std::sqrt(v1*v1 + v2*v2 + v3*v3);
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();

    if (norm > T(1e-5)) {
        T axis[3] = {v1/norm, v2/norm, v3/norm};
        Eigen::Matrix<T, 3, 3> K;
        K << T(0), -axis[2], axis[1],
             axis[2], T(0), -axis[0],
             -axis[1], axis[0], T(0);
        return Eye3 + std::sin(norm) * K + (T(1) - std::cos(norm)) * K * K;
    } else {
        return Eye3;
    }
}

/**
 * @brief Logarithmic map: SO(3) -> so(3)
 * @param R Rotation matrix
 * @return Angle-axis representation (rotation vector)
 */
template<typename T>
inline Eigen::Matrix<T, 3, 1> Log(const Eigen::Matrix<T, 3, 3>& R) {
    T trace = R.trace();
    T theta = (trace > T(3) - T(1e-6)) ? T(0) : std::acos(T(0.5) * (trace - T(1)));
    Eigen::Matrix<T, 3, 1> K(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));

    if (std::abs(theta) < T(0.001)) {
        return T(0.5) * K;
    } else {
        return T(0.5) * theta / std::sin(theta) * K;
    }
}

/**
 * @brief Convert rotation matrix to Euler angles (ZYX convention)
 * @param rot Rotation matrix
 * @return Euler angles [roll, pitch, yaw]
 */
template<typename T>
inline Eigen::Matrix<T, 3, 1> RotMtoEuler(const Eigen::Matrix<T, 3, 3>& rot) {
    T sy = std::sqrt(rot(0,0)*rot(0,0) + rot(1,0)*rot(1,0));
    bool singular = sy < T(1e-6);

    T x, y, z;
    if (!singular) {
        x = std::atan2(rot(2, 1), rot(2, 2));
        y = std::atan2(-rot(2, 0), sy);
        z = std::atan2(rot(1, 0), rot(0, 0));
    } else {
        x = std::atan2(-rot(1, 2), rot(1, 1));
        y = std::atan2(-rot(2, 0), sy);
        z = T(0);
    }

    return Eigen::Matrix<T, 3, 1>(x, y, z);
}

/**
 * @brief Compute rotation matrix that aligns vector 'from' with vector 'to'
 * @param from Source vector (will be normalized)
 * @param to Target vector (will be normalized)
 * @return Rotation matrix R such that R * from.normalized() = to.normalized()
 */
template<typename T>
inline Eigen::Matrix<T, 3, 3> rotationFromTwoVectors(
    const Eigen::Matrix<T, 3, 1>& from,
    const Eigen::Matrix<T, 3, 1>& to) {

    Eigen::Matrix<T, 3, 1> f = from.normalized();
    Eigen::Matrix<T, 3, 1> t = to.normalized();

    T dot = f.dot(t);

    // Check if vectors are nearly parallel
    if (dot > T(1) - T(1e-7)) {
        // Same direction - identity rotation
        return Eigen::Matrix<T, 3, 3>::Identity();
    }

    if (dot < T(-1) + T(1e-7)) {
        // Opposite direction - 180 degree rotation
        // Find an orthogonal axis
        Eigen::Matrix<T, 3, 1> axis = Eigen::Matrix<T, 3, 1>::UnitX().cross(f);
        if (axis.norm() < T(1e-6)) {
            axis = Eigen::Matrix<T, 3, 1>::UnitY().cross(f);
        }
        axis.normalize();
        // 180 degree rotation about axis
        return T(2) * axis * axis.transpose() - Eigen::Matrix<T, 3, 3>::Identity();
    }

    // General case: use Rodrigues formula
    Eigen::Matrix<T, 3, 1> v = f.cross(t);
    T s = v.norm();  // sin(angle)
    T c = dot;       // cos(angle)

    Eigen::Matrix<T, 3, 3> vx = skew_sym_mat(v);

    // R = I + vx + vx^2 * (1-c)/s^2
    return Eigen::Matrix<T, 3, 3>::Identity() + vx + vx * vx * ((T(1) - c) / (s * s));
}

// Namespace for additional SO3 utilities
namespace so3_math {

/**
 * @brief Convert rotation matrix to Euler angles (ZYX convention)
 * Alias for RotMtoEuler with cleaner naming
 */
template<typename T>
inline Eigen::Matrix<T, 3, 1> rotationToEuler(const Eigen::Matrix<T, 3, 3>& R) {
    return RotMtoEuler(R);
}

} // namespace so3_math

} // namespace slam

#endif // SLAM_SO3_MATH_HPP
