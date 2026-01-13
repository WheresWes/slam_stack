# SLAM Stack - Project Architecture & Design Notes

## Overview

This project is a **ROS-free port of FAST-LIO2** with custom extensions for the Livox Mid-360 LiDAR.

## Core Architecture Decisions

### 1. FAST-LIO2 as the Foundation

The SLAM engine is **directly based on FAST-LIO2**, NOT a custom reimplementation. The following components are direct ports:

| Original FAST-LIO2 | Our Port | Status |
|-------------------|----------|--------|
| `laserMapping.cpp` | `slam_engine.hpp` | Direct port, ROS removed |
| `IMU_Processing.hpp` | `imu_processing.hpp` | Direct port + gravity alignment fix |
| `use_ikfom.hpp` | `use_ikfom.hpp` | Identical |
| `ikd-Tree/` | `ikd-Tree/` | Identical |
| `IKFoM_toolkit/` | `IKFoM_toolkit/` | Identical |

### 2. The "ICP" in FAST-LIO is NOT Traditional ICP

FAST-LIO uses an **Iterated Extended Kalman Filter (IEKF)** with point-to-plane residuals. The key function is:
```cpp
kf_.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
```

This performs:
1. Transform points to world frame using current state estimate
2. Query ikd-tree for nearest neighbors
3. Fit planes to neighbors
4. Compute point-to-plane residuals
5. Build measurement Jacobian H
6. EKF update step
7. Iterate until convergence (default 2-4 iterations)

This is mathematically different from traditional ICP (no explicit correspondence search/transform loop).

### 3. Livox Mid-360 Native Driver

We have a **custom native driver** for the Livox Mid-360 that:
- Communicates directly with the LiDAR via UDP (no Livox SDK)
- Handles the proprietary packet format
- Provides point cloud + IMU data in our native format
- Location: `slam_stack/livox_driver/` (or examples using direct UDP)

### 3.1 Mid-360 Specifications

| Parameter | Value |
|-----------|-------|
| Horizontal FOV | 360° (rotating prism) |
| Vertical FOV | 59° (-7° to +52°) |
| Detection Range | 40m (70m max @ 10% reflectivity) |
| Point Rate | 200,000 pts/s |
| IMU Rate | 200 Hz |
| IP Address | 192.168.1.1XX (XX = last 2 digits of serial) |

**Note on `fov_deg` parameter**: The original FAST-LIO `mid360.yaml` has `fov_degree: 60` which refers to the **vertical FOV** (59°). However, this parameter is **dead code** - it's computed into `HALF_FOV_COS` but never used in the SLAM algorithm. Our implementation keeps it for documentation purposes only.

### 4. Operating Modes

The system supports three modes:

#### A. SLAM Mode (Default)
- Full FAST-LIO2 operation
- Builds map while localizing
- Outputs: Map (PLY), Trajectory (PLY)

#### B. Localization-Only Mode
- Load pre-built map via `loadMap()`
- Enable with `setLocalizationMode(true)`
- Uses IEKF for odometry, but doesn't add new points to map
- Based on FAST-LIO-Localization approach

#### C. Global Re-localization
- For initial pose estimation when starting localization
- Uses coarse-to-fine ICP against pre-built map
- Call `globalRelocalize()` with optional initial guess
- Refines pose, then hands off to IEKF localization

### 5. CRITICAL: Localization Flyaway Issue (January 2026)

**Problem**: Our localization mode causes "flyaway" (pose divergence) when moving through areas with sparse pre-built map coverage.

**Root Cause**: Architectural difference from original FAST-LIO-Localization.

#### Original FAST-LIO-Localization Architecture (Reference: `fast_lio_localization/`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  ORIGINAL FAST-LIO-LOCALIZATION                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PROCESS 1: laserMapping.cpp (C++)           PROCESS 2: global_localization.py │
│  ┌────────────────────────────────┐          ┌────────────────────────┐│
│  │ • FULL FAST-LIO SLAM - ALWAYS  │          │ • Runs at 0.5 Hz       ││
│  │ • map_incremental() ALWAYS     │  ◄──────►│ • ICP scan-to-map      ││
│  │ • Builds LOCAL ikd-tree        │          │ • Computes T_map_to_odom│
│  │ • Outputs /Odometry (odom)     │          │ • Fitness threshold 95%││
│  └────────────────────────────────┘          └────────────────────────┘│
│                │                                        │              │
│                └──────────────────┬─────────────────────┘              │
│                                   ▼                                    │
│                    transform_fusion: pose = T_map_to_odom × T_odom     │
│                                                                        │
│  KEY: FAST-LIO NEVER stops building its local map!                     │
│       Global localization is a TRANSFORM CORRECTION on top.            │
└────────────────────────────────────────────────────────────────────────┘
```

#### Our Implementation (FIXED - January 2026)

The localization flyaway issue has been **resolved** by implementing transform fusion architecture:

```cpp
// slam_engine.hpp - FIXED IMPLEMENTATION:
// 1. mapIncremental() is ALWAYS called (never skipped)
mapIncremental();  // ALWAYS called, never skipped

// 2. Periodic global localization updates T_map_to_odom
if (localization_mode_ && prebuilt_map_loaded_) {
    if (current_time - last_global_localization_time_ >= LOCALIZATION_PERIOD_S) {
        attemptGlobalLocalizationUpdate();  // ICP local→prebuilt map
    }
}

// 3. Output pose applies transform fusion
M4D getPose() const {
    M4D local_pose = getLocalPose();  // Raw SLAM pose
    if (localization_mode_ && transform_initialized_) {
        return T_map_to_odom_ * local_pose;  // Corrected pose
    }
    return local_pose;
}

// 4. Safety checks in attemptGlobalLocalizationUpdate():
// - Fitness threshold: 90% required (LOCALIZATION_FITNESS_MIN)
// - Pose jump rejection: > 1m jump rejected (POSE_JUMP_THRESHOLD)
```

#### Implementation Summary

| Aspect | Original FAST-LIO-Localization | Our FIXED Implementation | Status |
|--------|-------------------------------|---------------------------|--------|
| Map Updates | ALWAYS enabled | ✅ ALWAYS enabled | ✅ FIXED |
| Architecture | Two-process | Single-process with transform fusion | ✅ FIXED |
| Transform Fusion | Yes (T_map_to_odom × odom) | ✅ Yes (T_map_to_odom × local_pose) | ✅ FIXED |
| Fitness Threshold | 95% required | ✅ 90% required | ✅ FIXED |
| Pose Jump Rejection | Implicit | ✅ Explicit 1m threshold | ✅ FIXED |
| Fallback on Failure | Keeps old transform | ✅ Keeps old transform | ✅ FIXED |

**Key Files Changed:**
- `slam_engine.hpp`: Transform fusion implementation, periodic ICP, pose jump detection

**Reference**: Original FAST-LIO-Localization is in `C:\Users\wmuld\OneDrive\Desktop\Documents\ATLASCpp\fast_lio_localization\`

## Key Implementation Files

```
slam_stack/
├── slam_engine/
│   ├── include/slam/
│   │   ├── slam_engine.hpp      # Main SLAM class (port of laserMapping.cpp)
│   │   ├── imu_processing.hpp   # IMU processing (port of IMU_Processing.hpp)
│   │   ├── use_ikfom.hpp        # EKF state manifold definitions
│   │   ├── types.hpp            # Custom point/state types
│   │   ├── preprocess.hpp       # Point cloud preprocessing
│   │   ├── icp.hpp              # Traditional ICP for global localization
│   │   ├── ikd-Tree/            # Incremental KD-tree (direct copy)
│   │   └── IKFoM_toolkit/       # EKF on manifolds (direct copy)
│   └── src/
│       └── visualization.cpp    # Rerun visualization (PIMPL pattern)
├── examples/
│   ├── live_slam.cpp            # Real-time SLAM with live LiDAR
│   └── replay_slam.cpp          # Offline SLAM from recorded data
└── CMakeLists.txt
```

## Performance Notes

### Real-time Requirements
- Mid-360 scan rate: 10 Hz (100ms per scan)
- Target SLAM time: <50ms per scan
- ICP (IEKF update) time scales with:
  - Number of points going into matching (~2000 optimal)
  - Map size (ikd-tree query time)
  - Number of iterations (2 recommended for real-time)

### Optimizations Applied
- `max_iterations = 2` (reduced from 4)
- `max_points_icp = 2000` (limits points in IEKF matching)
- OpenMP parallelization enabled (`#pragma omp parallel for`)

## Reference Repositories

- **FAST-LIO2**: https://github.com/hku-mars/FAST_LIO
- **FAST-LIO-Localization**: https://github.com/HViktorTsoi/FAST_LIO_LOCALIZATION
- **ikd-Tree**: https://github.com/hku-mars/ikd-Tree
- **IKFoM**: https://github.com/hku-mars/IKFoM

## Debugging Guidelines

### CRITICAL: When Issues Arise, Compare with Original FAST-LIO

The original FAST-LIO Mid-360 repository is at:
```
C:\Users\wmuld\OneDrive\Desktop\Documents\ATLASCpp\fast_lio_mid360\
```

**Always compare our implementation against the original when debugging:**
1. `src/laserMapping.cpp` - Main SLAM loop
2. `src/IMU_Processing.hpp` - IMU processing and undistortion
3. `include/use_ikfom.hpp` - State definitions
4. `config/mid360.yaml` - Default parameters

### Known Issues and Investigations

1. **~~Map Point Corruption~~** (RESOLVED - was analysis script bug)
   - Symptom appeared to show points at 100-160m but was caused by `analyze_ply.py` reading 12 bytes/point instead of 16 bytes (missing intensity field)
   - The PLY export is correct - verified with fixed analyzer
   - Debug logging added to `mapIncremental()` confirms all points transform correctly (body_max ~5.5m, world_max ~5.6m)

2. **~~IMU Acceleration Units~~** (RESOLVED - January 2026)
   - **Problem**: Livox Mid-360 outputs acceleration in **g units**, not m/s²
   - **Symptom**: Gravity magnitude showing ~1.0 instead of ~9.81, causing tracking loss during motion
   - **Fix**: In `live_slam.cpp` and `replay_slam.cpp`, multiply acceleration by 9.81:
     ```cpp
     constexpr double G_M_S2 = 9.81;
     imu.acc = frame.accel * G_M_S2;
     ```
   - **Note**: Gyroscope is already in rad/s (no conversion needed)
   - Reference: [Livox ETH Protocol](https://github.com/Livox-SDK/livox_wiki_en/blob/master/source/tutorials/new_product/mid360/livox_eth_protocol_mid360.md)

3. **~~Per-Point Timestamps~~** (RESOLVED - January 2026)
   - **Problem**: All points in a packet were getting the same timestamp, breaking deskewing
   - **Cause**: Native driver wasn't using `time_interval` from packet header
   - **Fix**: In `livox_mid360.hpp`, calculate per-point offset:
     ```cpp
     float total_time_us = hdr->time_interval * 0.1f;  // time_interval is in 0.1us
     float point_time_us = (i * total_time_us / (num_points - 1));
     ```
   - Points in a packet are equally spaced in time across the `time_interval`

4. **IMU Initialization Requirements**
   - Sensor MUST be stationary during first ~2 seconds
   - If gyro bias > 0.02 rad/s, initialization was likely bad
   - Normal gyro bias: ~0.01 rad/s

5. **Deskew (Motion Compensation)**
   - Must be ENABLED for motion tracking (`config.deskew_enabled = true`)
   - Requires proper IMU-LiDAR timestamp synchronization
   - Requires per-point timestamps (see fix #3 above)

6. **~~Fuzzy Map During Slow Motion~~** (RESOLVED - January 2026)
   - **Problem**: Map becomes fuzzy/noisy when moving the LiDAR slowly
   - **Root Causes Identified**:
     1. `gyr_cov = 0.01` was 10x too low (should be 0.1)
     2. `point_filter_num = 1` kept all points (should be 3 = keep every 3rd)
     3. `voxel_size = 0.1m` was too fine for noise filtering
     4. `blind = 0.1m` was too close (allows noisy near-field returns)
     5. No tag filtering (should filter to 0x00 and 0x10 returns only)
     6. `max_points_icp = 2000` was too low (reduces matching quality)
   - **Fixes Applied** (indoor-optimized settings based on FAST-LIO):
     ```cpp
     // live_slam.cpp
     config.gyr_cov = 0.1;           // Was 0.01 - critical fix!
     config.max_iterations = 3;       // Was 4
     voxel_size = 0.2;               // Was 0.1 (0.2m good for indoor)
     blind_distance = 0.5;           // Was 0.1 (0.5m for indoor)
     POINT_FILTER_NUM = 3;           // Keep every 3rd point
     // Added tag filtering: only 0x00/0x10 returns

     // slam_engine.hpp
     max_points_icp = 5000;          // Was 2000
     max_map_points = 500000;        // Was 200000
     cube_len = 200.0;               // Was 50.0
     ```
   - Reference: Original FAST-LIO mid360.launch parameters

7. **~~LiDAR Discovery After Disconnect~~** (RESOLVED - January 2026)
   - **Problem**: After disconnecting/reconnecting the LiDAR, broadcast discovery fails
   - **Root Causes**:
     1. LiDAR left in "Normal" (streaming) mode doesn't respond to discovery
     2. No graceful shutdown - sockets closed without stopping streaming
     3. No fallback mechanism when broadcast fails
   - **Fixes Applied**:
     1. **Graceful shutdown**: `stop()` now sends standby command before closing sockets
     ```cpp
     // livox_mid360.hpp stop() method
     sendParamCommandQuiet(PARAM_POINT_SEND_EN, 0);  // Disable streaming
     sendParamCommandQuiet(PARAM_WORK_MODE, 0x01);   // Set to Standby mode
     ```
     2. **Direct IP probing**: If broadcast fails, probes 192.168.1.100-199 directly
     3. **Wake-up mechanism**: Sends standby command to all IPs before probing to wake stuck devices
   - **Usage**: Discovery should now work reliably after disconnect. If still failing:
     - Power cycle the LiDAR (10+ second wait after power-on)
     - Use `--device 192.168.1.1XX` to specify IP directly

### Data Recording for Offline Testing

Use `--record` flag to save raw data:
```bash
./live_slam.exe --device 192.168.1.144 --record motion_test.bin --time 30
```

Replay with:
```bash
./replay_slam.exe --input motion_test.bin
```

## VESC 6 Edu CAN Bus Control

### Hardware Setup
- **Motor Controllers**: 2x VESC 6 Edu (with IMU)
- **CAN Adapter**: CANable (USB to CAN) on COM3
- **CAN Bitrate**: 500 kbps (VESC default)
- **VESC IDs**: ID 1 (left) and ID 126 (right) (verified January 2026)

### Robot Geometry (4-wheel skid-steer)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Track Width (outer) | 160 mm | Outer edge to outer edge of treads |
| Track Width (center) | 120 mm | Center-to-center of treads |
| Wheel Base | 95 mm | Front axle to rear axle |
| Tread Width | 40 mm | Width of each tread |
| Effective Track | ~156 mm | Includes scrub factor (~1.3x for skid-steer) |

**Notes**:
- 4 wheels total (2 left, 2 right)
- Left wheels connected to VESC ID 1 (single motor drives both)
- Right wheels connected to VESC ID 126 (single motor drives both)
- Skid-steer turning: rotate by driving wheels in opposite directions

### Odometry Calibration (January 2026)
- **Ticks per meter**: 14,093 avg (Left: 14,052, Right: 14,133)
- **Calibration distance**: 2050mm
- **Method**: Open-loop duty cycle drive, measured tachometer delta

### Motor Control Notes
- Motors have asymmetric efficiency - right motor ~13-61% faster at same duty

### Hybrid Control Architecture (CRITICAL)

The VESC PID controller is **unstable below ~2000 ERPM** due to noisy/infrequent encoder feedback at low speeds. This requires a **hybrid control strategy**:

| Speed Range | Control Mode | Why |
|-------------|--------------|-----|
| **< 2000 ERPM** | Open-loop duty cycle | PID unstable, use calibrated duty scaling |
| **≥ 2000 ERPM** | Closed-loop RPM (PID) | PID stable, VESC handles motor asymmetry |

**Transition Logic:**
```
┌─────────────────────────────────────────────────────────────────┐
│  STOPPED                                                         │
│    │                                                             │
│    ▼ Start motion                                                │
│  OPEN-LOOP DUTY (with calibration scaling)                       │
│    │                                                             │
│    │ Monitor ERPM from STATUS messages                           │
│    │                                                             │
│    ▼ ERPM ≥ 2000 (both wheels)                                   │
│  CLOSED-LOOP RPM (VESC PID, no scaling needed)                   │
│    │                                                             │
│    │ ERPM drops below 1800 (hysteresis)                          │
│    ▼                                                             │
│  OPEN-LOOP DUTY (with calibration scaling)                       │
│    │                                                             │
│    ▼ Duty → 0                                                    │
│  STOPPED                                                         │
└─────────────────────────────────────────────────────────────────┘
```

**Key Implementation Points:**
1. **Duty scaling calibration** only applies during open-loop mode (< 2000 ERPM)
2. **RPM control** handles motor asymmetry automatically via VESC PID
3. **Hysteresis** (2000 up, 1800 down) prevents oscillation at transition
4. **Smooth handoff**: When transitioning duty→RPM, calculate target RPM from current ERPM
5. **Both wheels must be above threshold** before switching to RPM mode

**Why This Matters:**
- Test showed rightward drift during reverse at ~3000+ ERPM
- At that speed, should have been in RPM mode (no duty scaling)
- Duty scaling was incorrectly applied, causing the asymmetry

**ERPM Reference (from calibration data):**
| Duty | Left ERPM | Right ERPM |
|------|-----------|------------|
| 0.030 | 891 | 1433 |
| 0.035 | 1563 | 1954 |
| 0.040 | 2046 | 2508 |  ← Transition zone
| 0.050 | 2801 | 3230 |
| 0.060 | 3877 | 4416 |
| 0.070 | 4645 | 5259 |

At duty ~0.04, we cross the 2000 ERPM threshold and should switch to RPM control.

### Per-Wheel Duty Scaling Calibration

**Problem**: Left and right motors have different efficiency. At same duty cycle, right wheel travels faster.

**Calibration Data (January 2026)**:
| Duty | L ERPM | R ERPM | R/L Ratio | Scale R |
|------|--------|--------|-----------|---------|
| 0.030 | 891 | 1433 | 1.61x | 0.622 |
| 0.035 | 1563 | 1954 | 1.25x | 0.800 |
| 0.040 | 2046 | 2508 | 1.23x | 0.816 |
| 0.050 | 2801 | 3230 | 1.15x | 0.867 |
| 0.060 | 3877 | 4416 | 1.14x | 0.878 |
| 0.070 | 4645 | 5259 | 1.13x | 0.883 |

**Minimum Duty Thresholds**:
| Threshold | Left | Right | Description |
|-----------|------|-------|-------------|
| Starting | 0.040 | 0.035 | From standstill (static friction) |
| Keeping | 0.030 | 0.020 | While moving (kinetic friction) |

**Algorithm**:
```python
def set_duty(target_duty, is_moving):
    # 1. Apply interpolated scaling to right wheel
    scale = interpolate(target_duty, CALIBRATION_TABLE)
    duty_left = target_duty
    duty_right = target_duty * scale

    # 2. Check minimum thresholds
    if is_moving:
        min_duty = MIN_DUTY_KEEP  # 0.030
    else:
        min_duty = MIN_DUTY_START  # 0.040

    # 3. If EITHER wheel below threshold, stop BOTH
    #    (prevents one wheel moving while other is stuck)
    if duty_left < min_duty or duty_right < min_duty:
        duty_left = 0
        duty_right = 0
        is_moving = False
    else:
        is_moving = True

    send_duty(VESC_LEFT, duty_left)
    send_duty(VESC_RIGHT, duty_right)
    return is_moving
```

**Calibration Procedure** (for GUI):
1. **Duty Scaling Calibration**:
   - Run at multiple duty levels (0.03 to 0.07) using open-loop duty
   - Measure resulting ERPM for each wheel
   - Calculate scale_R = ERPM_L / ERPM_R for each point
   - Store as lookup table for interpolation

2. **Minimum Duty Calibration**:
   - From standstill, increase duty until each wheel starts moving
   - Record as MIN_DUTY_START_LEFT and MIN_DUTY_START_RIGHT
   - Use higher of the two as the system threshold

3. **Odometry Calibration**:
   - Drive known distance (1-2m) with compensated duty
   - Record tachometer delta for each wheel
   - Calculate ticks_per_meter = delta_tach / distance

**Test Result**: With calibration applied, wheels travel within **0.2%** of each other.

### CAN Frame Structure (29-bit Extended ID)

```
┌─────────────────┬────────────────┬─────────────┐
│ Unused (B28-16) │ Command (B15-8)│ VESC ID (B7-0)│
└─────────────────┴────────────────┴─────────────┘
Data: 4-8 bytes, BIG-ENDIAN
```

### Control Commands (TX → VESC)

| Command | ID | Scaling | Unit | Notes |
|---------|-----|---------|------|-------|
| SET_DUTY | 0 | ×100,000 | -1.0 to 1.0 | PWM duty cycle |
| SET_CURRENT | 1 | ×1,000 | Amps | Torque control |
| SET_CURRENT_BRAKE | 2 | ×1,000 | Amps | Regenerative braking |
| SET_RPM | 3 | ×1 | ERPM | Velocity control |
| SET_POS | 4 | ×1,000,000 | Degrees | Position control |
| SET_CURRENT_REL | 10 | ×100,000 | -1.0 to 1.0 | Relative to max |
| SET_CURRENT_BRAKE_REL | 11 | ×100,000 | -1.0 to 1.0 | Relative brake |

**Encoding Example (10A current to VESC ID 1):**
```cpp
uint32_t can_id = (1 << 8) | 1;  // Command=1, VESC_ID=1 → 0x00000101
int32_t value = 10 * 1000;       // 10A × 1000 = 10000
uint8_t data[4] = {
    (value >> 24) & 0xFF,  // MSB first (big-endian)
    (value >> 16) & 0xFF,
    (value >> 8) & 0xFF,
    value & 0xFF
};
// Send extended CAN frame: ID=0x101, data=[0x00, 0x00, 0x27, 0x10]
```

### Status Messages (RX ← VESC)

**Must enable in VESC Tool**: App Settings → General → CAN Status Messages Rate

#### STATUS (ID 9) - Primary Telemetry
| Bytes | Field | Scale | Unit |
|-------|-------|-------|------|
| 0-3 | ERPM | ÷1 | Electrical RPM |
| 4-5 | Current | ÷10 | Amps |
| 6-7 | Duty Cycle | ÷1000 | -1.0 to 1.0 |

#### STATUS_4 (ID 16) - Temperatures
| Bytes | Field | Scale | Unit |
|-------|-------|-------|------|
| 0-1 | FET Temp | ÷10 | °C |
| 2-3 | Motor Temp | ÷10 | °C |
| 4-5 | Input Current | ÷10 | Amps |
| 6-7 | PID Position | ÷50 | Degrees |

#### STATUS_5 (ID 27) - Odometry & Voltage
| Bytes | Field | Scale | Unit |
|-------|-------|-------|------|
| 0-3 | Tachometer | ÷6 | Electrical revs |
| 4-5 | Input Voltage | ÷10 | Volts |

**Decoding Example:**
```cpp
void decodeStatus5(uint8_t* data, float& voltage, int32_t& tachometer) {
    tachometer = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
    int16_t voltage_raw = (data[4] << 8) | data[5];
    voltage = voltage_raw / 10.0f;
}
```

### Wheel Odometry Calculation

```cpp
// Motor parameters (set these for your motors)
const int MOTOR_POLES = 14;           // Typical for hub motors
const float GEAR_RATIO = 1.0f;        // Direct drive = 1.0
const float WHEEL_DIAMETER_M = 0.165f; // 165mm wheel
const float WHEEL_CIRCUMFERENCE = M_PI * WHEEL_DIAMETER_M;

// Convert tachometer to distance
float tachToDistance(int32_t tach_value) {
    float motor_revs = tach_value / (MOTOR_POLES * 3.0f);
    float wheel_revs = motor_revs / GEAR_RATIO;
    return wheel_revs * WHEEL_CIRCUMFERENCE;
}

// Convert ERPM to wheel velocity (m/s)
float erpmToVelocity(int32_t erpm) {
    float motor_rpm = erpm / (MOTOR_POLES / 2.0f);
    float wheel_rpm = motor_rpm / GEAR_RATIO;
    return (wheel_rpm / 60.0f) * WHEEL_CIRCUMFERENCE;
}
```

### IMU Data Access

VESC 6 Edu has onboard IMU (BMI160 or LSM6DS3). Access via UART command, not CAN status:
- Send: `COMM_GET_IMU_DATA` (ID=65) with mask 0xFFFF
- Response: roll, pitch, yaw, accX, accY, accZ, gyrX, gyrY, gyrZ

For CAN-only systems, configure VESC to forward IMU data or use LiDAR IMU instead.

### CANable Setup (Windows)

#### Option 1: SLCAN Firmware (Default)
```python
import can
bus = can.Bus(interface='slcan', channel='COM3', bitrate=500000)
```

#### Option 2: Candlelight Firmware (Better Performance)
1. Flash candlelight firmware via DFU
2. Install Zadig, set driver to WinUSB
3. Use gs_usb interface:
```python
bus = can.Bus(interface='gs_usb', channel=0, bitrate=500000)
```

### Critical Implementation Notes

1. **Timeout**: Motor stops after 0.5s without commands. Send at ≥50Hz (20ms interval)

2. **Command Loop Pattern**:
   ```cpp
   while (running) {
       sendRPMCommand(VESC_LEFT, left_rpm);
       sendRPMCommand(VESC_RIGHT, right_rpm);
       std::this_thread::sleep_for(std::chrono::milliseconds(20));
   }
   ```

3. **Differential Drive**:
   - Convention: Positive RPM = forward rotation
   - For tank steering: invert one motor's direction in VESC Tool, OR negate RPM in software

4. **Safety**: Always implement E-stop that sends zero current/duty to both VESCs

5. **Status Message Rates** (configure in VESC Tool):
   - STATUS (ID 9): 50 Hz (ERPM for odometry)
   - STATUS_4 (ID 16): 10 Hz (temperatures for monitoring)
   - STATUS_5 (ID 27): 50 Hz (tachometer for odometry, voltage)

### Reference Implementation

See these libraries for implementation patterns:
- **libVescCan**: https://github.com/AlvaroBajceps/libVescCan
- **vesc_can_bus_arduino**: https://github.com/craigg96/vesc_can_bus_arduino
- **Official VESC CAN docs**: https://github.com/vedderb/bldc/blob/master/documentation/comm_can.md

### C++ Class Structure (Implemented)

Located in `slam_engine/include/slam/` and `slam_engine/src/`:

```cpp
// vesc_can_interface.hpp - Low-level SLCAN communication
class VescCanInterface {
public:
    bool open(const std::string& port, int bitrate = 500000);
    void close();
    bool send(const CanMessage& msg);
    void setReceiveCallback(CanMessageCallback callback);
    // Statistics
    uint64_t getMessagesSent() const;
    uint64_t getMessagesReceived() const;
};

// vesc_driver.hpp - High-level motor controller
class VescDriver {
public:
    bool init(const std::string& port, uint8_t vesc_left = 1, uint8_t vesc_right = 126);
    void shutdown();

    // Motor Control (with calibration)
    void setDuty(float duty, bool apply_scaling = true);
    void setDutyRaw(float duty_left, float duty_right);
    void setDutyDifferential(float linear, float angular);
    void stop();
    void rampTo(float target_duty, float duration_s = 1.0f);

    // Odometry
    WheelOdometry getOdometry() const;
    void resetOdometry();

    // Calibration
    bool loadCalibration(const std::string& path);
    bool saveCalibration(const std::string& path) const;
    CalibrationResult runDutyCalibration(ProgressCallback);
    CalibrationResult runMinDutyCalibration(ProgressCallback);
    CalibrationResult runRotationCalibration(ImuCallback, ProgressCallback);
    CalibrationResult runCalibrationWithLocalization(
        ImuCallback, PointCloudCallback, ProgressCallback);
};
```

**Key structs**:
- `RobotGeometry` - track width, wheel base, effective track
- `WheelCalibration` - per-wheel duty scaling table, ticks/meter
- `WheelOdometry` - distance, velocity, tachometer readings
- `VescStatus` - ERPM, current, temperature, voltage

## Sensor Fusion: Wheel Odometry + SLAM

### Problem Statement

The robot uses two sensor modalities with complementary characteristics:

| Sensor | Strengths | Weaknesses |
|--------|-----------|------------|
| **Wheel Odometry** | Perfect stationary detection, accurate linear velocity, no jitter | Unreliable rotation (skid-steer slip), drift over time |
| **LiDAR SLAM** | Accurate absolute position, correct rotation | Jittery (especially during motion), latency |

**Goal**: Fuse these sensors to get smooth, accurate pose estimation with:
- Zero false motion when stationary (reject SLAM jitter)
- Accurate linear velocity from wheels during straight-line travel
- Accurate rotation from SLAM during turns (wheel slip makes odom unreliable)
- Long-term drift correction from SLAM

### Skid-Steer Kinematics Reality

For differential/skid-steer robots, wheel encoders measure **wheel rotation**, not robot motion:

```
STRAIGHT-LINE TRAVEL:
  - Wheels roll without slipping
  - Encoder ticks ≈ actual distance traveled
  - ✓ Wheel odometry RELIABLE

TURNING (IN-PLACE OR DURING MOTION):
  - Wheels scrub sideways against the ground
  - Significant slip occurs
  - Encoder ticks ≠ actual rotation
  - ✗ Wheel odometry UNRELIABLE for rotation
  - Example: Encoders report 90° turn, actual rotation was 60°
```

**Key Insight**: During turns, trust SLAM for rotation. During straight-line, trust wheel odometry for velocity.

### Motion State Detection

The fusion algorithm uses three motion states:

```cpp
enum class MotionState {
    STATIONARY,      // Wheels not moving
    STRAIGHT_LINE,   // Moving forward/backward, minimal steering
    TURNING          // Significant angular velocity
};
```

**Detection Logic:**
```cpp
if (|ERPM_left| < 50 && |ERPM_right| < 50) {
    state = STATIONARY;
} else if (|angular_command| < 0.05 rad/s) {
    state = STRAIGHT_LINE;
} else {
    state = TURNING;
}
```

### Constraint-Based Fusion Algorithm

#### STATIONARY State
```
┌────────────────────────────────────────────────────────────┐
│ WHEEL ODOM: 100% trust - wheels say no motion              │
│ SLAM:       IGNORED - reject all position/heading updates  │
│ OUTPUT:     Frozen pose (eliminates SLAM jitter)           │
└────────────────────────────────────────────────────────────┘
```
**Rationale**: If the wheels aren't turning, the robot hasn't moved. Period. SLAM might report ±2cm jitter due to measurement noise, but we have perfect information from the wheels.

#### STRAIGHT_LINE State
```
┌────────────────────────────────────────────────────────────┐
│ WHEEL ODOM: Trust linear velocity, IGNORE angular velocity │
│             Apply "no lateral motion" constraint           │
│ SLAM:       Low-pass filter, slow correction               │
│ OUTPUT:     Dead-reckoning + gradual SLAM correction       │
└────────────────────────────────────────────────────────────┘

Position Update:
  forward_delta = wheel_linear_velocity × dt
  pose.x += forward_delta × cos(pose.theta)
  pose.y += forward_delta × sin(pose.theta)

  // NO lateral motion - physical constraint
  lateral_delta = 0  // Robot cannot slide sideways

Heading Correction (slow):
  heading_error = slam_smoothed.theta - pose.theta
  pose.theta += heading_error × HEADING_RATE × dt
  // HEADING_RATE ≈ 0.5/s (corrects ~50% per second)

Position Correction (slow):
  pos_error = slam_smoothed - pose
  pose += pos_error × POSITION_RATE × dt
  // POSITION_RATE ≈ 0.3/s (corrects ~30% per second)
```
**Rationale**: Wheel velocity is very accurate during straight travel (no slip). The "no lateral motion" constraint rejects any sideways jitter from SLAM. Slow corrections handle long-term drift.

#### TURNING State
```
┌────────────────────────────────────────────────────────────┐
│ WHEEL ODOM: IGNORED - slip makes it unreliable             │
│ SLAM:       Trust for rotation, moderate trust for position│
│ OUTPUT:     Follows SLAM (smoothed)                        │
└────────────────────────────────────────────────────────────┘

Heading Update:
  pose.theta = slam_smoothed.theta  // Trust SLAM completely

Position Update:
  pose.x = lerp(pose.x, slam_smoothed.x, 0.8 × dt)
  pose.y = lerp(pose.y, slam_smoothed.y, 0.8 × dt)
  // Faster correction rate since wheel odom is unreliable
```
**Rationale**: During skid-steer turns, wheel encoders report garbage for rotation. SLAM handles this correctly using LiDAR feature matching. Even position from wheels is questionable during turns.

### SLAM Jitter Smoothing

Apply heavy low-pass filtering to raw SLAM output:

```cpp
// Alpha = 0.15 → 85% old, 15% new
// At 10Hz SLAM, this gives ~0.5s smoothing time constant
slam_smoothed.x = lerp(slam_smoothed.x, slam_raw.x, 0.15);
slam_smoothed.y = lerp(slam_smoothed.y, slam_raw.y, 0.15);
slam_smoothed.theta = lerpAngle(slam_smoothed.theta, slam_raw.theta, 0.20);
```

**Why different alpha for heading?** Heading changes from SLAM are more reliable (less noise) than position changes, so we can use slightly less smoothing.

### Hull Surface Constraint (Deployment)

In the final deployment on boat hulls, an additional powerful constraint applies:

```
┌────────────────────────────────────────────────────────────┐
│ CONSTRAINT: Robot is ALWAYS on the hull surface mesh       │
│                                                            │
│ This provides:                                             │
│ 1. Z position locked to mesh surface                       │
│ 2. Roll/Pitch aligned to surface normal                    │
│ 3. Velocity constrained to surface tangent plane           │
│                                                            │
│ Reduces 6DOF → 3DOF (x_surface, y_surface, heading)        │
└────────────────────────────────────────────────────────────┘
```

**Implementation:**
```cpp
struct HullConstraint {
    // Mesh representation of hull surface
    std::shared_ptr<TriangleMesh> hull_mesh;

    // Project any 3D point onto the hull surface
    Eigen::Vector3f projectToSurface(const Eigen::Vector3f& point) const;

    // Get surface normal at a point (for roll/pitch)
    Eigen::Vector3f getSurfaceNormal(const Eigen::Vector3f& point) const;

    // Constrain velocity to surface tangent plane
    Eigen::Vector3f constrainVelocity(
        const Eigen::Vector3f& position,
        const Eigen::Vector3f& velocity) const;
};

void SensorFusion::applyHullConstraint() {
    if (!hull_constraint_) return;

    // 1. Project position onto hull surface
    Eigen::Vector3f pos_3d(fused_pose_.x, fused_pose_.y, fused_pose_.z);
    pos_3d = hull_constraint_->projectToSurface(pos_3d);
    fused_pose_.x = pos_3d.x();
    fused_pose_.y = pos_3d.y();
    fused_pose_.z = pos_3d.z();

    // 2. Align orientation to surface normal
    Eigen::Vector3f normal = hull_constraint_->getSurfaceNormal(pos_3d);
    // Compute roll/pitch from normal, keep yaw from fusion
    alignToSurfaceNormal(normal);

    // 3. Constrain velocity to tangent plane
    fused_velocity_ = hull_constraint_->constrainVelocity(pos_3d, fused_velocity_);
}
```

**Benefits of Hull Constraint:**
1. **Reject vertical drift**: SLAM might drift in Z, but robot can't leave the hull
2. **Stabilize orientation**: Roll/pitch locked to hull geometry
3. **Improve position accuracy**: Reduces search space for SLAM matching
4. **Physical realism**: Fused pose is always a valid robot configuration

### Data Flow Architecture

```
┌─────────────────────┐       ┌─────────────────────┐
│  MotionController   │       │     SlamEngine      │
│      (50 Hz)        │       │      (10 Hz)        │
│                     │       │                     │
│ • wheel velocity    │       │ • SLAM pose (6DOF)  │
│ • ERPM feedback     │       │ • covariance        │
│ • motion commands   │       │                     │
│ • motion state      │       │                     │
└──────────┬──────────┘       └──────────┬──────────┘
           │                             │
           │ updateWheelState()          │ updateSlamPose()
           │ 50 Hz                       │ 10 Hz
           ▼                             ▼
    ┌──────────────────────────────────────────────────┐
    │                 SensorFusion                      │
    │                                                   │
    │  ┌────────────────┐    ┌─────────────────────┐   │
    │  │ Motion State   │    │ SLAM Low-Pass       │   │
    │  │ Detector       │    │ Filter              │   │
    │  │                │    │                     │   │
    │  │ STATIONARY     │    │ α = 0.15 (position) │   │
    │  │ STRAIGHT_LINE  │    │ α = 0.20 (heading)  │   │
    │  │ TURNING        │    │                     │   │
    │  └───────┬────────┘    └──────────┬──────────┘   │
    │          │                        │              │
    │          ▼                        ▼              │
    │  ┌─────────────────────────────────────────┐     │
    │  │      Constraint-Based Fusion            │     │
    │  │                                         │     │
    │  │  STATIONARY:   freeze pose              │     │
    │  │  STRAIGHT:     wheel DR + SLAM correct  │     │
    │  │  TURNING:      trust SLAM               │     │
    │  └────────────────────┬────────────────────┘     │
    │                       │                          │
    │                       ▼                          │
    │  ┌─────────────────────────────────────────┐     │
    │  │      Hull Surface Constraint            │     │
    │  │      (when mesh available)              │     │
    │  │                                         │     │
    │  │  • Project to surface                   │     │
    │  │  • Align roll/pitch to normal           │     │
    │  │  • Constrain velocity to tangent        │     │
    │  └────────────────────┬────────────────────┘     │
    │                       │                          │
    └───────────────────────┼──────────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Fused Pose     │
                   │  (smooth,       │
                   │   accurate,     │
                   │   on-surface)   │
                   └─────────────────┘
```

### Tuning Parameters (OPTIMIZED January 2026)

```cpp
struct FusionConfig {
    // Motion state detection
    int stationary_erpm_threshold = 50;        // Below this = stationary
    float turning_angular_threshold = 0.2f;    // rad/s command threshold

    // SLAM low-pass filter coefficients
    // Higher alpha = faster response but more noise
    float slam_position_alpha = 0.60f;         // OPTIMAL
    float slam_heading_alpha = 0.65f;          // OPTIMAL

    // Correction rates (per second) - STRAIGHT_LINE mode
    float straight_heading_correction = 4.0f;  // OPTIMAL
    float straight_position_correction = 4.0f; // OPTIMAL

    // Correction rates (per second) - TURNING mode
    float turning_position_correction = 12.0f; // OPTIMAL

    // Hull constraint (optional)
    bool enable_hull_constraint = false;
    float hull_snap_distance = 0.1f;           // Max distance to snap to surface
};
```

### Test Results (January 2026)

**Test Scenario**: Forward ~0.9m, pause, small turns, pause, return to origin

| Metric | Wheel Odom | SLAM | **Fused** |
|--------|------------|------|-----------|
| Return-to-Origin Error | 2.3 cm | 2.9 cm | **0.1 cm** |
| Stationary Jitter | N/A | 1.02 mm | **0.00 mm** |
| Overall Jitter | N/A | 1.75 mm | 1.63 mm |

**Key Findings**:
- Fused position is **23x more accurate** than wheel odometry alone
- Fused position is **29x more accurate** than SLAM alone
- **Zero jitter when stationary** - perfect rejection of SLAM noise
- Motion state distribution: 8% STATIONARY, 73% STRAIGHT_LINE, 18% TURNING

### Offline Replay for Tuning

The sensor fusion can be tuned offline using recorded data:

```bash
# 1. Record raw sensor data (live robot)
./record_fusion_data.exe --device 192.168.1.144 --time 60 --output test_run.bin

# 2. Replay and tune parameters (offline)
./replay_fusion_data.exe --input test_run.bin
```

Recording captures: IMU @ 200Hz, Point clouds @ 10Hz, VESC status @ 50Hz, Wheel odom @ 50Hz

### Testing Methodology

1. **Stationary Jitter Test**:
   - Place robot stationary
   - Run SLAM + fusion for 60 seconds
   - Measure max deviation of fused pose (should be ~0)
   - Compare to raw SLAM deviation (may be ±2cm)

2. **Straight-Line Accuracy Test**:
   - Drive forward 2m and back
   - Compare wheel odom, SLAM, and fused return-to-origin error
   - Fused should match or beat wheel odom

3. **Turn Recovery Test**:
   - Execute 90° turns
   - Check that fused heading matches SLAM
   - Check that position doesn't jump during turn

4. **Combined Motion Test**:
   - Complex path: Forward → Turn → Forward → Turn → Return
   - Measure absolute position error vs ground truth
   - Should combine benefits of both sensors

## TODO / Future Work

- [x] ~~Debug map point corruption issue~~ (RESOLVED - was analysis script bug)
- [x] ~~Fix IMU acceleration units~~ (RESOLVED - multiply by 9.81)
- [x] ~~Fix per-point timestamps for deskewing~~ (RESOLVED - use time_interval)
- [x] ~~Fix fuzzy map during slow motion~~ (RESOLVED - gyr_cov=0.1, blind=0.5m)
- [x] ~~Fix LiDAR discovery after disconnect~~ (RESOLVED - graceful shutdown + direct IP probe)
- [ ] Add loop closure detection
- [ ] Implement proper global localization with descriptors (e.g., ScanContext)
- [ ] Multi-threaded map update (separate from SLAM thread)
- [ ] GPU acceleration for ikd-tree queries (optional)
- [x] VESC CAN bus driver for differential drive control (Python prototype complete)
  - CAN communication working (500kbps, VESC IDs 1 and 126)
  - Open-loop duty control for smooth operation
  - Per-wheel scaling calibration implemented
  - Odometry calibration complete (14,093 ticks/m)
- [x] Port VESC driver to C++ for integration with SLAM
  - vesc_can_interface.hpp/cpp - SLCAN protocol for Windows
  - vesc_driver.hpp/cpp - Full driver with calibration support
  - Duty scaling, odometry, and rotation calibration
  - Combined calibration + localization sequence
- [x] GUI calibration panel for motor control
  - Robot geometry settings
  - Per-wheel duty scaling table
  - Minimum duty thresholds
  - Calibration buttons (ready for implementation)
- [ ] Combined calibration + localization sequence (in progress)
- [x] Wheel odometry fusion with LiDAR SLAM (RESOLVED January 2026)
  - Motion-state-aware fusion: STATIONARY/STRAIGHT_LINE/TURNING
  - 0.1cm return-to-origin error (23x better than wheel odom, 29x better than SLAM)
  - Zero stationary jitter (perfect noise rejection)
  - Optimal config: alpha=0.60/0.65, correction=4.0/4.0/12.0
- [ ] **Implement hybrid motor control** (CRITICAL for straight-line tracking)
  - Open-loop duty with scaling below 2000 ERPM
  - Closed-loop RPM control above 2000 ERPM
  - Hysteresis at transition (2000 up, 1800 down)
  - Smooth handoff between modes
  - Test showed rightward drift when duty scaling applied at high ERPM

## Combined Calibration + Localization Sequence

### Concept

A single calibration run performed **before any job** that:
1. Calibrates motor duty scaling (consistent wheel speeds)
2. Calibrates rotation (ticks per radian, effective track width)
3. Builds a local point cloud map for global localization

### Movement Pattern

```
Start Position (X)
       |
       | Phase 1: Forward drive (duty calibration)
       |   - Test duty levels 0.035, 0.040, 0.050, 0.060
       |   - Measure ERPM L vs R, calculate forward scaling
       v
    [Forward ~0.5m]
       |
       | Phase 2: Reverse drive (duty calibration)
       |   - Test same duty levels in reverse
       |   - Measure ERPM L vs R, calculate reverse scaling
       |   - Return to start position
       v
    [Back to Start]
       |
       | Phase 3: Turn 90° Left (rotation calibration)
       |   - Use IMU yaw for accurate angle
       |   - Measure tachometer delta
       |   - Calculate ticks_per_radian
       v
    [Left 90°]
       |
       | Phase 4: Return to center
       v
    [Center]
       |
       | Phase 5: Turn 90° Right
       |   - Confirm rotation calibration symmetry
       v
    [Right 90°]
```

**Why reverse calibration matters:**
- Motor efficiency may differ in reverse direction
- Friction characteristics may be asymmetric
- Robot needs accurate control for backing up, parking, obstacle avoidance

### Data Collected

1. **Motor Calibration**:
   - Per-wheel duty scaling table FORWARD (duty → scale_R_fwd)
   - Per-wheel duty scaling table REVERSE (duty → scale_R_rev)
   - Minimum duty thresholds (start vs keep) for each direction
   - ERPM measurements at each duty level, both directions

2. **Rotation Calibration**:
   - Ticks per radian (tach_delta / IMU_angle)
   - Effective track width (includes scrub factor for skid-steer)
   - Left vs right turn symmetry check

3. **Localization Map**:
   - Local point cloud built during entire sequence
   - ~180° coverage (forward + left + right views)
   - Used to match against pre-built global map

### Implementation

**C++ (planned)**: `examples/calibration_run.cpp`
- Integrates VescDriver + SlamEngine + native Livox driver
- Single application for complete calibration
- Outputs: `calibration.ini` + `local_map.ply`

**Visualization**: Custom D3D11 SlamViewer (slam_viewer.cpp)
- Hardware-accelerated point cloud rendering
- Alternative to Rerun for live data (not yet tested with live SLAM)
