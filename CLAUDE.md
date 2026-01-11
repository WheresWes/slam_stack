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
