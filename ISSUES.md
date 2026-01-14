# SLAM Stack - Known Issues and TODO

## Critical

### 0. SLAM Flyaway Issue (Pose Divergence)
**Status**: MITIGATED (January 2026)
**Description**: SLAM pose suddenly diverges ("flies away") even when processing time is acceptable.

**Symptoms:**
- Pose suddenly jumps to incorrect location
- Often occurs after extended mapping session
- Can happen even with SLAM time < 50ms
- Map becomes corrupted after flyaway

**Mitigations Implemented:**

1. **Pose Jump Rejection** (slam_engine.hpp)
   - Position jump > 0.3m → update rejected, fallback to IMU propagation
   - Rotation jump > 20° → update rejected, fallback to IMU propagation
   - Configurable via `max_position_jump` and `max_rotation_jump_deg`

2. **Minimum Effective Points Threshold** (slam_engine.hpp)
   - If matched points < 50 → update rejected as degenerate
   - Configurable via `min_effective_points`

3. **Max Map Points Warning** (slam_engine.hpp)
   - Warns at 90% capacity instead of silently stopping
   - Continues to warn every 500 frames at capacity

**GUI Controls Added** (Settings > SLAM):
- Max Pos Jump (m) - default 0.3m
- Max Rot Jump (deg) - default 20°
- Min Effective Pts - default 50

**Still to investigate if issues persist:**
- [ ] Covariance bounds (prevent unbounded growth)
- [ ] IMU bias estimation stability
- [ ] ikd-tree edge cases

---

## High Priority

### 1. Calibration Process Overhaul
**Status**: Needs complete redesign
**Description**: The current calibration process is incomplete and unreliable.

**Required calibrations:**
- [ ] Duty scaling calibration (left vs right motor efficiency)
- [ ] Minimum duty threshold calibration (static vs kinetic friction)
- [ ] Rotation calibration (ticks per radian, effective track width)
- [ ] Odometry calibration (ticks per meter)
- [ ] Combined calibration sequence with localization

**Current issues:**
- Calibration GUI buttons are placeholders
- No persistent calibration storage
- Motor asymmetry causes drift

---

### 2. Crawler Not Tracking Straight (Closed-Loop Issue)
**Status**: Active bug
**Description**: Robot drifts when it should be driving straight in closed-loop RPM mode.

**Symptoms:**
- Rightward/leftward drift during forward travel
- Drift direction varies with speed

**Potential causes:**
- Duty scaling being applied in RPM mode (should only apply in open-loop)
- VESC PID tuning differences between motors
- Incorrect ERPM threshold for mode switching (currently 2000)
- Missing hybrid control implementation (see CLAUDE.md)

**Investigation needed:**
- Log ERPM values during straight-line travel
- Verify which control mode is active
- Check if duty scaling is incorrectly applied

---

### 3. Crawler Travelling Too Slow
**Status**: Needs tuning
**Description**: Robot maximum speed is too conservative.

**Current settings:**
- `max_duty = 0.15` (15% duty cycle)
- `max_speed = 0.5` m/s

**Tasks:**
- [ ] Test higher duty cycles safely
- [ ] Determine safe maximum speed on flat surfaces
- [ ] Consider terrain-adaptive speed limiting
- [ ] Add speed presets (slow/medium/fast)

---

### 4. Map Point Update Rate Too Slow
**Status**: By design, needs optimization
**Description**: Map visualization only updates at ~1-2 Hz instead of matching SLAM rate (10 Hz).

**Root cause** (slam_gui.cpp:754):
```cpp
// Update map points at 2Hz (every 0.5 seconds) to reduce overhead
if (elapsed < 0.5f) return;
```

**Why it's slow:**
1. `g_slam->getMapPoints()` copies entire map (500K+ points = ~10MB)
2. Two format conversion loops (RenderPoint + PointData)
3. Full GPU buffer upload each update

**Optimization options:**
- [ ] Incremental point updates (only send new points)
- [ ] Double-buffered GPU uploads
- [ ] Separate "active scan" visualization from "full map"
- [ ] LOD (level-of-detail) for distant points
- [ ] Spatial indexing for visible region only

---

### 5. Large Map Support (Beyond max_map_points)
**Status**: Architecture change needed
**Description**: Need to build maps larger than active SLAM area can handle.

**Current limitation:**
- `max_map_points = 500K` prevents ikd-tree slowdown
- But this limits total explorable area

**Proposed solution - Tile-based mapping:**
```
┌─────────────────────────────────────────────────────────┐
│                    TILE MANAGER                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Tile A   │  │ Tile B   │  │ Tile C   │  ...          │
│  │ (saved)  │  │ (active) │  │ (saved)  │               │
│  │ .ply     │  │ ikd-tree │  │ .ply     │               │
│  └──────────┘  └──────────┘  └──────────┘               │
│                     ▲                                    │
│                     │                                    │
│              Robot is here                               │
│                                                          │
│  - Only ONE tile active in ikd-tree at a time           │
│  - Adjacent tiles loaded for localization               │
│  - Distant tiles saved to disk                          │
│  - Seamless transitions as robot moves                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Implementation tasks:**
- [ ] Define tile size (e.g., 50m x 50m)
- [ ] Tile boundary detection
- [ ] Tile save/load mechanism
- [ ] Cross-tile localization during transitions
- [ ] Tile stitching for final map export
- [ ] Memory management for loaded tiles

---

## Medium Priority

### 6. Code Optimization Review
**Status**: Ongoing
**Description**: General performance audit needed.

**Areas to review:**
- [ ] SLAM engine hot paths (IEKF update, ikd-tree queries)
- [ ] Point cloud processing pipeline
- [ ] Mutex contention in worker threads
- [ ] Memory allocation patterns (avoid allocations in loops)
- [ ] GPU buffer management
- [ ] IMU processing efficiency

**Profiling needed:**
- [ ] CPU profiling during mapping
- [ ] GPU utilization analysis
- [ ] Memory allocation tracking
- [ ] Thread synchronization analysis

---

### 7. Viewer Color Scales Not Respected
**Status**: Bug
**Description**: Color scale/colormap settings in viewer are not being applied correctly.

**Symptoms:**
- Changing colormap in settings has no effect
- Points render with wrong colors
- Height-based coloring may be broken

**Investigation needed:**
- [ ] Check `slam_viewer.cpp` colormap application
- [ ] Verify shader uniform updates
- [ ] Test each colormap option
- [ ] Check z-range calculation for height coloring

**Relevant code:**
- `visualization/slam_viewer.cpp`
- `visualization/slam_viewer_impl.hpp`
- Colormap enum and shader uniforms

---

## Low Priority / Future Enhancements

### 8. Hybrid Motor Control Implementation
**Description**: Implement proper open-loop/closed-loop switching based on ERPM.

**Per CLAUDE.md spec:**
- Below 2000 ERPM: Open-loop duty with calibrated scaling
- Above 2000 ERPM: Closed-loop RPM control (VESC PID)
- Hysteresis: 2000 up, 1800 down

---

### 9. Loop Closure Detection
**Description**: Detect when robot revisits previously mapped area.

---

### 10. Global Localization with Descriptors
**Description**: Implement ScanContext or similar for robust initial localization.

---

## Completed

- [x] Add max_points_icp setting (2000 default)
- [x] Add max_map_points setting (500K default)
- [x] Diagnostic logging system
- [x] Disable map viewer toggle for performance testing
- [x] Fix stale buffer bug on map clear/restart
- [x] Reduce max_iterations to 2 for real-time

---

## Notes

### Performance Baselines
With current settings (max_points_icp=2000, max_iterations=2):
- Target SLAM rate: 10 Hz
- Target SLAM time: <50ms per scan
- Map limit before slowdown: ~500K points

### Hardware Reference
- LiDAR: Livox Mid-360 (10 Hz, 200K pts/s)
- IMU: Livox Mid-360 internal (200 Hz)
- Motors: 2x VESC 6 Edu (CAN bus, 500 kbps)
- Compute: Windows PC (test various GPU/CPU combos)
