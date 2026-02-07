# Global Localization Without Hint - Development Task

## Objective
Develop a fast, reliable global localization system that determines the robot's pose within a pre-built 3D point cloud map WITHOUT requiring a position hint. The robot operates on a mesh surface (for testing, assume the ground plane of the map IS the mesh surface).

## Success Criteria
1. **Primary**: Localize successfully in ALL 4 test recordings with `--no-hint`
2. **Timing**: Complete localization in under 15 seconds (ideally under 10s)
3. **Confidence**: Achieve confidence >= 0.60 on all recordings
4. **Accuracy**: Position within 0.5m and heading within 10 degrees of ground truth
5. When all criteria are met, write "COMPLETE" to `ralph/COMPLETE.txt`

## Constraints
- C++ only (no Python in the hot path)
- Must integrate with existing SlamEngine (header-only, `slam_engine.hpp`)
- Must work with the `replay_localization.exe` test harness
- Can use any open-source header-only or easily-integrable C++ libraries
- The robot has a Livox Mid-360 LiDAR (360° horizontal, 59° vertical FOV) and IMU
- Pre-built map is a PLY file with XYZ + intensity points (~28K points for test map)
- Build system: CMake + MSVC (Visual Studio 2022), vcpkg for dependencies

## Physical Constraints (USE THESE!)
- **Gravity vector**: IMU provides gravity direction → roll and pitch are known → reduces 6DOF to 4DOF (x, y, z, yaw)
- **Mesh surface constraint**: Robot is ON a surface (ground plane for testing) → z is constrained → reduces to 3DOF (x, y, yaw)
- **Mid-360 FOV**: 360° horizontal scan provides rich geometric context from any position
- These constraints mean you only need to search over (x, y, yaw) - a very tractable space for a house-sized map

## Available Resources

### Test Recordings (4 sessions, recorded with live hardware)
```
C:\Users\wmuld\OneDrive\Desktop\Documents\ATLASCpp\loc_session_20260207_125823.bin (43MB)
C:\Users\wmuld\OneDrive\Desktop\Documents\ATLASCpp\loc_session_20260207_132057.bin (14MB)
C:\Users\wmuld\OneDrive\Desktop\Documents\ATLASCpp\loc_session_20260207_132123.bin (48MB)
C:\Users\wmuld\OneDrive\Desktop\Documents\ATLASCpp\loc_session_20260207_141500.bin (42MB)
```

### Pre-built Map
```
C:\Users\wmuld\OneDrive\Desktop\Documents\ATLASCpp\house.ply (920KB, 58866 points)
Map bounds: house-scale (~12.6m x 13.3m)
Note: This is the primary map. Recordings were made in this house.
```

### Test Harness
```bash
# Build and test all recordings (no hint):
bash ralph/test_localization.sh

# Test single recording:
./build/Release/replay_localization.exe \
    --input "C:\Users\wmuld\OneDrive\Desktop\Documents\ATLASCpp\loc_session_20260207_141500.bin" \
    --map "C:\Users\wmuld\OneDrive\Desktop\Documents\ATLASCpp\house.ply" \
    --no-hint --max-time 60 --json --verbose
```

### Build Command
```bash
cd slam_stack/build
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" \
    replay_localization.vcxproj -p:Configuration=Release -p:Platform=x64 -v:minimal
```

### Key Source Files (modify these)
- `slam_engine/include/slam/progressive_localizer.hpp` - Main localization algorithm (THIS IS THE PRIMARY FILE TO MODIFY)
- `slam_engine/include/slam/icp.hpp` - ICP alignment (already has gravity constraint support)
- `slam_engine/include/slam/slam_engine.hpp` - Engine integration (startProgressiveLocalization, checkProgressiveLocalization, runGlobalLocalization)
- `examples/replay_localization.cpp` - Replay tool (may need CLI flag additions)

### Key Existing Capabilities
- `ICP::align()` with gravity constraint (constrain_gravity=true → 4DOF)
- `ICP::alignMultiScale()` for coarse-to-fine refinement
- `VoxelOccupancyMap` for fast voxel-based hypothesis scoring
- `ScanContext` ring descriptor (disabled, partially implemented)
- PLY loading via `loadPrebuiltMap()`
- Full FAST-LIO SLAM engine for local map building

## Architecture Context

The localization workflow is:
1. FAST-LIO runs to build a local point cloud map from the robot's LiDAR
2. After sufficient coverage (~400+ voxels, ~45° rotation), global localization runs
3. Global localization matches local map against pre-built map to find the transform
4. The transform `T_map_to_odom` is applied: `global_pose = T_map_to_odom * local_slam_pose`
5. FAST-LIO continues running locally; the transform corrects its output to map frame

**Your job is step 3** - finding the transform efficiently without a position hint.

The function to implement/replace is `ProgressiveGlobalLocalizer::attemptGlobalLocalizationWithProgress()` in `progressive_localizer.hpp`. It receives:
- `local_map`: vector of 3D points (the local scan accumulated by FAST-LIO)
- `prebuilt_map`: vector of 3D points (the reference map)
- `current_pose`: current SLAM pose (4x4 matrix, in local frame)
- Returns: `LocalizationResult` with `transform` (4x4), `pose` (4x4), `confidence`, `status`

## Development Strategy

### DO:
- Read PROGRESS.md first to see what has been tried and what worked/failed
- Try ONE approach at a time, test it, record results in PROGRESS.md
- Use the test harness (`bash ralph/test_localization.sh`) to evaluate
- Commit working improvements to git with descriptive messages
- Think about the physics: gravity, surface constraints, scan geometry
- Consider approaches from multiple domains: robotics, computer vision, point cloud processing
- If an approach partially works, try tuning it before abandoning
- Use existing ICP infrastructure for refinement (it works well for fine alignment)
- Profile performance if timing is an issue

### DON'T:
- Don't give up on an approach after one attempt - iterate and tune
- Don't modify slam_gui.cpp (the GUI) - work only on the localization algorithm
- Don't break the existing hint-based localization (it should still work as fallback)
- Don't add heavy external dependencies that are hard to build on Windows/MSVC
- Don't change the recording format or replay tool interface (except adding new CLI flags)

## Suggested Approach Order (but feel free to innovate!)

### Phase 1: Gravity-Constrained 2D Projection Matching
The most promising quick win. Project both local scan and pre-built map to the ground plane using the gravity vector. This creates a 2D "floor plan" which is much easier to match.
- Extract ground plane height from map (cluster Z values)
- Project points to XY plane (gravity-aligned)
- Use 2D correlative scan matching or occupancy grid correlation
- Refine with 3D ICP using the 2D match as initial guess

### Phase 2: Feature-Based Global Registration
Extract geometric features (FPFH, SHOT, or simpler custom descriptors) from both point clouds, establish correspondences, and solve for the rigid transform.
- Compute surface normals
- Extract descriptors at keypoints
- RANSAC-based correspondence filtering
- Refine with ICP

### Phase 3: Smart Search Space Reduction
Instead of searching the entire map, use geometric signatures to identify candidate regions.
- Height histogram of local scan → match against sliding window on map
- Structural features (walls, corners) as landmarks
- Reduce search space to a few candidate positions, then run ICP on each

### Phase 4: Hybrid Approach
Combine the best elements of what worked:
- Fast coarse matching to get 5-10 candidate positions
- Multi-hypothesis ICP refinement (already implemented)
- Distinctiveness check (already implemented)

## Workflow Per Iteration

1. Read `ralph/PROGRESS.md` to understand current state
2. Choose what to work on next
3. Implement changes in the source files
4. Build: `cd slam_stack/build && MSBuild replay_localization.vcxproj ...`
5. Test: `bash ralph/test_localization.sh`
6. Record results in `ralph/PROGRESS.md`
7. If tests pass: `git add -A && git commit -m "description of changes"`
8. If all criteria met: write "COMPLETE" to `ralph/COMPLETE.txt`
9. If not done: continue to next approach/iteration
