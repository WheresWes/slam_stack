# Ralph Loop - Global Localization Development

## Your Task
Read `ralph/PROMPT.md` for the full task specification. Read `ralph/PROGRESS.md` for what has been tried so far. Then continue development.

## Quick Reference

### Build
```bash
cd slam_stack/build
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" \
    replay_localization.vcxproj -p:Configuration=Release -p:Platform=x64 -v:minimal
```

### Test (all recordings, no hint)
```bash
cd slam_stack
bash ralph/test_localization.sh
```

### Test (single recording, verbose)
```bash
cd slam_stack/build/Release
./replay_localization.exe \
    --input "C:/Users/wmuld/OneDrive/Desktop/Documents/ATLASCpp/loc_session_20260207_141500.bin" \
    --map "C:/Users/wmuld/OneDrive/Desktop/Documents/ATLASCpp/house.ply" \
    --no-hint --max-time 60 --json --verbose
```

### Key Files to Modify
- `slam_engine/include/slam/progressive_localizer.hpp` - **PRIMARY** localization algorithm
- `slam_engine/include/slam/icp.hpp` - ICP alignment (has gravity constraint)
- `slam_engine/include/slam/slam_engine.hpp` - Engine integration
- `examples/replay_localization.cpp` - Replay tool

### Map Info
- 28037 points, house-scale (~12.6m x 13.3m)
- PLY format with XYZ + intensity
- Ground plane approximately at z ≈ 0

### Recording Format
FUSN v2 binary: IMU (200Hz) + LiDAR scans (10Hz) + VESC odometry (~50Hz)
See `common/include/slam/localization_recording.hpp`

### Critical Rules
1. Always read PROGRESS.md first
2. Try ONE approach at a time
3. Test with the harness before and after changes
4. Record ALL results in PROGRESS.md (even failures - they're valuable)
5. Commit working improvements to git
6. Don't break existing hint-based localization
7. Don't modify slam_gui.cpp

### MSBuild Note
Use double-quote paths. The MSBuild path has spaces. Use `-p:` flags (dash prefix).

### Git
```bash
cd slam_stack
git add <specific files>
git commit -m "ralph: description of changes"
```
Do NOT push (human will review and push).
