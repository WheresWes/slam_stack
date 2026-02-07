# Global Localization Development Progress

## Status: IN PROGRESS

## Current Approach
None yet - starting fresh.

## Baseline Performance (no hint)
- Map: house.ply (58866 points, ~12.6m x 13.3m house)
- Grid search generates 503M+ hypotheses for house map
- Completely impractical without hint
- With hint (10m radius): works but slow (~30s+)

## Test Results Log
| Date | Approach | Pass Rate | Avg Time | Best Confidence | Notes |
|------|----------|-----------|----------|-----------------|-------|
| baseline | hint-based grid search | N/A | >60s | N/A | 503M hypotheses without hint |

## Approaches Tried
(none yet)

## Approaches To Consider
- [ ] Gravity-constrained 2D projection matching (reduce to 3DOF: x, y, yaw)
- [ ] FPFH feature descriptors + RANSAC global registration
- [ ] Scan Context place recognition (partially implemented, currently disabled)
- [ ] NDT (Normal Distribution Transform) grid matching
- [ ] Correlative scan matching (lookup table / FFT acceleration)
- [ ] Height histogram signatures for coarse place recognition
- [ ] Branch-and-bound optimal search
- [ ] TEASER++ style certifiable registration
- [ ] Learned descriptors (PointNetVLAD etc.) - may be overkill for house-scale
- [ ] Combination approaches (coarse descriptor → fine ICP)

## Key Insights
- Map is a house: ~12.6m x 13.3m, 28037 points
- Gravity vector available from IMU → reduces 6DOF to 4DOF (x,y,z,yaw)
- Ground plane constraint → reduces to 3DOF (x,y,yaw)
- 3DOF search over house-scale map is very tractable
- Local scan accumulates ~4000 points, 900+ voxels in ~10s
- Target: localize in <15s without hint

## Files Modified
(track all files changed here so next iteration knows what was done)
