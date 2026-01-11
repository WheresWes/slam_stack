#!/usr/bin/env python3
"""Quick PLY analysis to verify point cloud data makes sense."""

import struct
import sys
from pathlib import Path

def analyze_ply(filename):
    with open(filename, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        # Parse header
        num_points = 0
        is_binary = False
        for line in header_lines:
            if line.startswith('element vertex'):
                num_points = int(line.split()[-1])
            if 'binary' in line:
                is_binary = True

        print(f"File: {filename}")
        print(f"Points: {num_points:,}")
        print(f"Format: {'binary' if is_binary else 'ascii'}")
        print()

        if not is_binary or num_points == 0:
            return

        # Count properties from header
        properties = []
        for line in header_lines:
            if line.startswith('property float'):
                properties.append(line.split()[-1])
            elif line.startswith('property uchar'):
                properties.append('uchar:' + line.split()[-1])

        # Calculate bytes per point
        bytes_per_point = 0
        for prop in properties:
            if prop.startswith('uchar:'):
                bytes_per_point += 1
            else:
                bytes_per_point += 4  # float

        print(f"Properties: {properties}")
        print(f"Bytes per point: {bytes_per_point}")
        print()

        # Read binary point data
        points = []
        for i in range(num_points):
            data = f.read(bytes_per_point)
            if len(data) < bytes_per_point:
                break
            # First 3 floats are always x, y, z
            x, y, z = struct.unpack('<fff', data[:12])
            points.append((x, y, z))

        if not points:
            print("No points read!")
            return

        # Calculate statistics
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]

        print("=== Point Cloud Statistics ===")
        print(f"X range: {min(xs):.3f} to {max(xs):.3f} m (span: {max(xs)-min(xs):.3f} m)")
        print(f"Y range: {min(ys):.3f} to {max(ys):.3f} m (span: {max(ys)-min(ys):.3f} m)")
        print(f"Z range: {min(zs):.3f} to {max(zs):.3f} m (span: {max(zs)-min(zs):.3f} m)")
        print()

        # Calculate centroid
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        cz = sum(zs) / len(zs)
        print(f"Centroid: ({cx:.3f}, {cy:.3f}, {cz:.3f})")

        # Calculate distances from origin
        distances = [(x**2 + y**2 + z**2)**0.5 for x, y, z in points]
        print(f"Distance from origin: {min(distances):.3f} to {max(distances):.3f} m")
        print(f"Average distance: {sum(distances)/len(distances):.3f} m")
        print()

        # Check for invalid points
        invalid = sum(1 for d in distances if d < 0.1 or d > 100)
        print(f"Points < 0.1m or > 100m: {invalid} ({100*invalid/len(points):.1f}%)")

        # Sample some points
        print()
        print("=== Sample Points (first 10) ===")
        for i, (x, y, z) in enumerate(points[:10]):
            d = (x**2 + y**2 + z**2)**0.5
            print(f"  [{i}] ({x:8.3f}, {y:8.3f}, {z:8.3f}) - distance: {d:.3f}m")

        print()
        print("=== Sample Points (random 10) ===")
        import random
        random.seed(42)
        samples = random.sample(points, min(10, len(points)))
        for i, (x, y, z) in enumerate(samples):
            d = (x**2 + y**2 + z**2)**0.5
            print(f"  [{i}] ({x:8.3f}, {y:8.3f}, {z:8.3f}) - distance: {d:.3f}m")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Find most recent ply file
        ply_files = list(Path('.').glob('livox*.ply'))
        if ply_files:
            analyze_ply(str(ply_files[0]))
        else:
            print("Usage: python analyze_ply.py <filename.ply>")
    else:
        analyze_ply(sys.argv[1])
