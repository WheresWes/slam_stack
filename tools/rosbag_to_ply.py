#!/usr/bin/env python3
"""
rosbag_to_ply.py - Extract point clouds from ROS bags to PLY files

Works WITHOUT ROS installation using the 'rosbags' library.

Usage:
    pip install rosbags numpy
    python rosbag_to_ply.py <bag_file> [options]

Examples:
    python rosbag_to_ply.py data.bag
    python rosbag_to_ply.py data.bag --topic /livox/lidar --output ./ply_frames
    python rosbag_to_ply.py data.bag --merge --output merged_cloud.ply
"""

import argparse
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Run: pip install numpy")
    sys.exit(1)

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    print("ERROR: rosbags not installed. Run: pip install rosbags")
    sys.exit(1)


def parse_pointcloud2(msg):
    """
    Parse a PointCloud2 message into numpy array of XYZ points.
    Handles various point formats (XYZI, XYZRGB, etc.)
    """
    # Get field information
    fields = {f.name: f for f in msg.fields}

    # Determine point structure
    point_step = msg.point_step
    data = bytes(msg.data)
    num_points = msg.width * msg.height

    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Find XYZ field offsets and types
    x_offset = fields['x'].offset if 'x' in fields else 0
    y_offset = fields['y'].offset if 'y' in fields else 4
    z_offset = fields['z'].offset if 'z' in fields else 8

    # Extract points
    points = []
    for i in range(num_points):
        base = i * point_step
        try:
            x = struct.unpack_from('f', data, base + x_offset)[0]
            y = struct.unpack_from('f', data, base + y_offset)[0]
            z = struct.unpack_from('f', data, base + z_offset)[0]

            # Filter invalid points (NaN, inf, or zero)
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                if not (x == 0 and y == 0 and z == 0):
                    points.append([x, y, z])
        except struct.error:
            continue

    return np.array(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)


def parse_livox_custom_msg(msg):
    """
    Parse Livox custom point cloud message format.
    Livox uses a custom message type with different structure.
    """
    # Try to extract points from Livox custom format
    # This handles livox_ros_driver2 CustomMsg format
    try:
        points = []
        for point in msg.points:
            x, y, z = point.x, point.y, point.z
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                if not (x == 0 and y == 0 and z == 0):
                    points.append([x, y, z])
        return np.array(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)
    except AttributeError:
        return None


def save_ply(filename, points, binary=True):
    """Save points to PLY file format."""
    num_points = len(points)

    with open(filename, 'wb' if binary else 'w') as f:
        # Header
        header = f"""ply
format {'binary_little_endian' if binary else 'ascii'} 1.0
element vertex {num_points}
property float x
property float y
property float z
end_header
"""
        if binary:
            f.write(header.encode('ascii'))
            points.astype(np.float32).tofile(f)
        else:
            f.write(header)
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")


def list_topics(bag_path):
    """List all topics in a rosbag."""
    print(f"\nTopics in {bag_path}:")
    print("-" * 60)

    with AnyReader([Path(bag_path)]) as reader:
        for conn in reader.connections:
            count = sum(1 for _ in reader.messages(connections=[conn]))
            print(f"  {conn.topic:<40} [{conn.msgtype}] ({count} msgs)")

    print("-" * 60)


def find_pointcloud_topics(bag_path):
    """Find topics that likely contain point cloud data."""
    pointcloud_types = [
        'sensor_msgs/msg/PointCloud2',
        'sensor_msgs/PointCloud2',
        'livox_ros_driver2/msg/CustomMsg',
        'livox_ros_driver/CustomMsg',
    ]

    topics = []
    with AnyReader([Path(bag_path)]) as reader:
        for conn in reader.connections:
            if any(pc_type in conn.msgtype for pc_type in pointcloud_types):
                topics.append((conn.topic, conn.msgtype))
            # Also check for common Livox topic names
            if 'livox' in conn.topic.lower() or 'lidar' in conn.topic.lower():
                if (conn.topic, conn.msgtype) not in topics:
                    topics.append((conn.topic, conn.msgtype))

    return topics


def extract_frames(bag_path, output_dir, topic=None, max_frames=None, skip_frames=0):
    """Extract individual point cloud frames to PLY files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Auto-detect topic if not specified
    if topic is None:
        pc_topics = find_pointcloud_topics(bag_path)
        if not pc_topics:
            print("ERROR: No point cloud topics found in bag file.")
            print("Use --list to see all topics.")
            return
        topic = pc_topics[0][0]
        print(f"Auto-detected topic: {topic}")

    print(f"\nExtracting from: {bag_path}")
    print(f"Topic: {topic}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    frame_count = 0
    total_points = 0

    with AnyReader([Path(bag_path)]) as reader:
        connections = [c for c in reader.connections if c.topic == topic]

        if not connections:
            print(f"ERROR: Topic '{topic}' not found in bag file.")
            return

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            if frame_count < skip_frames:
                frame_count += 1
                continue

            if max_frames and (frame_count - skip_frames) >= max_frames:
                break

            msg = reader.deserialize(rawdata, conn.msgtype)

            # Try different parsing methods
            points = None
            if 'PointCloud2' in conn.msgtype:
                points = parse_pointcloud2(msg)
            else:
                points = parse_livox_custom_msg(msg)

            if points is None or len(points) == 0:
                print(f"  Frame {frame_count}: No valid points (skipping)")
                frame_count += 1
                continue

            # Save frame
            output_file = output_path / f"frame_{frame_count:06d}.ply"
            save_ply(output_file, points)

            total_points += len(points)
            print(f"  Frame {frame_count}: {len(points):,} points -> {output_file.name}")

            frame_count += 1

    print("-" * 60)
    print(f"Extracted {frame_count - skip_frames} frames, {total_points:,} total points")


def extract_merged(bag_path, output_file, topic=None, max_frames=None, voxel_size=None):
    """Extract and merge all point clouds into a single PLY file."""

    # Auto-detect topic if not specified
    if topic is None:
        pc_topics = find_pointcloud_topics(bag_path)
        if not pc_topics:
            print("ERROR: No point cloud topics found.")
            return
        topic = pc_topics[0][0]
        print(f"Auto-detected topic: {topic}")

    print(f"\nMerging point clouds from: {bag_path}")
    print(f"Topic: {topic}")
    print(f"Output: {output_file}")
    if voxel_size:
        print(f"Voxel downsampling: {voxel_size}m")
    print("-" * 60)

    all_points = []
    frame_count = 0

    with AnyReader([Path(bag_path)]) as reader:
        connections = [c for c in reader.connections if c.topic == topic]

        if not connections:
            print(f"ERROR: Topic '{topic}' not found.")
            return

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            if max_frames and frame_count >= max_frames:
                break

            msg = reader.deserialize(rawdata, conn.msgtype)

            points = None
            if 'PointCloud2' in conn.msgtype:
                points = parse_pointcloud2(msg)
            else:
                points = parse_livox_custom_msg(msg)

            if points is not None and len(points) > 0:
                all_points.append(points)
                print(f"  Frame {frame_count}: {len(points):,} points")

            frame_count += 1

    if not all_points:
        print("ERROR: No points extracted!")
        return

    # Merge all points
    merged = np.vstack(all_points)
    print(f"\nTotal points before filtering: {len(merged):,}")

    # Optional voxel downsampling
    if voxel_size and voxel_size > 0:
        merged = voxel_downsample(merged, voxel_size)
        print(f"After voxel downsampling ({voxel_size}m): {len(merged):,}")

    # Save merged cloud
    save_ply(output_file, merged)
    print(f"\nSaved merged cloud to: {output_file}")


def voxel_downsample(points, voxel_size):
    """Simple voxel grid downsampling."""
    if len(points) == 0:
        return points

    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # Use dictionary to keep one point per voxel
    voxel_dict = {}
    for i, idx in enumerate(voxel_indices):
        key = tuple(idx)
        if key not in voxel_dict:
            voxel_dict[key] = points[i]

    return np.array(list(voxel_dict.values()), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Extract point clouds from ROS bags to PLY files (no ROS required)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.bag --list                    # List all topics
  %(prog)s data.bag                           # Extract frames to ./ply_output
  %(prog)s data.bag --topic /livox/lidar      # Specify topic
  %(prog)s data.bag --merge -o map.ply        # Merge all frames
  %(prog)s data.bag --merge --voxel 0.05      # Merge with 5cm downsampling
        """
    )

    parser.add_argument('bag_file', help='Path to rosbag file (.bag or .db3)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all topics in the bag file')
    parser.add_argument('--topic', '-t', type=str, default=None,
                        help='Topic name to extract (auto-detects if not specified)')
    parser.add_argument('--output', '-o', type=str, default='./ply_output',
                        help='Output directory (frames) or file (merged)')
    parser.add_argument('--merge', '-m', action='store_true',
                        help='Merge all frames into single PLY file')
    parser.add_argument('--max-frames', '-n', type=int, default=None,
                        help='Maximum number of frames to extract')
    parser.add_argument('--skip', '-s', type=int, default=0,
                        help='Number of initial frames to skip')
    parser.add_argument('--voxel', '-v', type=float, default=None,
                        help='Voxel size for downsampling (meters)')

    args = parser.parse_args()

    bag_path = Path(args.bag_file)
    if not bag_path.exists():
        print(f"ERROR: File not found: {bag_path}")
        sys.exit(1)

    if args.list:
        list_topics(bag_path)

        pc_topics = find_pointcloud_topics(bag_path)
        if pc_topics:
            print("\nDetected point cloud topics:")
            for topic, msgtype in pc_topics:
                print(f"  {topic} [{msgtype}]")
        return

    if args.merge:
        output_file = args.output if args.output.endswith('.ply') else args.output + '.ply'
        extract_merged(bag_path, output_file, args.topic, args.max_frames, args.voxel)
    else:
        extract_frames(bag_path, args.output, args.topic, args.max_frames, args.skip)


if __name__ == '__main__':
    main()
