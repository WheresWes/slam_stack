#!/usr/bin/env python3
"""
Analyze SLAM diagnostic log (.slog) files for performance issues.
"""

import struct
import sys
from pathlib import Path
from collections import defaultdict
import statistics

# Message types (must match diagnostic_logger.hpp)
class LogMsgType:
    IMU_SAMPLE = 0x01
    POINT_CLOUD = 0x02
    VESC_STATUS = 0x03
    GAMEPAD_INPUT = 0x04
    LIDAR_STATUS = 0x05

    SLAM_POSE = 0x10
    SLAM_TIMING = 0x11
    SLAM_IKF_STATE = 0x12
    SLAM_MAP_STATS = 0x13
    FUSED_POSE = 0x14
    WHEEL_ODOM = 0x15

    BUFFER_STATUS = 0x20
    THREAD_TIMING = 0x21
    MEMORY_STATUS = 0x22
    FRAME_DROP = 0x23

    STATE_CHANGE = 0x30
    COMMAND_RECV = 0x31
    COMMAND_EXEC = 0x32
    ERROR_EVENT = 0x33
    WARNING_EVENT = 0x34
    INFO_EVENT = 0x35
    MARKER = 0x36


def parse_header(data):
    """Parse log file header."""
    magic = data[0:4].decode('ascii')
    if magic != 'SLOG':
        raise ValueError(f"Invalid magic: {magic}")

    version, start_time_ns, flags = struct.unpack_from('<IQI', data, 4)
    session_id = data[16:48].decode('ascii').rstrip('\x00')

    return {
        'version': version,
        'start_time_ns': start_time_ns,
        'flags': flags,
        'session_id': session_id
    }


def parse_messages(data, offset=52):  # Header is 52 bytes (4+4+8+4+32)
    """Parse all messages from log file."""
    messages = []
    errors = 0

    while offset < len(data):
        # Message header: type (1) + timestamp (8) + payload_size (4) = 13 bytes
        if offset + 13 > len(data):
            break

        msg_type, timestamp_ns, payload_size = struct.unpack_from('<BQI', data, offset)
        offset += 13

        # Sanity check payload size
        if payload_size > 1000000:  # 1MB max
            errors += 1
            if errors > 10:
                print(f"  Too many parse errors, stopping at offset {offset}")
                break
            continue

        if offset + payload_size > len(data):
            break

        payload = data[offset:offset + payload_size]
        offset += payload_size

        messages.append({
            'type': msg_type,
            'timestamp_ns': timestamp_ns,
            'timestamp_s': timestamp_ns / 1e9,
            'payload': payload
        })

    return messages


def parse_slam_timing(payload):
    """Parse SLAM_TIMING message."""
    # LogSlamTiming: total_ms, imu_ms, undistort_ms, downsample_ms, ikf_ms, map_ms,
    #                pts_in, pts_filter, pts_ds, pts_match
    if len(payload) < 40:
        return None

    values = struct.unpack_from('<6f4I', payload)
    return {
        'total_ms': values[0],
        'imu_ms': values[1],
        'undistort_ms': values[2],
        'downsample_ms': values[3],
        'ikf_ms': values[4],
        'map_ms': values[5],
        'pts_in': values[6],
        'pts_filter': values[7],
        'pts_ds': values[8],
        'pts_match': values[9]
    }


def parse_buffer_status(payload):
    """Parse BUFFER_STATUS message."""
    if len(payload) < 16:
        return None

    values = struct.unpack_from('<4I', payload)
    return {
        'imu_buffer': values[0],
        'scan_buffer': values[1],
        'command_buffer': values[2],
        'log_buffer': values[3]
    }


def parse_slam_map_stats(payload):
    """Parse SLAM_MAP_STATS message."""
    if len(payload) < 36:
        return None

    # map_points, tree_nodes, tree_depth, extent[6], added, deleted
    values = struct.unpack_from('<3I6f2I', payload)
    return {
        'map_points': values[0],
        'tree_nodes': values[1],
        'tree_depth': values[2],
        'extent': values[3:9],
        'added': values[9],
        'deleted': values[10]
    }


def parse_imu_sample(payload):
    """Parse IMU_SAMPLE message."""
    if len(payload) < 29:
        return None

    values = struct.unpack_from('<7fB', payload)
    return {
        'acc_x': values[0], 'acc_y': values[1], 'acc_z': values[2],
        'gyr_x': values[3], 'gyr_y': values[4], 'gyr_z': values[5],
        'gravity_mag': values[6],
        'initialized': values[7]
    }


def analyze_log(filepath):
    """Analyze a .slog file for performance issues."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {filepath}")
    print(f"{'='*60}\n")

    with open(filepath, 'rb') as f:
        data = f.read()

    print(f"File size: {len(data) / 1024:.1f} KB")

    # Parse header
    header = parse_header(data)
    print(f"Session: {header['session_id']}")
    print(f"Version: {header['version']}")

    # Parse messages
    messages = parse_messages(data)
    print(f"Total messages: {len(messages)}")

    # Count message types
    type_counts = defaultdict(int)
    for msg in messages:
        type_counts[msg['type']] += 1

    print("\n--- Message Type Distribution ---")
    type_names = {
        0x01: 'IMU_SAMPLE', 0x02: 'POINT_CLOUD', 0x03: 'VESC_STATUS',
        0x10: 'SLAM_POSE', 0x11: 'SLAM_TIMING', 0x12: 'SLAM_IKF_STATE',
        0x13: 'SLAM_MAP_STATS', 0x14: 'FUSED_POSE', 0x15: 'WHEEL_ODOM',
        0x20: 'BUFFER_STATUS', 0x31: 'COMMAND_RECV', 0x35: 'INFO_EVENT'
    }
    for msg_type, count in sorted(type_counts.items()):
        name = type_names.get(msg_type, f'0x{msg_type:02X}')
        print(f"  {name}: {count}")

    # Extract timing messages
    timing_msgs = [msg for msg in messages if msg['type'] == LogMsgType.SLAM_TIMING]
    buffer_msgs = [msg for msg in messages if msg['type'] == LogMsgType.BUFFER_STATUS]
    map_msgs = [msg for msg in messages if msg['type'] == LogMsgType.SLAM_MAP_STATS]
    imu_msgs = [msg for msg in messages if msg['type'] == LogMsgType.IMU_SAMPLE]

    # Analyze SLAM timing
    if timing_msgs:
        print(f"\n--- SLAM Timing Analysis ({len(timing_msgs)} samples) ---")

        timings = [parse_slam_timing(msg['payload']) for msg in timing_msgs]
        timings = [t for t in timings if t is not None]

        if timings:
            total_times = [t['total_ms'] for t in timings]

            print(f"  Total SLAM time:")
            print(f"    Min:    {min(total_times):.1f} ms")
            print(f"    Max:    {max(total_times):.1f} ms")
            print(f"    Mean:   {statistics.mean(total_times):.1f} ms")
            print(f"    Median: {statistics.median(total_times):.1f} ms")
            if len(total_times) > 1:
                print(f"    StdDev: {statistics.stdev(total_times):.1f} ms")

            # Find slow frames
            slow_threshold = 100  # ms
            slow_frames = [(timing_msgs[i]['timestamp_s'], t['total_ms'])
                          for i, t in enumerate(timings) if t['total_ms'] > slow_threshold]

            if slow_frames:
                print(f"\n  SLOW FRAMES (>{slow_threshold}ms): {len(slow_frames)} occurrences")
                print("  Worst 10:")
                for ts, time_ms in sorted(slow_frames, key=lambda x: -x[1])[:10]:
                    print(f"    t={ts:.2f}s: {time_ms:.1f} ms")

            # Analyze timing breakdown (if available)
            ikf_times = [t['ikf_ms'] for t in timings if t['ikf_ms'] > 0]
            if ikf_times:
                print(f"\n  IKF Update time (core matching):")
                print(f"    Mean: {statistics.mean(ikf_times):.1f} ms")
                print(f"    Max:  {max(ikf_times):.1f} ms")

            # Calculate SLAM rate
            if len(timing_msgs) > 1:
                duration = (timing_msgs[-1]['timestamp_s'] - timing_msgs[0]['timestamp_s'])
                if duration > 0:
                    rate = len(timing_msgs) / duration
                    print(f"\n  Effective SLAM rate: {rate:.1f} Hz")
                    if rate < 8:
                        print(f"    WARNING: Rate below 10Hz target!")

    # Analyze buffer status
    if buffer_msgs:
        print(f"\n--- Buffer Status Analysis ({len(buffer_msgs)} samples) ---")

        buffers = [parse_buffer_status(msg['payload']) for msg in buffer_msgs]
        buffers = [b for b in buffers if b is not None]

        if buffers:
            imu_bufs = [b['imu_buffer'] for b in buffers]
            scan_bufs = [b['scan_buffer'] for b in buffers]

            print(f"  IMU buffer:")
            print(f"    Max: {max(imu_bufs)}")
            print(f"    Mean: {statistics.mean(imu_bufs):.1f}")

            print(f"  Scan buffer:")
            print(f"    Max: {max(scan_bufs)}")
            print(f"    Mean: {statistics.mean(scan_bufs):.1f}")

            # Find buffer backups
            backup_times = [(buffer_msgs[i]['timestamp_s'], b['scan_buffer'])
                           for i, b in enumerate(buffers) if b['scan_buffer'] > 2]

            if backup_times:
                print(f"\n  SCAN BUFFER BACKUP (>2): {len(backup_times)} occurrences")
                print("  First 5:")
                for ts, count in backup_times[:5]:
                    print(f"    t={ts:.2f}s: {count} scans waiting")

    # Analyze map growth
    if map_msgs:
        print(f"\n--- Map Growth Analysis ({len(map_msgs)} samples) ---")

        maps = [(msg['timestamp_s'], parse_slam_map_stats(msg['payload']))
                for msg in map_msgs]
        maps = [(ts, m) for ts, m in maps if m is not None]

        if maps:
            first_ts, first_map = maps[0]
            last_ts, last_map = maps[-1]

            print(f"  Initial map: {first_map['map_points']} points")
            print(f"  Final map: {last_map['map_points']} points")

            duration = last_ts - first_ts
            if duration > 0:
                growth_rate = (last_map['map_points'] - first_map['map_points']) / duration
                print(f"  Growth rate: {growth_rate:.0f} points/sec")

    # Analyze IMU rate
    if imu_msgs and len(imu_msgs) > 1:
        print(f"\n--- IMU Analysis ({len(imu_msgs)} samples) ---")

        duration = (imu_msgs[-1]['timestamp_s'] - imu_msgs[0]['timestamp_s'])
        if duration > 0:
            # Remember: we downsample by 5x in logging
            logged_rate = len(imu_msgs) / duration
            actual_rate = logged_rate * 5  # Approximate actual rate
            print(f"  Logged IMU rate: {logged_rate:.1f} Hz (downsampled 5x)")
            print(f"  Estimated actual rate: {actual_rate:.0f} Hz")

            if actual_rate < 180:
                print(f"    WARNING: IMU rate below expected 200 Hz!")

    # Calculate inter-SLAM timing
    if len(timing_msgs) > 1:
        print(f"\n--- Inter-SLAM Timing ---")

        intervals = []
        for i in range(1, len(timing_msgs)):
            dt = timing_msgs[i]['timestamp_s'] - timing_msgs[i-1]['timestamp_s']
            intervals.append(dt * 1000)  # Convert to ms

        print(f"  Time between SLAM updates:")
        print(f"    Min:    {min(intervals):.1f} ms")
        print(f"    Max:    {max(intervals):.1f} ms")
        print(f"    Mean:   {statistics.mean(intervals):.1f} ms")
        print(f"    Median: {statistics.median(intervals):.1f} ms")

        # Find gaps
        gap_threshold = 200  # ms
        gaps = [(timing_msgs[i]['timestamp_s'], intervals[i-1])
                for i in range(1, len(timing_msgs)) if intervals[i-1] > gap_threshold]

        if gaps:
            print(f"\n  GAPS (>{gap_threshold}ms between updates): {len(gaps)}")
            print("  Worst 10:")
            for ts, gap in sorted(gaps, key=lambda x: -x[1])[:10]:
                print(f"    t={ts:.2f}s: {gap:.0f} ms gap")

    print("\n" + "="*60)
    print("Analysis complete")
    print("="*60)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Default to latest log
        log_dir = Path(__file__).parent.parent / 'build' / 'Release' / 'logs'
        logs = sorted(log_dir.glob('*.slog'))
        if logs:
            filepath = logs[-1]
            print(f"Using latest log: {filepath.name}")
        else:
            print("Usage: python analyze_slog.py <path_to_slog>")
            sys.exit(1)
    else:
        filepath = Path(sys.argv[1])

    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    analyze_log(filepath)
