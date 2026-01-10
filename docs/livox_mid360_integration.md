# Livox Mid-360 Integration Guide

## Overview

This document describes the native C++ driver for the Livox Mid-360 LiDAR sensor, implemented without any ROS or Livox SDK dependencies. The driver communicates directly with the sensor using the Livox SDK2 UDP protocol.

## Hardware Specifications

| Parameter | Value |
|-----------|-------|
| Model | Livox Mid-360 |
| Detection Range | 40m (10% reflectivity) |
| Point Rate | ~200,000 points/sec |
| FOV | 360° × 59° |
| Range Precision | ≤2cm (1σ @ 20m) |
| Angular Precision | <0.15° |
| Interface | 100Mbps Ethernet |
| Power | 9-27V DC, ~10W typical |
| Default IP | 192.168.1.1XX (XX = last 2 digits of serial) |

## Network Configuration

### Mid-360 Default Settings
- **IP Address**: 192.168.1.1XX (where XX = last 2 digits of serial number)
- **Subnet Mask**: 255.255.255.0
- **Gateway**: 192.168.1.1

### Host PC Requirements
- **IP Address**: Must be on same subnet (e.g., 192.168.1.50)
- **Subnet Mask**: 255.255.255.0
- **Firewall**: Allow UDP ports 56000-56500

### Configuration Steps (Windows)

1. **Set Static IP on Ethernet Adapter**
   ```cmd
   # Run as Administrator
   netsh interface ip set address "Ethernet" static 192.168.1.50 255.255.255.0
   ```

2. **Add Firewall Rules**
   ```cmd
   # Run as Administrator
   netsh advfirewall firewall add rule name="Livox Mid-360" dir=in action=allow protocol=UDP localport=56000-56500
   ```

3. **Verify Connection**
   ```cmd
   ping 192.168.1.1XX   # Replace XX with last 2 digits of serial
   ```

### Configuration Persistence
- Windows remembers static IP settings per adapter
- Firewall rules are persistent
- One-time setup unless Ethernet is used for other purposes

## UDP Protocol

### Port Assignments

| Function | LiDAR Port | Host Port | Description |
|----------|------------|-----------|-------------|
| Discovery | 56000 | Any | Broadcast device query |
| Commands | 56100 | 56101 | Control commands |
| Push Messages | 56200 | 56201 | Status notifications |
| Point Cloud | 56300 | 56301 | LiDAR data stream |
| IMU Data | 56400 | 56401 | Inertial measurements |
| Logs | 56500 | 56501 | Debug logging |

### Discovery Protocol

1. Host broadcasts to 255.255.255.255:56000 (or subnet broadcast 192.168.1.255:56000)
2. Command ID: 0x0000 (Device Query)
3. LiDAR responds with device type, serial number, and IP address
4. Response is broadcast so it works even if host is on different subnet

### Point Cloud Data Format

**Packet Header (24 bytes):**
```cpp
struct LivoxPointCloudHeader {
    uint8_t  version;
    uint16_t length;
    uint16_t time_interval;  // 0.1 microseconds
    uint16_t dot_num;        // Points in packet (typically 96)
    uint16_t udp_cnt;        // Packet counter
    uint8_t  frame_cnt;      // Frame counter
    uint8_t  data_type;      // 1=32bit Cartesian, 2=16bit, 3=Spherical
    uint8_t  time_type;
    uint8_t  reserved[12];
    uint32_t crc32;
    uint64_t timestamp;      // Nanoseconds
};
```

**Point Data Type 1 (32-bit Cartesian, 14 bytes/point):**
```cpp
struct LivoxPointCartesian32 {
    int32_t x;              // millimeters
    int32_t y;              // millimeters
    int32_t z;              // millimeters
    uint8_t reflectivity;   // 0-255
    uint8_t tag;            // Point classification
};
```

### CRC Validation
- **Header CRC-16**: CCITT-FALSE polynomial (0x1021), initial 0xFFFF
- **Data CRC-32**: Standard polynomial (0x04C11DB7), initial/XOR 0xFFFFFFFF

## Driver Architecture

### Files

| File | Description |
|------|-------------|
| `slam_engine/include/slam/livox_mid360.hpp` | Main driver header (header-only) |
| `examples/test_livox_connection.cpp` | Test/demo application |
| `tools/configure_livox_network.bat` | Windows network setup script |

### Key Classes

```cpp
namespace slam {

// Device information from discovery
struct LivoxDeviceInfo {
    std::string serial_number;
    std::string ip_address;
    uint8_t device_type;
    bool is_mid360;
};

// Point cloud frame
struct LivoxPointCloudFrame {
    uint64_t timestamp_ns;
    std::vector<V3D> points;           // XYZ in meters
    std::vector<uint8_t> reflectivities;
    std::vector<uint8_t> tags;
};

// IMU frame
struct LivoxIMUFrame {
    uint64_t timestamp_ns;
    V3D gyro;    // rad/s
    V3D accel;   // m/s^2
};

// Main driver class
class LivoxMid360 {
public:
    // Discovery
    std::vector<LivoxDeviceInfo> discover(int timeout_ms = 2000,
                                          const std::string& host_ip = "");

    // Connection
    bool connect(const std::string& device_ip = "",
                 const std::string& host_ip = "192.168.1.50");

    // Streaming
    bool startStreaming();
    void stop();

    // Callbacks
    void setPointCloudCallback(PointCloudCallback cb);
    void setIMUCallback(IMUCallback cb);

    // Status
    bool isConnected() const;
    bool isStreaming() const;
    uint64_t getPointCount() const;
};

// Convenience functions
void scanLivoxDevices(bool show_guidance = true);
NetworkDiagnostic checkNetworkConfiguration();

}
```

### Usage Example

```cpp
#include "slam/livox_mid360.hpp"

int main() {
    slam::LivoxMid360 lidar;

    // Option 1: Auto-discover
    auto devices = lidar.discover();
    if (!devices.empty()) {
        lidar.connect(devices[0].ip_address);
    }

    // Option 2: Connect directly using serial number
    // Serial ending in "544" -> IP 192.168.1.144
    lidar.connect("192.168.1.144", "192.168.1.50");

    // Set callback for point cloud data
    lidar.setPointCloudCallback([](const slam::LivoxPointCloudFrame& frame) {
        std::cout << "Received " << frame.points.size() << " points\n";
        // Process points...
    });

    // Start streaming
    lidar.startStreaming();

    // ... run for desired duration ...

    lidar.stop();
    return 0;
}
```

## Command Line Tool

### Build
```bash
cd slam_stack
cmake -B build -DCMAKE_PREFIX_PATH=C:/vcpkg/installed/x64-windows
cmake --build build --config Release --target test_livox_connection
```

### Usage
```bash
# Show help
./build/Release/test_livox_connection.exe --help

# Scan for devices
./build/Release/test_livox_connection.exe --scan

# Connect using serial number (recommended)
./build/Release/test_livox_connection.exe --serial 544 --time 10

# Connect using IP directly
./build/Release/test_livox_connection.exe --device 192.168.1.144 --time 10

# Capture and save to PLY
./build/Release/test_livox_connection.exe --serial 544 --time 10 --save
```

### Options
| Option | Description |
|--------|-------------|
| `--scan` | Scan for devices and exit |
| `--serial <num>` | Last 2-3 digits of serial number |
| `--device <ip>` | Device IP address directly |
| `--host <ip>` | Host IP (default: 192.168.1.50) |
| `--time <sec>` | Capture duration (default: 5) |
| `--save` | Save captured points to PLY file |

## Troubleshooting

### Network Configuration Issues

If the driver shows "NETWORK CONFIGURATION REQUIRED":
1. Verify Ethernet adapter has IP in 192.168.1.x range
2. Run `configure_livox_network.bat` as Administrator
3. Or manually configure via Network Connections (ncpa.cpl)

### Device Not Detected

If broadcast discovery fails:
1. Verify LiDAR is powered (9-27V DC)
2. Check Ethernet cable connection
3. Add firewall rule for UDP 56000-56500
4. Use `--device` or `--serial` to bypass discovery

### No Point Cloud Data

If connected but no data received:
1. Verify firewall allows UDP 56301
2. Check that LiDAR has completed boot sequence (~10 seconds)
3. Verify no other application is using the same ports

### Low Point Rate

Expected: ~180,000-200,000 points/sec (achieved)
If receiving significantly fewer points:
1. Ensure no other application (Livox Viewer, ROS) is using the LiDAR
2. Check for packet loss (network issues, firewall blocking)
3. Verify LiDAR destination IP is configured to match host IP (use Livox Viewer to check)
4. Socket receive buffer may be too small on some systems

## Performance Notes

The driver achieves ~180k pts/sec using optimized UDP reception:
- **4MB socket receive buffer** to handle burst traffic
- **select()-based polling** for efficient waiting
- **Tight drain loops** to read all available packets without delays
- Typical performance: 180-200k points/sec depending on environment

## Known Limitations

1. **Discovery requires firewall exception**: UDP broadcast on port 56000 must be allowed
2. **Single subnet only**: Host and LiDAR must be on same subnet
3. **Initial configuration**: LiDAR must be configured (via Livox Viewer) to send data to host IP
4. **Windows only**: Current implementation tested on Windows; Linux support requires minor modifications

## References

- [Livox SDK2 GitHub](https://github.com/Livox-SDK/Livox-SDK2)
- [Livox Mid-360 Protocol Documentation](https://livox-wiki-en.readthedocs.io/en/latest/tutorials/new_product/mid360/livox_eth_protocol_mid360.html)
- [Livox Mid-360 User Manual](https://www.livoxtech.com/mid-360/downloads)
