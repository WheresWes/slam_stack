/**
 * @file livox_mid360.hpp
 * @brief Native Livox Mid-360 driver with auto-discovery (no ROS/SDK dependency)
 *
 * Implements the Livox SDK2 protocol for Mid-360:
 * - UDP broadcast discovery on port 56000
 * - Point cloud streaming on port 56300
 * - IMU data streaming on port 56400
 *
 * Reference: https://livox-wiki-en.readthedocs.io/en/latest/tutorials/new_product/mid360/
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <iostream>
#include <iomanip>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #define WIN32_LEAN_AND_MEAN
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef int socklen_t;
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #define SOCKET int
    #define INVALID_SOCKET -1
    #define SOCKET_ERROR -1
    #define closesocket close
#endif

#include "slam/types.hpp"

namespace slam {

//=============================================================================
// Protocol Constants
//=============================================================================

namespace livox {

// UDP Ports
constexpr uint16_t PORT_DISCOVERY = 56000;
constexpr uint16_t PORT_COMMAND   = 56100;
constexpr uint16_t PORT_PUSH      = 56200;
constexpr uint16_t PORT_POINTCLOUD = 56300;
constexpr uint16_t PORT_IMU       = 56400;
constexpr uint16_t PORT_LOG       = 56500;

// Host ports (where we receive data)
constexpr uint16_t HOST_PORT_COMMAND    = 56101;
constexpr uint16_t HOST_PORT_PUSH       = 56201;
constexpr uint16_t HOST_PORT_POINTCLOUD = 56301;
constexpr uint16_t HOST_PORT_IMU        = 56401;

// Command IDs (8-bit command ID with sub-command in data)
// The cmd_id field is 8-bit. For lidar commands, cmd_id=0x01 and sub-command is in data
constexpr uint8_t CMD_DEVICE_QUERY      = 0x00;  // General command
constexpr uint8_t CMD_LIDAR             = 0x01;  // Lidar command set
// Sub-commands for lidar (first byte of data payload)
constexpr uint8_t SUBCMD_PARAM_CONFIG   = 0x00;  // Set parameter
constexpr uint8_t SUBCMD_PARAM_QUERY    = 0x01;  // Query parameter

// Parameter keys
constexpr uint16_t PARAM_PCL_DATA_TYPE  = 0x0000;  // Point cloud data type
constexpr uint16_t PARAM_PATTERN_MODE   = 0x0001;  // Scan pattern mode
constexpr uint16_t PARAM_POINT_SEND_EN  = 0x0003;  // 0x00=Disable, 0x01=Enable point cloud sending
constexpr uint16_t PARAM_POINTCLOUD_TYPE = 0x0012; // Data format type (alias)
constexpr uint16_t PARAM_WORK_MODE      = 0x001A;  // 0x00=PowerOn, 0x01=Standby, 0x02=Normal, 0x03=Upgrade
constexpr uint16_t PARAM_IMU_ENABLE     = 0x001C;  // 0x00=Disable, 0x01=Enable

// Protocol constants
constexpr uint8_t FRAME_START = 0xAA;
constexpr uint8_t CMD_TYPE_REQ = 0x00;
constexpr uint8_t CMD_TYPE_ACK = 0x01;

// Device types
constexpr uint8_t DEVICE_TYPE_MID360 = 9;

// Points per packet for each data type
constexpr size_t POINTS_PER_PACKET = 96;

} // namespace livox

//=============================================================================
// Data Structures (packed for network protocol)
//=============================================================================

#pragma pack(push, 1)

// Command frame header (SDK2 format)
// Based on Pack() analysis: 11-byte header, then data, then CRC32
struct LivoxCmdHeader {
    uint8_t  sof;           // Start of frame: 0xAA (offset 0)
    uint8_t  version;       // Protocol version (offset 1)
    uint16_t length;        // Frame length (offset 2)
    uint16_t seq_num;       // Sequence number (offset 4) - 16-bit!
    uint8_t  cmd_id;        // Command ID (offset 6) - 8-bit!
    uint8_t  cmd_type;      // 0x00=REQ, 0x01=ACK (offset 7)
    uint8_t  sender_type;   // Sender type (offset 8)
    uint16_t crc16;         // CRC-16 of bytes 0-8 (offset 9)
    // Total 11 bytes - data follows, then CRC32 at end
};

// Full frame structure with data and CRC32 will be:
// [Header 11 bytes] [Data N bytes] [CRC32 4 bytes]
// Total length = 11 + N + 4

// Discovery response data
struct LivoxDiscoveryAck {
    uint8_t  ret_code;
    uint8_t  dev_type;
    char     serial_number[16];
    uint8_t  lidar_ip[4];
    uint16_t cmd_port;
};

// Point cloud data header (24 bytes)
struct LivoxPointCloudHeader {
    uint8_t  version;
    uint16_t length;
    uint16_t time_interval;  // 0.1 microseconds
    uint16_t dot_num;        // Points in this packet
    uint16_t udp_cnt;        // UDP packet counter
    uint8_t  frame_cnt;      // Frame counter
    uint8_t  data_type;      // 0=IMU, 1=32bit Cartesian, 2=16bit Cartesian, 3=Spherical
    uint8_t  time_type;      // Timestamp type
    uint8_t  reserved[12];
    uint32_t crc32;          // CRC-32
    uint64_t timestamp;      // Nanoseconds
};

// Point data type 1: 32-bit Cartesian (14 bytes per point)
struct LivoxPointCartesian32 {
    int32_t x;              // mm
    int32_t y;              // mm
    int32_t z;              // mm
    uint8_t reflectivity;
    uint8_t tag;
};

// Point data type 2: 16-bit Cartesian (8 bytes per point)
struct LivoxPointCartesian16 {
    int16_t x;              // 10mm resolution
    int16_t y;
    int16_t z;
    uint8_t reflectivity;
    uint8_t tag;
};

// IMU data structure
struct LivoxIMUData {
    float gyro_x;           // rad/s
    float gyro_y;
    float gyro_z;
    float acc_x;            // g
    float acc_y;
    float acc_z;
};

#pragma pack(pop)

//=============================================================================
// Discovered Device Info
//=============================================================================

struct LivoxDeviceInfo {
    std::string serial_number;
    std::string ip_address;
    uint8_t device_type;
    uint16_t cmd_port;
    bool is_mid360;

    std::string getTypeName() const {
        switch (device_type) {
            case 9: return "Mid-360";
            case 8: return "HAP";
            case 7: return "Avia";
            default: return "Unknown(" + std::to_string(device_type) + ")";
        }
    }
};

//=============================================================================
// Point Cloud Frame
//=============================================================================

struct LivoxPointCloudFrame {
    uint64_t timestamp_ns;
    std::vector<V3D> points;
    std::vector<uint8_t> reflectivities;
    std::vector<uint8_t> tags;
    std::vector<float> time_offsets_us;  // Per-point time offset from timestamp_ns (microseconds)
};

struct LivoxIMUFrame {
    uint64_t timestamp_ns;
    V3D gyro;    // rad/s
    V3D accel;   // g-units (SLAM engine scales internally)
};

//=============================================================================
// CRC Calculation (Livox-specific initial values!)
// CRC16: CCITT polynomial 0x1021, init = 0x4c49
// CRC32: Standard polynomial, init = 0x564f580a
//=============================================================================

class LivoxCRC {
public:
    static uint16_t crc16(const uint8_t* data, size_t len) {
        static const uint16_t table[256] = {
            0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
            0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
            0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6,
            0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C, 0xF3FF, 0xE3DE,
            0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485,
            0xA56A, 0xB54B, 0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D,
            0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4,
            0xB75B, 0xA77A, 0x9719, 0x8738, 0xF7DF, 0xE7FE, 0xD79D, 0xC7BC,
            0x48C4, 0x58E5, 0x6886, 0x78A7, 0x0840, 0x1861, 0x2802, 0x3823,
            0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969, 0xA90A, 0xB92B,
            0x5AF5, 0x4AD4, 0x7AB7, 0x6A96, 0x1A71, 0x0A50, 0x3A33, 0x2A12,
            0xDBFD, 0xCBDC, 0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A,
            0x6CA6, 0x7C87, 0x4CE4, 0x5CC5, 0x2C22, 0x3C03, 0x0C60, 0x1C41,
            0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD, 0xAD2A, 0xBD0B, 0x8D68, 0x9D49,
            0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70,
            0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A, 0x9F59, 0x8F78,
            0x9188, 0x81A9, 0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F,
            0x1080, 0x00A1, 0x30C2, 0x20E3, 0x5004, 0x4025, 0x7046, 0x6067,
            0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C, 0xE37F, 0xF35E,
            0x02B1, 0x1290, 0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256,
            0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D,
            0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
            0xA7DB, 0xB7FA, 0x8799, 0x97B8, 0xE75F, 0xF77E, 0xC71D, 0xD73C,
            0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634,
            0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9, 0xB98A, 0xA9AB,
            0x5844, 0x4865, 0x7806, 0x6827, 0x18C0, 0x08E1, 0x3882, 0x28A3,
            0xCB7D, 0xDB5C, 0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A,
            0x4A75, 0x5A54, 0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92,
            0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B, 0x9DE8, 0x8DC9,
            0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0, 0x0CC1,
            0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8,
            0x6E17, 0x7E36, 0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0
        };

        // Livox uses init value 0x4c49, not 0xFFFF!
        uint16_t crc = 0x4c49;
        for (size_t i = 0; i < len; i++) {
            crc = (crc << 8) ^ table[((crc >> 8) ^ data[i]) & 0xFF];
        }
        return crc;
    }

    static uint32_t crc32(const uint8_t* data, size_t len) {
        static const uint32_t table[256] = {
            0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3,
            0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988, 0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91,
            0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
            0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5,
            0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172, 0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B,
            0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
            0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F,
            0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924, 0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D,
            0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
            0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01,
            0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E, 0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF70FC457,
            0x67B0D9C6, 0x10B7E950, 0x89BE98EA, 0xFEBD8E7C, 0x609F9ADF, 0x17986C49, 0x80913AF3, 0xF6961565,
            0x4669BE79, 0x316E8EEF, 0xA867DF55, 0xDF60EFC3, 0x41047A60, 0x3603AAF6, 0xAF0A1B4C, 0xD80D2BDA,
            0x48B2364B, 0x3FB506DD, 0xA6BC5767, 0xD1BB67F1, 0x4FDDD252, 0x38FAE2C4, 0xA1D3937E, 0xD6D46DE8,
            0x5BD4B01D, 0x2CD3808B, 0xB5EA3931, 0xC2ED09A7, 0x5C898804, 0x2B8EBC92, 0xB2870228, 0xC58032BE,
            0x55390D2F, 0x223E3CB9, 0xBB372703, 0xCC301795, 0x526B8236, 0x256CB2A0, 0xBC6543A1, 0xCB6280D9,
            0x9BDC06A7, 0xECDBB631, 0x75D2C68B, 0x02D5F61D, 0x9CB8B5BE, 0xEB8FA428, 0x72860F92, 0x05810604,
            0x953E1B95, 0xE2395B03, 0x7B3029B9, 0x0C37192F, 0x929B098C, 0xE59C391A, 0x7C9528A0, 0x0B925836,
            0x86906DC3, 0xF1970D55, 0x68880CEF, 0x1F8F3C79, 0x81C395DA, 0xF6C4A54C, 0x6FCDB4F6, 0x18CA8460,
            0x88759FF1, 0xFF72AF67, 0x667799DD, 0x116EA94B, 0x8FE23CE8, 0xF8E50C7E, 0x61EC1FC4, 0x16EB0B52,
            0xE3D24BA7, 0x94D57B31, 0x0DCC2A8B, 0x7ACB1A1D, 0xE4A69FBE, 0x93A1AF28, 0x0AA89E92, 0x7DAF0E04,
            0xED101395, 0x9A172303, 0x030E72B9, 0x7409422F, 0xEA65D78C, 0x9D62E71A, 0x046B96A0, 0x736CEE36,
            0xFE6CBCE3, 0x896C8C75, 0x106DDDCF, 0x6760ED59, 0xF904F8FA, 0x8E03C86C, 0x170A99D6, 0x600D0940,
            0xF0B216D1, 0x87B52647, 0x1EBCE7FD, 0x69BBFE6B, 0xF7D7CBC8, 0x80D0FB5E, 0x19D9A8E4, 0x6EDED672,
            0xAE64C286, 0xD9633210, 0x406263AA, 0x3765533C, 0xA901C69F, 0xDE06F609, 0x470FA7B3, 0x30089725,
            0xA0B78AB4, 0xD7B0BA22, 0x4EB9E998, 0x39BED90E, 0xA7D24CAD, 0xD0D55C3B, 0x47DC0D81, 0x30DB3D17,
            0xBDDB78E2, 0xCADC4874, 0x53D519CE, 0x24D22958, 0xBEB6BCFB, 0xC9B18C6D, 0x50B8DDD7, 0x27BF0D41,
            0xB70010D0, 0xC0072046, 0x590971FC, 0x2E0E416A, 0xB064D4C9, 0xC763E45F, 0x5E6AB5E5, 0x296B8573,
            0x191CE2C6, 0x6E1BD250, 0xF712E3EA, 0x8015D37C, 0x1E7146DF, 0x697E7649, 0xF0772FF3, 0x87703565,
            0x17CF28F4, 0x60C81862, 0xF9C149D8, 0x8EC6794E, 0x10A2ECED, 0x67A5DC7B, 0xFEACD6C1, 0x89ABAD57,
            0x04A2D8A2, 0x73A5E834, 0xEAACB98E, 0x9DAB8918, 0x03CF1CBB, 0x74C82C2D, 0xEDC17D97, 0x9AC64D01,
            0x0A795090, 0x7D7E6006, 0xE47731BC, 0x9370012A, 0x0D149489, 0x7A13A41F, 0xE31AF5A5, 0x94130533
        };

        // Livox uses init value 0x564f580a, not 0xFFFFFFFF!
        uint32_t crc = 0x564f580a;
        for (size_t i = 0; i < len; i++) {
            crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xFF];
        }
        // Note: Livox may not use final XOR, but standard CRC32 does
        return crc ^ 0xFFFFFFFF;
    }
};

//=============================================================================
// Mid-360 Driver Class
//=============================================================================

class LivoxMid360 {
public:
    using PointCloudCallback = std::function<void(const LivoxPointCloudFrame&)>;
    using IMUCallback = std::function<void(const LivoxIMUFrame&)>;
    using DeviceFoundCallback = std::function<void(const LivoxDeviceInfo&)>;

    LivoxMid360() {
        initWinsock();
    }

    ~LivoxMid360() {
        stop();
        cleanupWinsock();
    }

    //=========================================================================
    // Discovery - Scan for connected LiDARs
    //=========================================================================

    /**
     * Scan for Livox devices on the network using broadcast discovery.
     * Falls back to direct IP probing if broadcast fails.
     * @param timeout_ms How long to wait for responses (default 2 seconds)
     * @param host_ip Local IP address to bind to (empty = auto-detect)
     * @return Vector of discovered devices
     */
    std::vector<LivoxDeviceInfo> discover(int timeout_ms = 2000,
                                          const std::string& host_ip = "") {
        std::vector<LivoxDeviceInfo> devices;

        // Create UDP socket
        SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock == INVALID_SOCKET) {
            std::cerr << "Failed to create discovery socket\n";
            return devices;
        }

        // Enable broadcast
        int broadcast_enable = 1;
        setsockopt(sock, SOL_SOCKET, SO_BROADCAST,
                   (const char*)&broadcast_enable, sizeof(broadcast_enable));

        // Short timeout for individual probes
        #ifdef _WIN32
        DWORD timeout = 200;  // 200ms per probe
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
        #else
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 200000;  // 200ms
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        #endif

        // Bind to local address
        sockaddr_in local_addr = {};
        local_addr.sin_family = AF_INET;
        local_addr.sin_port = htons(0);  // Any port
        if (host_ip.empty()) {
            local_addr.sin_addr.s_addr = INADDR_ANY;
        } else {
            inet_pton(AF_INET, host_ip.c_str(), &local_addr.sin_addr);
        }

        if (bind(sock, (sockaddr*)&local_addr, sizeof(local_addr)) == SOCKET_ERROR) {
            std::cerr << "Failed to bind discovery socket\n";
            closesocket(sock);
            return devices;
        }

        // Build discovery request
        std::vector<uint8_t> request = buildDiscoveryRequest();

        std::cout << "Scanning for Livox devices...\n";

        // PHASE 1: Broadcast discovery
        sockaddr_in broadcast_addr = {};
        broadcast_addr.sin_family = AF_INET;
        broadcast_addr.sin_port = htons(livox::PORT_DISCOVERY);
        broadcast_addr.sin_addr.s_addr = INADDR_BROADCAST;

        sendto(sock, (const char*)request.data(), (int)request.size(), 0,
               (sockaddr*)&broadcast_addr, sizeof(broadcast_addr));

        // Also try direct subnet broadcast (192.168.1.255)
        inet_pton(AF_INET, "192.168.1.255", &broadcast_addr.sin_addr);
        sendto(sock, (const char*)request.data(), (int)request.size(), 0,
               (sockaddr*)&broadcast_addr, sizeof(broadcast_addr));

        // Listen for broadcast responses
        uint8_t buffer[1024];
        sockaddr_in sender_addr;
        socklen_t sender_len = sizeof(sender_addr);

        auto start = std::chrono::steady_clock::now();
        while (true) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_ms / 2) break;  // Use half time for broadcast

            int recv_len = recvfrom(sock, (char*)buffer, sizeof(buffer), 0,
                                    (sockaddr*)&sender_addr, &sender_len);

            if (recv_len > 0) {
                LivoxDeviceInfo info;
                if (parseDiscoveryResponse(buffer, recv_len, info)) {
                    char ip_str[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, &sender_addr.sin_addr, ip_str, sizeof(ip_str));
                    info.ip_address = ip_str;

                    bool found = false;
                    for (const auto& d : devices) {
                        if (d.serial_number == info.serial_number) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        devices.push_back(info);
                        std::cout << "  Found: " << info.getTypeName()
                                  << " [" << info.serial_number << "] at "
                                  << info.ip_address << "\n";
                    }
                }
            }
        }

        // PHASE 2: Direct IP probing if broadcast found nothing
        // Mid-360 IPs are typically 192.168.1.1XX where XX = last 2 digits of serial
        if (devices.empty()) {
            std::cout << "  Broadcast failed, trying direct IP probe...\n";

            // Common Mid-360 IP range: 192.168.1.100 to 192.168.1.199
            std::vector<std::string> probe_ips;
            for (int i = 100; i <= 199; i++) {
                probe_ips.push_back("192.168.1." + std::to_string(i));
            }

            // First, send "reset to standby" commands to wake up devices stuck in streaming mode
            std::vector<uint8_t> standby_cmd = buildStandbyCommand();

            for (const auto& ip : probe_ips) {
                sockaddr_in dev_addr = {};
                dev_addr.sin_family = AF_INET;
                dev_addr.sin_port = htons(livox::PORT_COMMAND);
                inet_pton(AF_INET, ip.c_str(), &dev_addr.sin_addr);

                // Send standby command (might wake up device from streaming mode)
                sendto(sock, (const char*)standby_cmd.data(), (int)standby_cmd.size(), 0,
                       (sockaddr*)&dev_addr, sizeof(dev_addr));
            }

            // Wait a moment for devices to process standby command
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Now probe each IP with discovery request
            for (const auto& ip : probe_ips) {
                sockaddr_in dev_addr = {};
                dev_addr.sin_family = AF_INET;
                dev_addr.sin_port = htons(livox::PORT_DISCOVERY);
                inet_pton(AF_INET, ip.c_str(), &dev_addr.sin_addr);

                sendto(sock, (const char*)request.data(), (int)request.size(), 0,
                       (sockaddr*)&dev_addr, sizeof(dev_addr));
            }

            // Listen for responses with short timeout
            start = std::chrono::steady_clock::now();
            while (true) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start).count();
                if (elapsed >= 500) break;  // 500ms for probing phase

                int recv_len = recvfrom(sock, (char*)buffer, sizeof(buffer), 0,
                                        (sockaddr*)&sender_addr, &sender_len);

                if (recv_len > 0) {
                    LivoxDeviceInfo info;
                    if (parseDiscoveryResponse(buffer, recv_len, info)) {
                        char ip_str[INET_ADDRSTRLEN];
                        inet_ntop(AF_INET, &sender_addr.sin_addr, ip_str, sizeof(ip_str));
                        info.ip_address = ip_str;

                        bool found = false;
                        for (const auto& d : devices) {
                            if (d.serial_number == info.serial_number) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            devices.push_back(info);
                            std::cout << "  Found (via probe): " << info.getTypeName()
                                      << " [" << info.serial_number << "] at "
                                      << info.ip_address << "\n";
                        }
                    }
                }
            }
        }

        closesocket(sock);
        return devices;
    }

    /**
     * Build a command to set device to standby mode (for waking up stuck devices)
     */
    std::vector<uint8_t> buildStandbyCommand() {
        size_t data_len = 1 + 1 + 2 + 2 + 1;  // subcmd + key_num + key + len + value
        size_t frame_size = 11 + data_len + 4;
        std::vector<uint8_t> frame(frame_size, 0);

        frame[0] = livox::FRAME_START;
        frame[1] = 0;
        *reinterpret_cast<uint16_t*>(&frame[2]) = static_cast<uint16_t>(frame_size);
        *reinterpret_cast<uint16_t*>(&frame[4]) = cmd_seq_++;
        frame[6] = livox::CMD_LIDAR;
        frame[7] = livox::CMD_TYPE_REQ;
        frame[8] = 0;
        *reinterpret_cast<uint16_t*>(&frame[9]) = LivoxCRC::crc16(frame.data(), 9);

        uint8_t* data = frame.data() + 11;
        data[0] = livox::SUBCMD_PARAM_CONFIG;
        data[1] = 1;
        *reinterpret_cast<uint16_t*>(data + 2) = livox::PARAM_WORK_MODE;
        *reinterpret_cast<uint16_t*>(data + 4) = 1;
        data[6] = 0x01;  // Standby mode
        *reinterpret_cast<uint32_t*>(&frame[frame_size - 4]) =
            LivoxCRC::crc32(data, data_len);

        return frame;
    }

    //=========================================================================
    // Connection
    //=========================================================================

    /**
     * Connect to a specific device by IP or auto-connect to first discovered device.
     * @param device_ip IP address (empty = auto-discover and connect to first)
     * @param host_ip Local IP address to use
     * @return true if connection successful
     */
    bool connect(const std::string& device_ip = "",
                 const std::string& host_ip = "192.168.1.50") {

        std::string target_ip = device_ip;

        // Auto-discover if no IP specified
        if (target_ip.empty()) {
            auto devices = discover(3000, host_ip);
            if (devices.empty()) {
                std::cerr << "No Livox devices found\n";
                return false;
            }
            target_ip = devices[0].ip_address;
            device_info_ = devices[0];
            std::cout << "Auto-connecting to: " << device_info_.getTypeName()
                      << " at " << target_ip << "\n";
        }

        device_ip_ = target_ip;
        host_ip_ = host_ip;

        // Create data receiving sockets
        if (!createDataSockets()) {
            return false;
        }

        // Configure the device
        if (!configureDevice()) {
            std::cerr << "Failed to configure device\n";
            return false;
        }

        connected_ = true;
        return true;
    }

    /**
     * Start streaming point cloud data.
     */
    bool startStreaming() {
        if (!connected_) {
            std::cerr << "Not connected\n";
            return false;
        }

        streaming_ = true;

        // Start receive thread
        receive_thread_ = std::thread(&LivoxMid360::receiveLoop, this);

        // Send configuration commands
        std::cout << "Enabling point cloud streaming...\n";

        // 1. Enable point cloud sending (key 0x0003 = 1)
        sendParamCommand(livox::PARAM_POINT_SEND_EN, 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // 2. Set work mode to Normal (key 0x001A = 0x02)
        sendWorkModeCommand(0x02);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // 3. Enable IMU data (key 0x001C = 1)
        sendParamCommand(livox::PARAM_IMU_ENABLE, 1);

        return true;
    }

    /**
     * Stop streaming.
     */
    void stop() {
        if (!streaming_ && !connected_) return;

        std::cout << "Stopping LiDAR...\n";

        // Gracefully stop the LiDAR before closing sockets
        if (cmd_socket_ != INVALID_SOCKET && connected_) {
            // Disable point cloud sending first
            sendParamCommandQuiet(livox::PARAM_POINT_SEND_EN, 0);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));

            // Set work mode to Standby (0x01) so it responds to discovery next time
            sendParamCommandQuiet(livox::PARAM_WORK_MODE, 0x01);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        streaming_ = false;
        connected_ = false;

        if (receive_thread_.joinable()) {
            receive_thread_.join();
        }

        if (pointcloud_socket_ != INVALID_SOCKET) {
            closesocket(pointcloud_socket_);
            pointcloud_socket_ = INVALID_SOCKET;
        }
        if (imu_socket_ != INVALID_SOCKET) {
            closesocket(imu_socket_);
            imu_socket_ = INVALID_SOCKET;
        }
        if (cmd_socket_ != INVALID_SOCKET) {
            closesocket(cmd_socket_);
            cmd_socket_ = INVALID_SOCKET;
        }

        std::cout << "LiDAR stopped cleanly.\n";
    }

    //=========================================================================
    // Callbacks
    //=========================================================================

    void setPointCloudCallback(PointCloudCallback cb) {
        pointcloud_callback_ = std::move(cb);
    }

    void setIMUCallback(IMUCallback cb) {
        imu_callback_ = std::move(cb);
    }

    //=========================================================================
    // State
    //=========================================================================

    bool isConnected() const { return connected_; }
    bool isStreaming() const { return streaming_; }

    const LivoxDeviceInfo& getDeviceInfo() const { return device_info_; }

    // Statistics
    uint64_t getPointCount() const { return total_points_; }
    uint64_t getFrameCount() const { return frame_count_; }

private:
    // Network
    SOCKET cmd_socket_ = INVALID_SOCKET;
    SOCKET pointcloud_socket_ = INVALID_SOCKET;
    SOCKET imu_socket_ = INVALID_SOCKET;

    std::string device_ip_;
    std::string host_ip_;
    LivoxDeviceInfo device_info_;

    // State
    std::atomic<bool> connected_{false};
    std::atomic<bool> streaming_{false};

    // Threading
    std::thread receive_thread_;
    std::mutex callback_mutex_;

    // Callbacks
    PointCloudCallback pointcloud_callback_;
    IMUCallback imu_callback_;

    // Stats
    std::atomic<uint64_t> total_points_{0};
    std::atomic<uint64_t> frame_count_{0};

    // Sequence number for commands
    uint32_t cmd_seq_ = 0;

    //=========================================================================
    // Platform-specific initialization
    //=========================================================================

    void initWinsock() {
        #ifdef _WIN32
        WSADATA wsa_data;
        WSAStartup(MAKEWORD(2, 2), &wsa_data);
        #endif
    }

    void cleanupWinsock() {
        #ifdef _WIN32
        WSACleanup();
        #endif
    }

    //=========================================================================
    // Protocol Implementation
    //=========================================================================

    std::vector<uint8_t> buildDiscoveryRequest() {
        // SDK2 format: 11-byte header + CRC32 (4 bytes) = 15 bytes for empty command
        size_t frame_size = 11 + 4;  // header + crc32, no data
        std::vector<uint8_t> frame(frame_size, 0);

        // Header (11 bytes)
        frame[0] = livox::FRAME_START;
        frame[1] = 0;  // version
        *reinterpret_cast<uint16_t*>(&frame[2]) = static_cast<uint16_t>(frame_size);
        *reinterpret_cast<uint16_t*>(&frame[4]) = cmd_seq_++;  // seq_num (16-bit)
        frame[6] = livox::CMD_DEVICE_QUERY;  // cmd_id (8-bit)
        frame[7] = livox::CMD_TYPE_REQ;      // cmd_type
        frame[8] = 0;                         // sender_type

        // CRC16 over bytes 0-8 (9 bytes)
        *reinterpret_cast<uint16_t*>(&frame[9]) = LivoxCRC::crc16(frame.data(), 9);

        // CRC32 for empty data (at offset 11)
        *reinterpret_cast<uint32_t*>(&frame[11]) = 0;

        return frame;
    }

    bool parseDiscoveryResponse(const uint8_t* data, size_t len, LivoxDeviceInfo& info) {
        if (len < sizeof(LivoxCmdHeader) + sizeof(LivoxDiscoveryAck)) {
            return false;
        }

        const LivoxCmdHeader* hdr = reinterpret_cast<const LivoxCmdHeader*>(data);

        // Verify it's an ACK for device query
        if (hdr->sof != livox::FRAME_START) return false;
        if (hdr->cmd_id != livox::CMD_DEVICE_QUERY) return false;
        if (hdr->cmd_type != livox::CMD_TYPE_ACK) return false;

        const LivoxDiscoveryAck* ack = reinterpret_cast<const LivoxDiscoveryAck*>(
            data + sizeof(LivoxCmdHeader));

        info.device_type = ack->dev_type;
        info.serial_number = std::string(ack->serial_number, 16);
        // Trim trailing nulls/spaces
        while (!info.serial_number.empty() &&
               (info.serial_number.back() == '\0' || info.serial_number.back() == ' ')) {
            info.serial_number.pop_back();
        }
        info.cmd_port = ack->cmd_port;
        info.is_mid360 = (ack->dev_type == livox::DEVICE_TYPE_MID360);

        // IP is set from sender address by caller
        return true;
    }

    bool createDataSockets() {
        // Point cloud socket
        pointcloud_socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (pointcloud_socket_ == INVALID_SOCKET) {
            std::cerr << "Failed to create point cloud socket\n";
            return false;
        }

        sockaddr_in pc_addr = {};
        pc_addr.sin_family = AF_INET;
        pc_addr.sin_port = htons(livox::HOST_PORT_POINTCLOUD);
        inet_pton(AF_INET, host_ip_.c_str(), &pc_addr.sin_addr);

        if (bind(pointcloud_socket_, (sockaddr*)&pc_addr, sizeof(pc_addr)) == SOCKET_ERROR) {
            std::cerr << "Failed to bind point cloud socket to " << host_ip_
                      << ":" << livox::HOST_PORT_POINTCLOUD << "\n";
            return false;
        }

        // Set non-blocking for receive loop
        #ifdef _WIN32
        u_long mode = 1;
        ioctlsocket(pointcloud_socket_, FIONBIO, &mode);
        #endif

        // IMU socket
        imu_socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (imu_socket_ == INVALID_SOCKET) {
            std::cerr << "Failed to create IMU socket\n";
            return false;
        }

        sockaddr_in imu_addr = {};
        imu_addr.sin_family = AF_INET;
        imu_addr.sin_port = htons(livox::HOST_PORT_IMU);
        inet_pton(AF_INET, host_ip_.c_str(), &imu_addr.sin_addr);

        if (bind(imu_socket_, (sockaddr*)&imu_addr, sizeof(imu_addr)) == SOCKET_ERROR) {
            std::cerr << "Failed to bind IMU socket\n";
            return false;
        }

        #ifdef _WIN32
        ioctlsocket(imu_socket_, FIONBIO, &mode);
        #endif

        // Command socket
        cmd_socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (cmd_socket_ == INVALID_SOCKET) {
            std::cerr << "Failed to create command socket\n";
            return false;
        }

        sockaddr_in cmd_addr = {};
        cmd_addr.sin_family = AF_INET;
        cmd_addr.sin_port = htons(livox::HOST_PORT_COMMAND);
        inet_pton(AF_INET, host_ip_.c_str(), &cmd_addr.sin_addr);

        if (bind(cmd_socket_, (sockaddr*)&cmd_addr, sizeof(cmd_addr)) == SOCKET_ERROR) {
            std::cerr << "Failed to bind command socket\n";
            return false;
        }

        std::cout << "Sockets bound to " << host_ip_ << "\n";
        return true;
    }

    bool configureDevice() {
        // Configure the Mid-360 to send data to our host
        // Key 0x0006: Point cloud destination (IP + ports)
        // Key 0x001A: Work mode (0x02 = Normal)

        std::cout << "Configuring LiDAR data destination...\n";

        // Step 1: Configure point cloud destination
        if (!sendHostConfig(0x0006, livox::HOST_PORT_POINTCLOUD, livox::PORT_POINTCLOUD)) {
            std::cerr << "Warning: Failed to configure point cloud destination\n";
        }

        // Step 2: Configure IMU destination
        if (!sendHostConfig(0x0007, livox::HOST_PORT_IMU, livox::PORT_IMU)) {
            std::cerr << "Warning: Failed to configure IMU destination\n";
        }

        // Step 3: Set work mode to Normal (streaming)
        sendWorkModeCommand(0x02);

        // Give the LiDAR time to process configuration
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        return true;
    }

    bool sendHostConfig(uint16_t key, uint16_t host_port, uint16_t lidar_port) {
        // SDK2 format: 11-byte header + data + 4-byte CRC32
        // Data: subcmd(1) + key_num(1) + key(2) + len(2) + ip(4) + dst_port(2) + src_port(2) = 14 bytes
        size_t data_len = 1 + 1 + 2 + 2 + 8;  // = 14 bytes
        size_t frame_size = 11 + data_len + 4;
        std::vector<uint8_t> frame(frame_size, 0);

        // Header (11 bytes)
        frame[0] = livox::FRAME_START;
        frame[1] = 0;
        *reinterpret_cast<uint16_t*>(&frame[2]) = static_cast<uint16_t>(frame_size);
        *reinterpret_cast<uint16_t*>(&frame[4]) = cmd_seq_++;
        frame[6] = livox::CMD_LIDAR;
        frame[7] = livox::CMD_TYPE_REQ;
        frame[8] = 0;

        // CRC16 over bytes 0-8
        *reinterpret_cast<uint16_t*>(&frame[9]) = LivoxCRC::crc16(frame.data(), 9);

        // Data section (starts at offset 11)
        uint8_t* data = frame.data() + 11;
        data[0] = livox::SUBCMD_PARAM_CONFIG;  // sub_cmd
        data[1] = 1;                            // key_num
        *reinterpret_cast<uint16_t*>(data + 2) = key;
        *reinterpret_cast<uint16_t*>(data + 4) = 8;  // length = 8 bytes for IP config

        // IP config value (8 bytes)
        in_addr addr;
        inet_pton(AF_INET, host_ip_.c_str(), &addr);
        memcpy(data + 6, &addr.s_addr, 4);                     // Host IP
        *reinterpret_cast<uint16_t*>(data + 10) = host_port;   // Destination port
        *reinterpret_cast<uint16_t*>(data + 12) = lidar_port;  // Source port

        // CRC32 at end of frame
        *reinterpret_cast<uint32_t*>(&frame[frame_size - 4]) =
            LivoxCRC::crc32(data, data_len);

        // Send
        sockaddr_in dev_addr = {};
        dev_addr.sin_family = AF_INET;
        dev_addr.sin_port = htons(livox::PORT_COMMAND);
        inet_pton(AF_INET, device_ip_.c_str(), &dev_addr.sin_addr);

        int sent = sendto(cmd_socket_, (const char*)frame.data(), (int)frame.size(), 0,
                          (sockaddr*)&dev_addr, sizeof(dev_addr));

        return sent > 0;
    }

    void sendParamCommand(uint16_t key, uint8_t value) {
        // SDK2 format: 11-byte header + data + 4-byte CRC32
        // Data: subcmd(1) + key_num(1) + key(2) + len(2) + value(1) = 7 bytes
        size_t data_len = 1 + 1 + 2 + 2 + 1;  // = 7 bytes
        size_t frame_size = 11 + data_len + 4;
        std::vector<uint8_t> frame(frame_size, 0);

        // Header (11 bytes)
        frame[0] = livox::FRAME_START;
        frame[1] = 0;
        *reinterpret_cast<uint16_t*>(&frame[2]) = static_cast<uint16_t>(frame_size);
        *reinterpret_cast<uint16_t*>(&frame[4]) = cmd_seq_++;
        frame[6] = livox::CMD_LIDAR;
        frame[7] = livox::CMD_TYPE_REQ;
        frame[8] = 0;

        // CRC16 over bytes 0-8
        *reinterpret_cast<uint16_t*>(&frame[9]) = LivoxCRC::crc16(frame.data(), 9);

        // Data section (starts at offset 11)
        uint8_t* data = frame.data() + 11;
        data[0] = livox::SUBCMD_PARAM_CONFIG;  // sub_cmd
        data[1] = 1;                            // key_num
        *reinterpret_cast<uint16_t*>(data + 2) = key;
        *reinterpret_cast<uint16_t*>(data + 4) = 1;   // length = 1 byte
        data[6] = value;

        // CRC32 at end of frame
        *reinterpret_cast<uint32_t*>(&frame[frame_size - 4]) =
            LivoxCRC::crc32(data, data_len);

        // Send to device
        sockaddr_in dev_addr = {};
        dev_addr.sin_family = AF_INET;
        dev_addr.sin_port = htons(livox::PORT_COMMAND);
        inet_pton(AF_INET, device_ip_.c_str(), &dev_addr.sin_addr);

        // Debug: dump the command frame
        std::cout << "  Sending to " << device_ip_ << ":56100, " << frame.size() << " bytes:\n    ";
        for (size_t i = 0; i < frame.size(); i++) {
            std::cout << std::hex << std::setfill('0') << std::setw(2) << (int)frame[i] << " ";
            if ((i + 1) % 16 == 0) std::cout << "\n    ";
        }
        std::cout << std::dec << "\n";

        sendto(cmd_socket_, (const char*)frame.data(), (int)frame.size(), 0,
               (sockaddr*)&dev_addr, sizeof(dev_addr));

        std::cout << "  Sent param 0x" << std::hex << key << " = " << std::dec << (int)value;

        // Wait for ack (with timeout)
        waitForAck(key);
    }

    // Quiet version for use during shutdown (no debug output)
    void sendParamCommandQuiet(uint16_t key, uint8_t value) {
        if (cmd_socket_ == INVALID_SOCKET || device_ip_.empty()) return;

        size_t data_len = 1 + 1 + 2 + 2 + 1;
        size_t frame_size = 11 + data_len + 4;
        std::vector<uint8_t> frame(frame_size, 0);

        frame[0] = livox::FRAME_START;
        frame[1] = 0;
        *reinterpret_cast<uint16_t*>(&frame[2]) = static_cast<uint16_t>(frame_size);
        *reinterpret_cast<uint16_t*>(&frame[4]) = cmd_seq_++;
        frame[6] = livox::CMD_LIDAR;
        frame[7] = livox::CMD_TYPE_REQ;
        frame[8] = 0;
        *reinterpret_cast<uint16_t*>(&frame[9]) = LivoxCRC::crc16(frame.data(), 9);

        uint8_t* data = frame.data() + 11;
        data[0] = livox::SUBCMD_PARAM_CONFIG;
        data[1] = 1;
        *reinterpret_cast<uint16_t*>(data + 2) = key;
        *reinterpret_cast<uint16_t*>(data + 4) = 1;
        data[6] = value;
        *reinterpret_cast<uint32_t*>(&frame[frame_size - 4]) =
            LivoxCRC::crc32(data, data_len);

        sockaddr_in dev_addr = {};
        dev_addr.sin_family = AF_INET;
        dev_addr.sin_port = htons(livox::PORT_COMMAND);
        inet_pton(AF_INET, device_ip_.c_str(), &dev_addr.sin_addr);

        sendto(cmd_socket_, (const char*)frame.data(), (int)frame.size(), 0,
               (sockaddr*)&dev_addr, sizeof(dev_addr));
    }

    void waitForAck(uint16_t expected_key) {
        // Set socket to non-blocking temporarily
        u_long mode = 1;
        ioctlsocket(cmd_socket_, FIONBIO, &mode);

        uint8_t buffer[512];
        auto start = std::chrono::steady_clock::now();

        while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(200)) {
            int recv_len = recv(cmd_socket_, (char*)buffer, sizeof(buffer), 0);
            if (recv_len > 0) {
                // Parse response
                if (recv_len >= sizeof(LivoxCmdHeader)) {
                    const LivoxCmdHeader* ack = reinterpret_cast<const LivoxCmdHeader*>(buffer);
                    if (ack->sof == livox::FRAME_START && ack->cmd_type == livox::CMD_TYPE_ACK) {
                        // Check return code in data
                        if (recv_len > sizeof(LivoxCmdHeader)) {
                            uint8_t ret_code = buffer[sizeof(LivoxCmdHeader)];
                            std::cout << " -> ACK (ret=" << (int)ret_code << ")\n";
                        } else {
                            std::cout << " -> ACK\n";
                        }
                        mode = 0;
                        ioctlsocket(cmd_socket_, FIONBIO, &mode);
                        return;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        std::cout << " -> NO ACK (timeout)\n";
        mode = 0;
        ioctlsocket(cmd_socket_, FIONBIO, &mode);
    }

    void sendWorkModeCommand(uint8_t mode) {
        // Use generic param command for work mode
        sendParamCommand(livox::PARAM_WORK_MODE, mode);
    }

    void receiveLoop() {
        uint8_t buffer[2048];

        std::cout << "Starting point cloud reception...\n";

        // Increase socket receive buffer to handle burst traffic
        int rcvbuf = 4 * 1024 * 1024;  // 4MB buffer
        setsockopt(pointcloud_socket_, SOL_SOCKET, SO_RCVBUF, (char*)&rcvbuf, sizeof(rcvbuf));
        setsockopt(imu_socket_, SOL_SOCKET, SO_RCVBUF, (char*)&rcvbuf, sizeof(rcvbuf));

        while (streaming_) {
            // Use select() to wait for data with timeout
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(pointcloud_socket_, &readfds);
            FD_SET(imu_socket_, &readfds);

            timeval timeout;
            timeout.tv_sec = 0;
            timeout.tv_usec = 1000;  // 1ms timeout

            SOCKET max_fd = (std::max)(pointcloud_socket_, imu_socket_) + 1;
            int ready = select((int)max_fd, &readfds, nullptr, nullptr, &timeout);

            if (ready > 0) {
                // Drain ALL available point cloud packets
                if (FD_ISSET(pointcloud_socket_, &readfds)) {
                    int recv_len;
                    while ((recv_len = recv(pointcloud_socket_, (char*)buffer, sizeof(buffer), 0)) > 0) {
                        if (recv_len > static_cast<int>(sizeof(LivoxPointCloudHeader))) {
                            processPointCloudPacket(buffer, recv_len);
                        }
                    }
                }

                // Drain ALL available IMU packets
                if (FD_ISSET(imu_socket_, &readfds)) {
                    int recv_len;
                    while ((recv_len = recv(imu_socket_, (char*)buffer, sizeof(buffer), 0)) > 0) {
                        processIMUPacket(buffer, recv_len);
                    }
                }
            }
        }
    }

    void processPointCloudPacket(const uint8_t* data, size_t len) {
        if (len < sizeof(LivoxPointCloudHeader)) return;

        const LivoxPointCloudHeader* hdr =
            reinterpret_cast<const LivoxPointCloudHeader*>(data);

        // Skip if no points
        if (hdr->dot_num == 0) return;

        const uint8_t* point_data = data + sizeof(LivoxPointCloudHeader);
        size_t point_data_len = len - sizeof(LivoxPointCloudHeader);

        LivoxPointCloudFrame frame;
        frame.timestamp_ns = hdr->timestamp;

        // Calculate per-point time offset
        // time_interval is in 0.1 microseconds, convert to microseconds
        float total_time_us = hdr->time_interval * 0.1f;
        uint16_t total_points_in_packet = hdr->dot_num;

        // Debug: Log time_interval values periodically
        static int time_debug_count = 0;
        if (time_debug_count++ < 5) {
            std::cout << "[DEBUG time_interval] raw=" << hdr->time_interval
                      << " -> " << total_time_us << "us, points=" << total_points_in_packet << std::endl;
        }

        // Parse based on data type
        static int unknown_type_count = 0;
        if (hdr->data_type == 1) {
            // 32-bit Cartesian (most common)
            size_t point_size = sizeof(LivoxPointCartesian32);
            size_t num_points = std::min((size_t)hdr->dot_num,
                                         point_data_len / point_size);

            frame.points.reserve(num_points);
            frame.reflectivities.reserve(num_points);
            frame.tags.reserve(num_points);
            frame.time_offsets_us.reserve(num_points);

            for (size_t i = 0; i < num_points; i++) {
                const LivoxPointCartesian32* pt =
                    reinterpret_cast<const LivoxPointCartesian32*>(
                        point_data + i * point_size);

                // Convert from mm to meters
                V3D p(pt->x * 0.001, pt->y * 0.001, pt->z * 0.001);

                // Filter invalid points
                if (p.norm() > 0.1 && p.norm() < 200.0) {
                    frame.points.push_back(p);
                    frame.reflectivities.push_back(pt->reflectivity);
                    frame.tags.push_back(pt->tag);
                    // Calculate time offset for this point (equally spaced within packet)
                    float point_time_us = (total_points_in_packet > 1) ?
                        (i * total_time_us / (total_points_in_packet - 1)) : 0.0f;
                    frame.time_offsets_us.push_back(point_time_us);
                }
            }
        }
        else if (hdr->data_type == 2) {
            // 16-bit Cartesian
            size_t point_size = sizeof(LivoxPointCartesian16);
            size_t num_points = std::min((size_t)hdr->dot_num,
                                         point_data_len / point_size);

            frame.points.reserve(num_points);
            frame.time_offsets_us.reserve(num_points);

            for (size_t i = 0; i < num_points; i++) {
                const LivoxPointCartesian16* pt =
                    reinterpret_cast<const LivoxPointCartesian16*>(
                        point_data + i * point_size);

                // Convert from 10mm to meters
                V3D p(pt->x * 0.01, pt->y * 0.01, pt->z * 0.01);

                if (p.norm() > 0.1 && p.norm() < 200.0) {
                    frame.points.push_back(p);
                    frame.reflectivities.push_back(pt->reflectivity);
                    frame.tags.push_back(pt->tag);
                    // Calculate time offset for this point (equally spaced within packet)
                    float point_time_us = (total_points_in_packet > 1) ?
                        (i * total_time_us / (total_points_in_packet - 1)) : 0.0f;
                    frame.time_offsets_us.push_back(point_time_us);
                }
            }
        }
        else {
            // Unknown/unsupported data type
            if (unknown_type_count++ < 10) {
                std::cerr << "[Livox] Unsupported data_type=" << (int)hdr->data_type
                          << " (only types 1,2 supported). dot_num=" << hdr->dot_num << std::endl;
            }
        }

        if (!frame.points.empty()) {
            total_points_ += frame.points.size();
            frame_count_++;

            if (pointcloud_callback_) {
                std::lock_guard<std::mutex> lock(callback_mutex_);
                pointcloud_callback_(frame);
            }
        }
    }

    void processIMUPacket(const uint8_t* data, size_t len) {
        // IMU packets have similar header structure
        if (len < sizeof(LivoxPointCloudHeader) + sizeof(LivoxIMUData)) return;

        const LivoxPointCloudHeader* hdr =
            reinterpret_cast<const LivoxPointCloudHeader*>(data);

        if (hdr->data_type != 0) return;  // Type 0 = IMU

        const LivoxIMUData* imu =
            reinterpret_cast<const LivoxIMUData*>(data + sizeof(LivoxPointCloudHeader));

        LivoxIMUFrame frame;
        frame.timestamp_ns = hdr->timestamp;
        frame.gyro = V3D(imu->gyro_x, imu->gyro_y, imu->gyro_z);
        // Keep acceleration in g-units (SLAM engine scales by G_m_s2 / mean_acc_.norm())
        frame.accel = V3D(imu->acc_x, imu->acc_y, imu->acc_z);

        if (imu_callback_) {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            imu_callback_(frame);
        }
    }
};

//=============================================================================
// Network Diagnostics
//=============================================================================

struct NetworkDiagnostic {
    bool ethernet_found = false;
    bool ethernet_connected = false;
    bool correct_subnet = false;
    std::string ethernet_ip;
    std::string adapter_name;
    std::vector<std::string> issues;
    std::string recommended_ip = "192.168.1.50";
};

/**
 * Check if an IP address is in the Livox subnet (192.168.1.x)
 */
inline bool isLivoxSubnet(const std::string& ip) {
    return ip.find("192.168.1.") == 0;
}

/**
 * Get local IP addresses and check network configuration
 */
inline NetworkDiagnostic checkNetworkConfiguration() {
    NetworkDiagnostic diag;

#ifdef _WIN32
    // Initialize Winsock
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        addrinfo hints = {};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_DGRAM;

        addrinfo* result = nullptr;
        if (getaddrinfo(hostname, nullptr, &hints, &result) == 0) {
            for (addrinfo* ptr = result; ptr != nullptr; ptr = ptr->ai_next) {
                sockaddr_in* addr = reinterpret_cast<sockaddr_in*>(ptr->ai_addr);
                char ip[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, &addr->sin_addr, ip, sizeof(ip));

                std::string ip_str(ip);

                // Skip loopback
                if (ip_str.find("127.") == 0) continue;

                // Check for Livox subnet
                if (isLivoxSubnet(ip_str)) {
                    diag.ethernet_found = true;
                    diag.ethernet_connected = true;
                    diag.correct_subnet = true;
                    diag.ethernet_ip = ip_str;
                }
            }
            freeaddrinfo(result);
        }
    }

    // If no 192.168.1.x found, check for any Ethernet adapter
    if (!diag.correct_subnet) {
        // Try to find an adapter with 169.254.x.x (unconfigured) or other
        // This is a simplified check - in production would use GetAdaptersInfo
        diag.issues.push_back("No network adapter found with IP in 192.168.1.x subnet");
        diag.issues.push_back("The Livox Mid-360 uses 192.168.1.1XX by default");
    }

    WSACleanup();
#endif

    return diag;
}

/**
 * Print network configuration guidance
 */
inline void printNetworkGuidance(const NetworkDiagnostic& diag) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           NETWORK CONFIGURATION REQUIRED                     ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  The Livox Mid-360 requires your Ethernet adapter to be      ║\n";
    std::cout << "║  configured on the 192.168.1.x subnet.                       ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  OPTION 1: Run as Administrator (one-time setup)             ║\n";
    std::cout << "║  ──────────────────────────────────────────────────────────  ║\n";
    std::cout << "║  Open CMD/PowerShell as Admin and run:                       ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  netsh interface ip set address \"Ethernet\" static \\         ║\n";
    std::cout << "║        192.168.1.50 255.255.255.0                            ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  OPTION 2: Windows Settings (GUI)                            ║\n";
    std::cout << "║  ──────────────────────────────────────────────────────────  ║\n";
    std::cout << "║  1. Press Win+R, type: ncpa.cpl                              ║\n";
    std::cout << "║  2. Right-click 'Ethernet' → Properties                      ║\n";
    std::cout << "║  3. Select 'Internet Protocol Version 4' → Properties        ║\n";
    std::cout << "║  4. Select 'Use the following IP address':                   ║\n";
    std::cout << "║       IP address:  192.168.1.50                              ║\n";
    std::cout << "║       Subnet mask: 255.255.255.0                             ║\n";
    std::cout << "║       Gateway:     (leave empty)                             ║\n";
    std::cout << "║  5. Click OK → OK                                            ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  NOTE: This is a one-time setup. Windows remembers the       ║\n";
    std::cout << "║  configuration for next time you connect the LiDAR.          ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  The Mid-360's default IP is 192.168.1.1XX where XX is       ║\n";
    std::cout << "║  the last two digits of its serial number.                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

/**
 * Print connection troubleshooting
 */
inline void printConnectionTroubleshooting() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           LIVOX MID-360 NOT DETECTED                         ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  Please verify:                                              ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  1. POWER                                                    ║\n";
    std::cout << "║     □ LiDAR is connected to power supply (9-27V DC)          ║\n";
    std::cout << "║     □ Wait 10+ seconds after power-on for boot               ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  2. ETHERNET CONNECTION                                      ║\n";
    std::cout << "║     □ Ethernet cable securely connected to LiDAR and PC      ║\n";
    std::cout << "║     □ PC Ethernet port shows link activity (green light)     ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  3. NETWORK CONFIGURATION                                    ║\n";
    std::cout << "║     □ PC Ethernet IP is 192.168.1.50                         ║\n";
    std::cout << "║     □ Subnet mask is 255.255.255.0                           ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "║  4. FIREWALL                                                 ║\n";
    std::cout << "║     □ Allow UDP ports 56000-56500 (or disable temporarily)   ║\n";
    std::cout << "║     □ Check Windows Defender Firewall settings               ║\n";
    std::cout << "║                                                              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

//=============================================================================
// Convenience function for quick device scan
//=============================================================================

inline void scanLivoxDevices(bool show_guidance = true) {
    // First check network configuration
    auto diag = checkNetworkConfiguration();

    if (!diag.correct_subnet && show_guidance) {
        printNetworkGuidance(diag);
        return;
    }

    std::cout << "Network: OK (" << diag.ethernet_ip << ")\n";

    LivoxMid360 scanner;
    auto devices = scanner.discover(3000, diag.ethernet_ip);

    if (devices.empty()) {
        if (show_guidance) {
            printConnectionTroubleshooting();
        } else {
            std::cout << "No Livox devices found.\n";
        }
    } else {
        std::cout << "\nFound " << devices.size() << " device(s):\n";
        for (const auto& dev : devices) {
            std::cout << "  ✓ " << dev.getTypeName()
                      << " [SN: " << dev.serial_number << "]"
                      << " @ " << dev.ip_address << "\n";
        }
    }
}

} // namespace slam
