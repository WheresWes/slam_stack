#pragma once

/**
 * VESC CAN Interface
 *
 * Platform-specific CAN bus communication for VESC motor controllers.
 * Uses SLCAN (Serial Line CAN) protocol for USB-to-CAN adapters like CANable.
 */

#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>

#ifdef _WIN32
#include <windows.h>
#endif

namespace slam {

/**
 * CAN message structure
 */
struct CanMessage {
    uint32_t id;           // CAN ID (29-bit for extended)
    bool is_extended;      // Extended (29-bit) vs standard (11-bit) ID
    uint8_t dlc;           // Data length code (0-8)
    uint8_t data[8];       // Data bytes

    CanMessage() : id(0), is_extended(true), dlc(0) {
        memset(data, 0, sizeof(data));
    }
};

/**
 * CAN message callback type
 */
using CanMessageCallback = std::function<void(const CanMessage&)>;

/**
 * SLCAN-based CAN interface for Windows
 *
 * Implements the SLCAN (Serial Line CAN) protocol used by CANable and similar adapters.
 * SLCAN commands:
 *   - O: Open CAN channel
 *   - C: Close CAN channel
 *   - S4: Set bitrate to 125k (S0=10k, S1=20k, S2=50k, S3=100k, S4=125k, S5=250k, S6=500k, S7=800k, S8=1M)
 *   - T: Transmit extended frame (T + 8 hex ID + 1 hex DLC + data)
 *   - t: Transmit standard frame
 *   - r: Received standard frame
 *   - R: Received extended frame
 */
class VescCanInterface {
public:
    VescCanInterface();
    ~VescCanInterface();

    /**
     * Open CAN interface
     * @param port Serial port (e.g., "COM3" on Windows)
     * @param bitrate CAN bitrate (typically 500000 for VESC)
     * @return true if successful
     */
    bool open(const std::string& port, int bitrate = 500000);

    /**
     * Close CAN interface
     */
    void close();

    /**
     * Check if interface is open
     */
    bool isOpen() const { return is_open_; }

    /**
     * Send CAN message
     * @return true if sent successfully
     */
    bool send(const CanMessage& msg);

    /**
     * Send CAN message (convenience method)
     */
    bool send(uint32_t id, const uint8_t* data, uint8_t len, bool extended = true);

    /**
     * Set callback for received messages
     */
    void setReceiveCallback(CanMessageCallback callback);

    /**
     * Get last error message
     */
    std::string getLastError() const { return last_error_; }

    /**
     * Get number of messages sent/received
     */
    uint64_t getMessagesSent() const { return messages_sent_; }
    uint64_t getMessagesReceived() const { return messages_received_; }

private:
    // Serial port handle
#ifdef _WIN32
    HANDLE serial_handle_ = INVALID_HANDLE_VALUE;
#else
    int serial_fd_ = -1;
#endif

    std::atomic<bool> is_open_{false};
    std::string last_error_;

    // Receive thread
    std::thread rx_thread_;
    std::atomic<bool> rx_running_{false};

    // Callback
    std::mutex callback_mutex_;
    CanMessageCallback rx_callback_;

    // Statistics
    std::atomic<uint64_t> messages_sent_{0};
    std::atomic<uint64_t> messages_received_{0};

    // Internal methods
    bool openSerial(const std::string& port);
    void closeSerial();
    bool writeSerial(const char* data, size_t len);
    int readSerial(char* buffer, size_t max_len, int timeout_ms = 100);

    void rxThreadFunc();
    bool parseSlcanMessage(const char* msg, CanMessage& out);
    std::string encodeSlcanMessage(const CanMessage& msg);

    int bitrateToSlcanCode(int bitrate);
};

// ============================================================================
// VESC CAN Protocol Helpers
// ============================================================================

namespace vesc {

// VESC CAN command IDs (bits 15-8 of extended CAN ID)
enum class Command : uint8_t {
    SET_DUTY = 0,
    SET_CURRENT = 1,
    SET_CURRENT_BRAKE = 2,
    SET_RPM = 3,
    SET_POS = 4,
    SET_CURRENT_REL = 10,
    SET_CURRENT_BRAKE_REL = 11,
    SET_CURRENT_HANDBRAKE = 12,
    SET_CURRENT_HANDBRAKE_REL = 13,
    STATUS = 9,
    STATUS_2 = 14,
    STATUS_3 = 15,
    STATUS_4 = 16,
    STATUS_5 = 27,
};

/**
 * Build VESC CAN ID
 * Extended CAN ID format: [unused:13][command:8][vesc_id:8]
 */
inline uint32_t makeCanId(Command cmd, uint8_t vesc_id) {
    return (static_cast<uint32_t>(cmd) << 8) | vesc_id;
}

/**
 * Parse VESC CAN ID
 */
inline void parseCanId(uint32_t can_id, Command& cmd, uint8_t& vesc_id) {
    vesc_id = can_id & 0xFF;
    cmd = static_cast<Command>((can_id >> 8) & 0xFF);
}

/**
 * Encode duty cycle command
 * @param duty -1.0 to 1.0
 */
inline void encodeDuty(float duty, uint8_t* data) {
    int32_t value = static_cast<int32_t>(duty * 100000.0f);
    data[0] = (value >> 24) & 0xFF;
    data[1] = (value >> 16) & 0xFF;
    data[2] = (value >> 8) & 0xFF;
    data[3] = value & 0xFF;
}

/**
 * Encode RPM command
 */
inline void encodeRpm(int32_t erpm, uint8_t* data) {
    data[0] = (erpm >> 24) & 0xFF;
    data[1] = (erpm >> 16) & 0xFF;
    data[2] = (erpm >> 8) & 0xFF;
    data[3] = erpm & 0xFF;
}

/**
 * Encode current command
 * @param current in Amps
 */
inline void encodeCurrent(float current, uint8_t* data) {
    int32_t value = static_cast<int32_t>(current * 1000.0f);
    data[0] = (value >> 24) & 0xFF;
    data[1] = (value >> 16) & 0xFF;
    data[2] = (value >> 8) & 0xFF;
    data[3] = value & 0xFF;
}

/**
 * Decode STATUS message (ID 9)
 * Returns: ERPM, current (A), duty cycle
 */
inline void decodeStatus(const uint8_t* data, int32_t& erpm, float& current, float& duty) {
    erpm = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
    int16_t current_raw = (data[4] << 8) | data[5];
    int16_t duty_raw = (data[6] << 8) | data[7];
    current = current_raw / 10.0f;
    duty = duty_raw / 1000.0f;
}

/**
 * Decode STATUS_4 message (ID 16)
 * Returns: FET temp, motor temp, input current, PID position
 */
inline void decodeStatus4(const uint8_t* data, float& temp_fet, float& temp_motor,
                          float& current_in, float& pid_pos) {
    int16_t temp_fet_raw = (data[0] << 8) | data[1];
    int16_t temp_motor_raw = (data[2] << 8) | data[3];
    int16_t current_in_raw = (data[4] << 8) | data[5];
    int16_t pid_pos_raw = (data[6] << 8) | data[7];
    temp_fet = temp_fet_raw / 10.0f;
    temp_motor = temp_motor_raw / 10.0f;
    current_in = current_in_raw / 10.0f;
    pid_pos = pid_pos_raw / 50.0f;
}

/**
 * Decode STATUS_5 message (ID 27)
 * Returns: tachometer, input voltage
 */
inline void decodeStatus5(const uint8_t* data, int32_t& tachometer, float& voltage) {
    tachometer = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
    int16_t voltage_raw = (data[4] << 8) | data[5];
    voltage = voltage_raw / 10.0f;
}

} // namespace vesc

} // namespace slam
