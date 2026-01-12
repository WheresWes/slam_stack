/**
 * VESC CAN Interface Implementation
 *
 * SLCAN (Serial Line CAN) protocol for Windows.
 */

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "slam/vesc_can_interface.hpp"
#include <cstring>
#include <cstdio>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#endif

namespace slam {

VescCanInterface::VescCanInterface() = default;

VescCanInterface::~VescCanInterface() {
    close();
}

bool VescCanInterface::open(const std::string& port, int bitrate) {
    if (is_open_) {
        close();
    }

    // Open serial port
    if (!openSerial(port)) {
        return false;
    }

    // Configure SLCAN
    // Close any existing CAN channel
    writeSerial("C\r", 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Set bitrate
    int slcan_code = bitrateToSlcanCode(bitrate);
    if (slcan_code < 0) {
        last_error_ = "Unsupported bitrate: " + std::to_string(bitrate);
        closeSerial();
        return false;
    }

    char cmd[8];
    snprintf(cmd, sizeof(cmd), "S%d\r", slcan_code);
    if (!writeSerial(cmd, strlen(cmd))) {
        last_error_ = "Failed to set bitrate";
        closeSerial();
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Open CAN channel
    if (!writeSerial("O\r", 2)) {
        last_error_ = "Failed to open CAN channel";
        closeSerial();
        return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    is_open_ = true;

    // Start receive thread
    rx_running_ = true;
    rx_thread_ = std::thread(&VescCanInterface::rxThreadFunc, this);

    return true;
}

void VescCanInterface::close() {
    if (!is_open_) return;

    // Stop receive thread
    rx_running_ = false;
    if (rx_thread_.joinable()) {
        rx_thread_.join();
    }

    // Close CAN channel
    writeSerial("C\r", 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    closeSerial();
    is_open_ = false;
}

bool VescCanInterface::send(const CanMessage& msg) {
    if (!is_open_) return false;

    std::string encoded = encodeSlcanMessage(msg);
    if (encoded.empty()) {
        return false;
    }

    if (writeSerial(encoded.c_str(), encoded.length())) {
        messages_sent_++;
        return true;
    }
    return false;
}

bool VescCanInterface::send(uint32_t id, const uint8_t* data, uint8_t len, bool extended) {
    CanMessage msg;
    msg.id = id;
    msg.is_extended = extended;
    msg.dlc = len;
    if (len > 0 && data) {
        memcpy(msg.data, data, std::min(len, (uint8_t)8));
    }
    return send(msg);
}

void VescCanInterface::setReceiveCallback(CanMessageCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    rx_callback_ = callback;
}

// ============================================================================
// Platform-specific serial port implementation
// ============================================================================

#ifdef _WIN32

bool VescCanInterface::openSerial(const std::string& port) {
    // Windows COM port path
    std::string full_port = "\\\\.\\" + port;

    serial_handle_ = CreateFileA(
        full_port.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        0,
        NULL
    );

    if (serial_handle_ == INVALID_HANDLE_VALUE) {
        last_error_ = "Failed to open " + port + " (error " + std::to_string(GetLastError()) + ")";
        return false;
    }

    // Configure serial port
    DCB dcb = {0};
    dcb.DCBlength = sizeof(DCB);

    if (!GetCommState(serial_handle_, &dcb)) {
        last_error_ = "GetCommState failed";
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    dcb.BaudRate = CBR_115200;  // SLCAN uses 115200 baud
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fParity = FALSE;
    dcb.fOutxCtsFlow = FALSE;
    dcb.fOutxDsrFlow = FALSE;
    dcb.fDtrControl = DTR_CONTROL_DISABLE;
    dcb.fRtsControl = RTS_CONTROL_ENABLE;  // CANable needs RTS
    dcb.fOutX = FALSE;
    dcb.fInX = FALSE;

    if (!SetCommState(serial_handle_, &dcb)) {
        last_error_ = "SetCommState failed";
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    // Set timeouts
    COMMTIMEOUTS timeouts = {0};
    timeouts.ReadIntervalTimeout = 10;
    timeouts.ReadTotalTimeoutConstant = 100;
    timeouts.ReadTotalTimeoutMultiplier = 1;
    timeouts.WriteTotalTimeoutConstant = 100;
    timeouts.WriteTotalTimeoutMultiplier = 1;

    if (!SetCommTimeouts(serial_handle_, &timeouts)) {
        last_error_ = "SetCommTimeouts failed";
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
        return false;
    }

    // Purge any existing data
    PurgeComm(serial_handle_, PURGE_RXCLEAR | PURGE_TXCLEAR);

    return true;
}

void VescCanInterface::closeSerial() {
    if (serial_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
    }
}

bool VescCanInterface::writeSerial(const char* data, size_t len) {
    if (serial_handle_ == INVALID_HANDLE_VALUE) return false;

    DWORD written;
    if (!WriteFile(serial_handle_, data, (DWORD)len, &written, NULL)) {
        return false;
    }
    return written == len;
}

int VescCanInterface::readSerial(char* buffer, size_t max_len, int timeout_ms) {
    if (serial_handle_ == INVALID_HANDLE_VALUE) return -1;

    // Update timeout
    COMMTIMEOUTS timeouts = {0};
    timeouts.ReadIntervalTimeout = 10;
    timeouts.ReadTotalTimeoutConstant = timeout_ms;
    timeouts.ReadTotalTimeoutMultiplier = 0;
    SetCommTimeouts(serial_handle_, &timeouts);

    DWORD bytesRead;
    if (!ReadFile(serial_handle_, buffer, (DWORD)max_len, &bytesRead, NULL)) {
        return -1;
    }
    return (int)bytesRead;
}

#else // Linux/Unix

bool VescCanInterface::openSerial(const std::string& port) {
    serial_fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (serial_fd_ < 0) {
        last_error_ = "Failed to open " + port;
        return false;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));

    if (tcgetattr(serial_fd_, &tty) != 0) {
        last_error_ = "tcgetattr failed";
        ::close(serial_fd_);
        serial_fd_ = -1;
        return false;
    }

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag |= CLOCAL | CREAD;

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);

    tty.c_lflag = 0;
    tty.c_oflag = 0;

    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;

    if (tcsetattr(serial_fd_, TCSANOW, &tty) != 0) {
        last_error_ = "tcsetattr failed";
        ::close(serial_fd_);
        serial_fd_ = -1;
        return false;
    }

    tcflush(serial_fd_, TCIOFLUSH);
    return true;
}

void VescCanInterface::closeSerial() {
    if (serial_fd_ >= 0) {
        ::close(serial_fd_);
        serial_fd_ = -1;
    }
}

bool VescCanInterface::writeSerial(const char* data, size_t len) {
    if (serial_fd_ < 0) return false;
    return ::write(serial_fd_, data, len) == (ssize_t)len;
}

int VescCanInterface::readSerial(char* buffer, size_t max_len, int timeout_ms) {
    if (serial_fd_ < 0) return -1;

    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(serial_fd_, &readfds);

    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    int ret = select(serial_fd_ + 1, &readfds, NULL, NULL, &tv);
    if (ret <= 0) return ret;

    return ::read(serial_fd_, buffer, max_len);
}

#endif // _WIN32

// ============================================================================
// SLCAN Protocol
// ============================================================================

void VescCanInterface::rxThreadFunc() {
    char buffer[256];
    char msg_buffer[64];
    int msg_len = 0;

    while (rx_running_) {
        int n = readSerial(buffer, sizeof(buffer) - 1, 50);
        if (n <= 0) continue;

        for (int i = 0; i < n; i++) {
            char c = buffer[i];

            if (c == '\r' || c == '\n') {
                if (msg_len > 0) {
                    msg_buffer[msg_len] = '\0';

                    CanMessage msg;
                    if (parseSlcanMessage(msg_buffer, msg)) {
                        messages_received_++;

                        std::lock_guard<std::mutex> lock(callback_mutex_);
                        if (rx_callback_) {
                            rx_callback_(msg);
                        }
                    }

                    msg_len = 0;
                }
            } else if (msg_len < (int)sizeof(msg_buffer) - 1) {
                msg_buffer[msg_len++] = c;
            }
        }
    }
}

bool VescCanInterface::parseSlcanMessage(const char* msg, CanMessage& out) {
    if (!msg || strlen(msg) < 5) return false;

    char type = msg[0];
    bool extended = (type == 'T' || type == 'R');
    bool is_tx = (type == 't' || type == 'T');
    bool is_rx = (type == 'r' || type == 'R');

    if (!is_tx && !is_rx) return false;

    out.is_extended = extended;

    // Parse ID
    int id_len = extended ? 8 : 3;
    char id_str[9] = {0};
    strncpy(id_str, msg + 1, id_len);
    out.id = strtoul(id_str, NULL, 16);

    // Parse DLC
    int dlc_pos = 1 + id_len;
    if (strlen(msg) <= (size_t)dlc_pos) return false;

    out.dlc = msg[dlc_pos] - '0';
    if (out.dlc > 8) return false;

    // Parse data
    int data_pos = dlc_pos + 1;
    for (int i = 0; i < out.dlc; i++) {
        if (strlen(msg) < (size_t)(data_pos + i * 2 + 2)) break;
        char byte_str[3] = {msg[data_pos + i * 2], msg[data_pos + i * 2 + 1], 0};
        out.data[i] = (uint8_t)strtoul(byte_str, NULL, 16);
    }

    return true;
}

std::string VescCanInterface::encodeSlcanMessage(const CanMessage& msg) {
    char buffer[32];

    if (msg.is_extended) {
        // Extended frame: T + 8 hex ID + 1 hex DLC + data + \r
        snprintf(buffer, sizeof(buffer), "T%08X%d", msg.id, msg.dlc);
    } else {
        // Standard frame: t + 3 hex ID + 1 hex DLC + data + \r
        snprintf(buffer, sizeof(buffer), "t%03X%d", msg.id & 0x7FF, msg.dlc);
    }

    std::string result = buffer;

    for (int i = 0; i < msg.dlc; i++) {
        snprintf(buffer, sizeof(buffer), "%02X", msg.data[i]);
        result += buffer;
    }

    result += "\r";
    return result;
}

int VescCanInterface::bitrateToSlcanCode(int bitrate) {
    switch (bitrate) {
        case 10000:   return 0;
        case 20000:   return 1;
        case 50000:   return 2;
        case 100000:  return 3;
        case 125000:  return 4;
        case 250000:  return 5;
        case 500000:  return 6;
        case 800000:  return 7;
        case 1000000: return 8;
        default:      return -1;
    }
}

} // namespace slam
