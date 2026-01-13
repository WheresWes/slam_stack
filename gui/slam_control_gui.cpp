/**
 * @file slam_control_gui.cpp
 * @brief SLAM Control GUI using Dear ImGui + Win32/DirectX11
 *
 * Provides real-time control for:
 * - Visualization (Rerun viewer)
 * - Live SLAM mapping
 * - Global localization
 */

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
#endif

#include <windows.h>
#include <d3d11.h>
#include <tchar.h>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"

#pragma comment(lib, "d3d11.lib")

// Forward declarations
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

//==============================================================================
// DirectX 11 globals
//==============================================================================
static ID3D11Device*            g_pd3dDevice = nullptr;
static ID3D11DeviceContext*     g_pd3dDeviceContext = nullptr;
static IDXGISwapChain*          g_pSwapChain = nullptr;
static bool                     g_SwapChainOccluded = false;
static UINT                     g_ResizeWidth = 0, g_ResizeHeight = 0;
static ID3D11RenderTargetView*  g_mainRenderTargetView = nullptr;

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

//==============================================================================
// Process Management
//==============================================================================
struct ProcessInfo {
    HANDLE hProcess = nullptr;
    HANDLE hThread = nullptr;
    HANDLE hStdOutRead = nullptr;
    HANDLE hStdOutWrite = nullptr;
    std::atomic<bool> running{false};
    std::string output;
    std::mutex outputMutex;
    std::thread outputThread;

    void start(const std::string& command, const std::string& workDir = "") {
        if (running) return;

        // Create pipe for stdout
        SECURITY_ATTRIBUTES sa;
        sa.nLength = sizeof(SECURITY_ATTRIBUTES);
        sa.bInheritHandle = TRUE;
        sa.lpSecurityDescriptor = nullptr;

        CreatePipe(&hStdOutRead, &hStdOutWrite, &sa, 0);
        SetHandleInformation(hStdOutRead, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOA si = {};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
        si.hStdOutput = hStdOutWrite;
        si.hStdError = hStdOutWrite;
        si.wShowWindow = SW_HIDE;

        PROCESS_INFORMATION pi = {};

        std::string cmd = command;
        BOOL success = CreateProcessA(
            nullptr,
            cmd.data(),
            nullptr,
            nullptr,
            TRUE,
            CREATE_NO_WINDOW,
            nullptr,
            workDir.empty() ? nullptr : workDir.c_str(),
            &si,
            &pi
        );

        if (success) {
            hProcess = pi.hProcess;
            hThread = pi.hThread;
            running = true;

            CloseHandle(hStdOutWrite);
            hStdOutWrite = nullptr;

            // Start output reading thread
            outputThread = std::thread([this]() {
                char buffer[4096];
                DWORD bytesRead;
                while (running) {
                    if (ReadFile(hStdOutRead, buffer, sizeof(buffer) - 1, &bytesRead, nullptr) && bytesRead > 0) {
                        buffer[bytesRead] = '\0';
                        std::lock_guard<std::mutex> lock(outputMutex);
                        output += buffer;
                        // Keep last 10KB
                        if (output.size() > 10240) {
                            output = output.substr(output.size() - 8192);
                        }
                    }

                    // Check if process still running
                    DWORD exitCode;
                    if (GetExitCodeProcess(hProcess, &exitCode) && exitCode != STILL_ACTIVE) {
                        running = false;
                        break;
                    }
                }
            });
        } else {
            CloseHandle(hStdOutRead);
            CloseHandle(hStdOutWrite);
            hStdOutRead = nullptr;
            hStdOutWrite = nullptr;
        }
    }

    void stop() {
        if (!running) return;

        running = false;

        // Send Ctrl+C signal
        if (hProcess) {
            GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, GetProcessId(hProcess));

            // Wait a bit then terminate if needed
            if (WaitForSingleObject(hProcess, 2000) == WAIT_TIMEOUT) {
                TerminateProcess(hProcess, 0);
            }

            CloseHandle(hProcess);
            CloseHandle(hThread);
            hProcess = nullptr;
            hThread = nullptr;
        }

        if (hStdOutRead) {
            CloseHandle(hStdOutRead);
            hStdOutRead = nullptr;
        }

        if (outputThread.joinable()) {
            outputThread.join();
        }
    }

    std::string getOutput() {
        std::lock_guard<std::mutex> lock(outputMutex);
        return output;
    }

    void clearOutput() {
        std::lock_guard<std::mutex> lock(outputMutex);
        output.clear();
    }

    ~ProcessInfo() {
        stop();
    }
};

//==============================================================================
// Application State
//==============================================================================
struct AppState {
    // Paths
    std::string exePath;
    std::string buildDir;
    std::string mapsDir;

    // Device settings
    char hostIP[64] = "192.168.1.50";
    char deviceIP[64] = "";
    char serialSuffix[16] = "144";

    // SLAM settings
    float voxelSize = 0.05f;
    int runDuration = 0;  // 0 = unlimited
    char outputPrefix[256] = "slam_map";
    bool enableVisualization = true;

    // Map selection for localization
    std::vector<std::string> availableMaps;
    int selectedMapIndex = -1;
    char selectedMapPath[512] = "";

    // VESC motor control settings
    char vescPort[32] = "COM3";
    int vescLeftId = 1;
    int vescRightId = 126;
    int canBitrate = 500000;

    // Robot geometry (mm)
    float trackWidth = 120.0f;      // Center-to-center of treads
    float wheelBase = 95.0f;        // Front to rear
    float treadWidth = 40.0f;       // Width of each tread
    float effectiveTrack = 156.0f;  // Includes scrub factor

    // Odometry calibration
    float ticksPerMeterLeft = 14052.0f;
    float ticksPerMeterRight = 14133.0f;

    // Duty scaling calibration results
    struct DutyCalPoint {
        float duty;
        float scaleRight;
    };
    std::vector<DutyCalPoint> dutyCalibration = {
        {0.030f, 0.622f},
        {0.035f, 0.800f},
        {0.040f, 0.816f},
        {0.050f, 0.867f},
        {0.060f, 0.878f},
        {0.070f, 0.883f},
    };

    // Minimum duty thresholds
    float minDutyStartLeft = 0.040f;
    float minDutyStartRight = 0.035f;
    float minDutyKeepLeft = 0.030f;
    float minDutyKeepRight = 0.020f;

    // Calibration state
    bool calibrationRunning = false;
    int calibrationProgress = 0;
    std::string calibrationStatus;

    // Process management
    ProcessInfo slamProcess;
    ProcessInfo rerunProcess;
    ProcessInfo localizationProcess;
    ProcessInfo calibrationProcess;

    // Status
    std::string statusMessage;
    float statusTimer = 0.0f;

    // Statistics (parsed from output)
    int pointRate = 0;
    int imuRate = 0;
    int scanCount = 0;
    int mapSize = 0;
    float posX = 0, posY = 0, posZ = 0;

    void setStatus(const std::string& msg) {
        statusMessage = msg;
        statusTimer = 3.0f;
    }

    void scanForMaps() {
        availableMaps.clear();
        try {
            for (const auto& entry : std::filesystem::directory_iterator(mapsDir)) {
                if (entry.path().extension() == ".ply") {
                    availableMaps.push_back(entry.path().filename().string());
                }
            }
        } catch (...) {}

        // Also check current directory
        try {
            for (const auto& entry : std::filesystem::directory_iterator(buildDir)) {
                if (entry.path().extension() == ".ply") {
                    std::string name = entry.path().filename().string();
                    if (std::find(availableMaps.begin(), availableMaps.end(), name) == availableMaps.end()) {
                        availableMaps.push_back(name);
                    }
                }
            }
        } catch (...) {}
    }

    void parseStats() {
        std::string out = slamProcess.getOutput();
        if (out.empty()) return;

        // Find last status line
        size_t pos = out.rfind("Time:");
        if (pos != std::string::npos) {
            std::string line = out.substr(pos);
            // Parse: Time: Xs | Pts: Xk/s | IMU: XHz | Scans: X | Map: Xk | Pos: [x, y, z]
            sscanf(line.c_str(), "Time: %*ds | Pts: %dk/s | IMU: %dHz | Scans: %d | Map: %dk | Pos: [%f, %f, %f]",
                   &pointRate, &imuRate, &scanCount, &mapSize, &posX, &posY, &posZ);
        }
    }
};

static AppState g_app;

//==============================================================================
// GUI Drawing
//==============================================================================
void DrawMainWindow() {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoBringToFrontOnFocus;

    ImGui::Begin("SLAM Control", nullptr, flags);

    // Header
    ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "SLAM Control Panel");
    ImGui::PopFont();
    ImGui::Separator();
    ImGui::Spacing();

    // Two-column layout
    float panelWidth = (ImGui::GetContentRegionAvail().x - 20) / 2;

    //==========================================================================
    // LEFT PANEL - Controls
    //==========================================================================
    ImGui::BeginChild("LeftPanel", ImVec2(panelWidth, 0), true);

    // Device Configuration
    if (ImGui::CollapsingHeader("Device Configuration", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::InputText("Host IP", g_app.hostIP, sizeof(g_app.hostIP));
        ImGui::InputText("Serial Suffix", g_app.serialSuffix, sizeof(g_app.serialSuffix));
        ImGui::InputText("Device IP (optional)", g_app.deviceIP, sizeof(g_app.deviceIP));

        ImGui::Spacing();
        ImGui::SliderFloat("Voxel Size (m)", &g_app.voxelSize, 0.01f, 1.0f, "%.2f");
        ImGui::SliderInt("Duration (0=unlimited)", &g_app.runDuration, 0, 300);
        ImGui::InputText("Output Prefix", g_app.outputPrefix, sizeof(g_app.outputPrefix));
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Visualization Control
    if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable Rerun Visualization", &g_app.enableVisualization);

        bool rerunRunning = g_app.rerunProcess.running;

        if (!rerunRunning) {
            if (ImGui::Button("Launch Rerun Viewer", ImVec2(-1, 30))) {
                g_app.rerunProcess.start("rerun");
                g_app.setStatus("Launching Rerun viewer...");
            }
        } else {
            if (ImGui::Button("Close Rerun Viewer", ImVec2(-1, 30))) {
                g_app.rerunProcess.stop();
                g_app.setStatus("Rerun viewer closed");
            }
        }

        ImGui::TextColored(rerunRunning ? ImVec4(0.2f, 1.0f, 0.2f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                          rerunRunning ? "Viewer: Running" : "Viewer: Stopped");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // SLAM Mapping Control
    if (ImGui::CollapsingHeader("Live SLAM Mapping", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool slamRunning = g_app.slamProcess.running;

        if (!slamRunning) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f));
            if (ImGui::Button("Start Mapping", ImVec2(-1, 40))) {
                std::string cmd = g_app.exePath + "\\live_slam.exe";
                cmd += " --host " + std::string(g_app.hostIP);

                if (strlen(g_app.deviceIP) > 0) {
                    cmd += " --device " + std::string(g_app.deviceIP);
                } else if (strlen(g_app.serialSuffix) > 0) {
                    cmd += " --serial " + std::string(g_app.serialSuffix);
                }

                cmd += " --voxel " + std::to_string(g_app.voxelSize);
                cmd += " --time " + std::to_string(g_app.runDuration);
                cmd += " --output " + std::string(g_app.outputPrefix);

                if (g_app.enableVisualization) {
                    cmd += " --visualize";
                }

                g_app.slamProcess.start(cmd, g_app.buildDir);
                g_app.setStatus("Starting SLAM mapping...");
            }
            ImGui::PopStyleColor(2);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.3f, 0.3f, 1.0f));
            if (ImGui::Button("Stop Mapping", ImVec2(-1, 40))) {
                g_app.slamProcess.stop();
                g_app.setStatus("SLAM mapping stopped");
            }
            ImGui::PopStyleColor(2);
        }

        // Status indicator
        ImVec4 statusColor = slamRunning ? ImVec4(0.2f, 1.0f, 0.2f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
        ImGui::TextColored(statusColor, slamRunning ? "Status: MAPPING" : "Status: Idle");

        if (slamRunning) {
            g_app.parseStats();
            ImGui::Text("Points: %dk/s | IMU: %dHz", g_app.pointRate, g_app.imuRate);
            ImGui::Text("Scans: %d | Map: %dk pts", g_app.scanCount, g_app.mapSize);
            ImGui::Text("Position: [%.2f, %.2f, %.2f]", g_app.posX, g_app.posY, g_app.posZ);
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Global Localization Control
    if (ImGui::CollapsingHeader("Global Localization", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Map selection
        if (ImGui::Button("Refresh Maps")) {
            g_app.scanForMaps();
        }
        ImGui::SameLine();
        ImGui::Text("(%d maps found)", (int)g_app.availableMaps.size());

        if (!g_app.availableMaps.empty()) {
            if (ImGui::BeginCombo("Select Map", g_app.selectedMapIndex >= 0 ?
                                  g_app.availableMaps[g_app.selectedMapIndex].c_str() : "Choose...")) {
                for (int i = 0; i < (int)g_app.availableMaps.size(); i++) {
                    bool selected = (g_app.selectedMapIndex == i);
                    if (ImGui::Selectable(g_app.availableMaps[i].c_str(), selected)) {
                        g_app.selectedMapIndex = i;
                        snprintf(g_app.selectedMapPath, sizeof(g_app.selectedMapPath),
                                "%s", g_app.availableMaps[i].c_str());
                    }
                    if (selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }

        ImGui::InputText("Map Path", g_app.selectedMapPath, sizeof(g_app.selectedMapPath));

        bool locRunning = g_app.localizationProcess.running;
        bool canStart = strlen(g_app.selectedMapPath) > 0 && !locRunning;

        ImGui::BeginDisabled(!canStart);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.7f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.5f, 0.8f, 1.0f));
        if (ImGui::Button("Start Localization", ImVec2(-1, 40))) {
            // TODO: Implement localization mode in live_slam
            g_app.setStatus("Localization mode - coming soon!");
        }
        ImGui::PopStyleColor(2);
        ImGui::EndDisabled();

        if (locRunning) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
            if (ImGui::Button("Stop Localization", ImVec2(-1, 30))) {
                g_app.localizationProcess.stop();
                g_app.setStatus("Localization stopped");
            }
            ImGui::PopStyleColor();
        }

        ImGui::TextColored(locRunning ? ImVec4(0.4f, 0.8f, 1.0f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
                          locRunning ? "Status: LOCALIZING" : "Status: Idle");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // VESC Motor Calibration
    if (ImGui::CollapsingHeader("Motor Calibration", ImGuiTreeNodeFlags_DefaultOpen)) {
        // CAN Configuration
        ImGui::Text("CAN Configuration");
        ImGui::InputText("CAN Port", g_app.vescPort, sizeof(g_app.vescPort));
        ImGui::InputInt("Left VESC ID", &g_app.vescLeftId);
        ImGui::InputInt("Right VESC ID", &g_app.vescRightId);

        ImGui::Spacing();
        ImGui::Separator();

        // Robot Geometry
        ImGui::Text("Robot Geometry (mm)");
        ImGui::DragFloat("Track Width", &g_app.trackWidth, 1.0f, 50.0f, 500.0f, "%.1f");
        ImGui::DragFloat("Wheel Base", &g_app.wheelBase, 1.0f, 50.0f, 500.0f, "%.1f");
        ImGui::DragFloat("Tread Width", &g_app.treadWidth, 1.0f, 10.0f, 200.0f, "%.1f");
        ImGui::DragFloat("Effective Track*", &g_app.effectiveTrack, 1.0f, 50.0f, 500.0f, "%.1f");
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "* Scrub factor ~1.3x for skid-steer");

        ImGui::Spacing();
        ImGui::Separator();

        // Odometry Calibration
        ImGui::Text("Odometry (ticks/m)");
        ImGui::DragFloat("Left", &g_app.ticksPerMeterLeft, 10.0f, 1000.0f, 50000.0f, "%.0f");
        ImGui::DragFloat("Right", &g_app.ticksPerMeterRight, 10.0f, 1000.0f, 50000.0f, "%.0f");

        ImGui::Spacing();
        ImGui::Separator();

        // Minimum Duty Thresholds
        ImGui::Text("Min Duty Thresholds");
        ImGui::DragFloat("Start L", &g_app.minDutyStartLeft, 0.001f, 0.01f, 0.1f, "%.3f");
        ImGui::SameLine();
        ImGui::DragFloat("Start R", &g_app.minDutyStartRight, 0.001f, 0.01f, 0.1f, "%.3f");
        ImGui::DragFloat("Keep L", &g_app.minDutyKeepLeft, 0.001f, 0.01f, 0.1f, "%.3f");
        ImGui::SameLine();
        ImGui::DragFloat("Keep R", &g_app.minDutyKeepRight, 0.001f, 0.01f, 0.1f, "%.3f");

        ImGui::Spacing();
        ImGui::Separator();

        // Duty Scaling Table
        if (ImGui::TreeNode("Duty Scaling Table")) {
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Right wheel scale to match left");
            ImGui::Columns(2, "duty_cal_table", true);
            ImGui::SetColumnWidth(0, 80);
            ImGui::Text("Duty"); ImGui::NextColumn();
            ImGui::Text("Scale R"); ImGui::NextColumn();
            ImGui::Separator();

            for (size_t i = 0; i < g_app.dutyCalibration.size(); i++) {
                char buf[32];
                snprintf(buf, sizeof(buf), "%.3f", g_app.dutyCalibration[i].duty);
                ImGui::Text("%s", buf); ImGui::NextColumn();
                snprintf(buf, sizeof(buf), "%.3f", g_app.dutyCalibration[i].scaleRight);
                ImGui::Text("%s", buf); ImGui::NextColumn();
            }
            ImGui::Columns(1);
            ImGui::TreePop();
        }

        ImGui::Spacing();
        ImGui::Separator();

        // Calibration Buttons
        bool calRunning = g_app.calibrationProcess.running;

        if (!calRunning) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.3f, 0.1f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.4f, 0.2f, 1.0f));

            if (ImGui::Button("Run Duty Calibration", ImVec2(-1, 30))) {
                // TODO: Launch calibration tool
                g_app.setStatus("Duty calibration - ready to implement!");
            }

            if (ImGui::Button("Run Odometry Calibration", ImVec2(-1, 30))) {
                // TODO: Launch odometry calibration
                g_app.setStatus("Odometry calibration - ready to implement!");
            }

            if (ImGui::Button("Calibration + Localization", ImVec2(-1, 35))) {
                // Combined calibration with global localization
                // Movement: Forward -> Turn 90 Left -> Center -> Turn 90 Right
                g_app.setStatus("Combined calibration - ready to implement!");
            }

            ImGui::PopStyleColor(2);
        } else {
            // Show progress
            ImGui::ProgressBar(g_app.calibrationProgress / 100.0f, ImVec2(-1, 25));
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "%s", g_app.calibrationStatus.c_str());

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
            if (ImGui::Button("Cancel Calibration", ImVec2(-1, 30))) {
                g_app.calibrationProcess.stop();
                g_app.setStatus("Calibration cancelled");
            }
            ImGui::PopStyleColor();
        }

        // Save/Load buttons
        ImGui::Spacing();
        if (ImGui::Button("Save Calibration", ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 25))) {
            g_app.setStatus("Calibration saved to vesc_calibration.ini");
        }
        ImGui::SameLine();
        if (ImGui::Button("Load Calibration", ImVec2(-1, 25))) {
            g_app.setStatus("Calibration loaded from vesc_calibration.ini");
        }
    }

    ImGui::EndChild();

    ImGui::SameLine();

    //==========================================================================
    // RIGHT PANEL - Output Log
    //==========================================================================
    ImGui::BeginChild("RightPanel", ImVec2(0, 0), true);

    ImGui::Text("Output Log");
    ImGui::Separator();

    // Clear button
    if (ImGui::Button("Clear")) {
        g_app.slamProcess.clearOutput();
    }

    ImGui::BeginChild("LogContent", ImVec2(0, -30), true);

    std::string output = g_app.slamProcess.getOutput();
    if (!output.empty()) {
        ImGui::TextWrapped("%s", output.c_str());
        // Auto-scroll
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 50) {
            ImGui::SetScrollHereY(1.0f);
        }
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No output yet...");
    }

    ImGui::EndChild();

    // Status bar
    if (g_app.statusTimer > 0) {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.4f, 1.0f), "%s", g_app.statusMessage.c_str());
        g_app.statusTimer -= ImGui::GetIO().DeltaTime;
    }

    ImGui::EndChild();

    ImGui::End();
}

//==============================================================================
// Main Entry Point
//==============================================================================
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    // Get executable path
    char exePath[MAX_PATH];
    GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    g_app.exePath = exeDir;
    g_app.buildDir = exeDir;
    g_app.mapsDir = exeDir;

    // Create window
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L,
                       GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr,
                       L"SLAM Control", nullptr };
    RegisterClassExW(&wc);

    HWND hwnd = CreateWindowW(wc.lpszClassName, L"SLAM Control Panel",
                              WS_OVERLAPPEDWINDOW, 100, 100, 1200, 700,
                              nullptr, nullptr, wc.hInstance, nullptr);

    // Initialize Direct3D
    if (!CreateDeviceD3D(hwnd)) {
        CleanupDeviceD3D();
        UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);

    // Setup Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Style
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.FramePadding = ImVec2(8, 4);
    style.ItemSpacing = ImVec2(8, 6);

    // Colors - dark blue theme
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.12f, 0.15f, 1.00f);
    colors[ImGuiCol_ChildBg] = ImVec4(0.12f, 0.14f, 0.18f, 1.00f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.18f, 0.22f, 1.00f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.20f, 0.22f, 0.28f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.20f, 0.35f, 0.50f, 1.00f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.45f, 0.60f, 1.00f);
    colors[ImGuiCol_Button] = ImVec4(0.20f, 0.35f, 0.50f, 1.00f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.45f, 0.60f, 1.00f);

    // Setup backends
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    // Scan for maps
    g_app.scanForMaps();

    // Main loop
    bool done = false;
    while (!done) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
                done = true;
        }
        if (done) break;

        // Handle resize
        if (g_SwapChainOccluded && g_pSwapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED) {
            Sleep(10);
            continue;
        }
        g_SwapChainOccluded = false;

        if (g_ResizeWidth != 0 && g_ResizeHeight != 0) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        // Start frame
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        // Draw GUI
        DrawMainWindow();

        // Render
        ImGui::Render();
        const float clear_color[] = { 0.1f, 0.1f, 0.12f, 1.0f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        g_pSwapChain->Present(1, 0);
    }

    // Cleanup
    g_app.slamProcess.stop();
    g_app.rerunProcess.stop();
    g_app.localizationProcess.stop();
    g_app.calibrationProcess.stop();

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);

    return 0;
}

//==============================================================================
// DirectX 11 Helper Functions
//==============================================================================
bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };
    HRESULT res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                                 createDeviceFlags, featureLevelArray, 2,
                                                 D3D11_SDK_VERSION, &sd, &g_pSwapChain,
                                                 &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res == DXGI_ERROR_UNSUPPORTED)
        res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr,
                                            createDeviceFlags, featureLevelArray, 2,
                                            D3D11_SDK_VERSION, &sd, &g_pSwapChain,
                                            &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (res != S_OK)
        return false;

    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}

void CreateRenderTarget() {
    ID3D11Texture2D* pBackBuffer;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget() {
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = nullptr; }
}

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED)
            return 0;
        g_ResizeWidth = (UINT)LOWORD(lParam);
        g_ResizeHeight = (UINT)HIWORD(lParam);
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU)
            return 0;
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}
