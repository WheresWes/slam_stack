cmake_minimum_required(VERSION 3.16)
project(slam_stack VERSION 1.0.0 LANGUAGES CXX)

#==============================================================================
# Build Configuration
#==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags (platform-specific)
if(MSVC)
    set(CMAKE_CXX_FLAGS "/W3 /EHsc")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    add_definitions(-D_USE_MATH_DEFINES)  # For M_PI
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_VISUALIZER "Build visualizer (requires Pangolin)" OFF)
option(BUILD_WITH_LIVOX_SDK "Build with Livox SDK2 support" ON)
option(BUILD_EXAMPLES "Build example applications" ON)
option(BUILD_WITH_RERUN "Build with Rerun visualization" ON)
option(BUILD_GUI "Build SLAM Control GUI" ON)

#==============================================================================
# Dependencies
#==============================================================================

# Eigen3 (required)
find_package(Eigen3 3.3 REQUIRED)
message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")

# OpenMP (for parallel processing)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
endif()

# Livox SDK2 (optional, for hardware interface)
if(BUILD_WITH_LIVOX_SDK)
    find_package(LivoxSDK2 QUIET)
    if(LivoxSDK2_FOUND)
        message(STATUS "Found Livox SDK2")
    else()
        message(STATUS "Livox SDK2 not found - hardware interface disabled")
        set(BUILD_WITH_LIVOX_SDK OFF)
    endif()
endif()

# Pangolin (optional, for visualizer)
if(BUILD_VISUALIZER)
    find_package(Pangolin QUIET)
    if(Pangolin_FOUND)
        message(STATUS "Found Pangolin")
    else()
        message(STATUS "Pangolin not found - visualizer disabled")
        set(BUILD_VISUALIZER OFF)
    endif()
endif()

# yaml-cpp (for configuration)
find_package(yaml-cpp QUIET)
if(yaml-cpp_FOUND)
    message(STATUS "Found yaml-cpp")
endif()

# nanoflann (header-only kd-tree, for fast nearest neighbor)
find_package(nanoflann CONFIG QUIET)
if(nanoflann_FOUND)
    message(STATUS "Found nanoflann - fast kd-tree enabled")
    set(HAS_NANOFLANN TRUE)
else()
    message(STATUS "nanoflann not found - using brute-force NN")
    set(HAS_NANOFLANN FALSE)
endif()

# Rerun SDK (optional, for real-time visualization)
if(BUILD_WITH_RERUN)
    find_package(rerun_sdk CONFIG QUIET)
    if(rerun_sdk_FOUND)
        message(STATUS "Found Rerun SDK - visualization enabled")
        set(HAS_RERUN TRUE)
    else()
        message(STATUS "Rerun SDK not found - visualization disabled")
        set(HAS_RERUN FALSE)
        set(BUILD_WITH_RERUN OFF)
    endif()
endif()

# PCL (optional, for point cloud processing and ikd-tree)
find_package(PCL 1.8 QUIET COMPONENTS common filters io)
if(PCL_FOUND)
    message(STATUS "Found PCL: ${PCL_VERSION}")
    include_directories(${PCL_INCLUDE_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    set(HAS_PCL TRUE)
else()
    message(STATUS "PCL not found - ikd-tree features disabled")
    set(HAS_PCL FALSE)
endif()

# Dear ImGui (optional, for GUI)
if(BUILD_GUI)
    find_package(imgui CONFIG QUIET)
    if(imgui_FOUND)
        message(STATUS "Found Dear ImGui - GUI enabled")
        set(HAS_IMGUI TRUE)
    else()
        message(STATUS "Dear ImGui not found - GUI disabled")
        set(HAS_IMGUI FALSE)
        set(BUILD_GUI OFF)
    endif()
endif()

#==============================================================================
# Include Directories
#==============================================================================

include_directories(
    ${PROJECT_SOURCE_DIR}/common/include
    ${PROJECT_SOURCE_DIR}/slam_engine/include
    ${EIGEN3_INCLUDE_DIR}
)

#==============================================================================
# Common Library
#==============================================================================

add_library(slam_common INTERFACE)
target_include_directories(slam_common INTERFACE
    ${PROJECT_SOURCE_DIR}/common/include
)

#==============================================================================
# SLAM Engine Library
#==============================================================================

if(HAS_PCL)
    # ikd-Tree source (needed for SLAM engine with PCL)
    set(SLAM_ENGINE_SRC
        ${PROJECT_SOURCE_DIR}/slam_engine/include/slam/ikd-Tree/ikd_Tree.cpp
    )

    # Visualization source (separated for faster incremental builds)
    if(HAS_RERUN)
        list(APPEND SLAM_ENGINE_SRC
            ${PROJECT_SOURCE_DIR}/slam_engine/src/visualization.cpp
        )
    endif()

    # SLAM engine library with PCL support
    add_library(slam_engine ${SLAM_ENGINE_SRC})
    target_include_directories(slam_engine PUBLIC
        ${PROJECT_SOURCE_DIR}/slam_engine/include
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
    )
    target_link_libraries(slam_engine PUBLIC
        slam_common
        Eigen3::Eigen
        ${PCL_LIBRARIES}
    )
    target_compile_definitions(slam_engine PUBLIC HAS_PCL)
else()
    # Header-only SLAM engine without PCL (ICP only)
    add_library(slam_engine INTERFACE)
    target_include_directories(slam_engine INTERFACE
        ${PROJECT_SOURCE_DIR}/slam_engine/include
        ${EIGEN3_INCLUDE_DIR}
    )
    target_link_libraries(slam_engine INTERFACE
        slam_common
        Eigen3::Eigen
    )
endif()

if(OpenMP_CXX_FOUND)
    if(HAS_PCL)
        target_link_libraries(slam_engine PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(slam_engine PUBLIC MP_EN)
    else()
        target_link_libraries(slam_engine INTERFACE OpenMP::OpenMP_CXX)
        target_compile_definitions(slam_engine INTERFACE MP_EN)
    endif()
endif()

if(HAS_NANOFLANN)
    if(HAS_PCL)
        target_link_libraries(slam_engine PUBLIC nanoflann::nanoflann)
        target_compile_definitions(slam_engine PUBLIC HAS_NANOFLANN)
    else()
        target_link_libraries(slam_engine INTERFACE nanoflann::nanoflann)
        target_compile_definitions(slam_engine INTERFACE HAS_NANOFLANN)
    endif()
endif()

# Rerun visualization support (linked to slam_engine for PIMPL pattern)
if(HAS_RERUN AND HAS_PCL)
    target_link_libraries(slam_engine PUBLIC rerun_sdk)
    target_compile_definitions(slam_engine PUBLIC HAS_RERUN)
endif()

#==============================================================================
# Examples
#==============================================================================

if(BUILD_EXAMPLES)
    # Example: Test gravity alignment
    add_executable(test_gravity_alignment
        examples/test_gravity_alignment.cpp
    )
    target_link_libraries(test_gravity_alignment
        slam_engine
    )

    # Example: Test PLY export
    add_executable(test_ply_export
        examples/test_ply_export.cpp
    )
    target_link_libraries(test_ply_export
        slam_common
    )

    # Example: Test preprocessor
    add_executable(test_preprocessor
        examples/test_preprocessor.cpp
    )
    target_link_libraries(test_preprocessor
        slam_engine
    )

    # Example: Test localization (ICP)
    add_executable(test_localization
        examples/test_localization.cpp
    )
    target_link_libraries(test_localization
        slam_engine
    )

    # Comprehensive ICP test suite
    add_executable(test_icp_suite
        examples/test_icp_suite.cpp
    )
    target_link_libraries(test_icp_suite
        slam_engine
    )

    # Real-world ICP test (Stanford Bunny)
    add_executable(test_real_icp
        examples/test_real_icp.cpp
    )
    target_link_libraries(test_real_icp
        slam_engine
    )

    # Global localization test
    add_executable(test_global_localization
        examples/test_global_localization.cpp
    )
    target_link_libraries(test_global_localization
        slam_engine
    )

    # Global localization benchmark
    add_executable(benchmark_global_localization
        examples/benchmark_global_localization.cpp
    )
    target_link_libraries(benchmark_global_localization
        slam_engine
    )

    # Relocalization test (coarse-to-fine ICP with real data)
    add_executable(test_relocalization
        examples/test_relocalization.cpp
    )
    target_link_libraries(test_relocalization
        slam_engine
    )

    # Global relocalization test (exhaustive search for large environments)
    add_executable(test_global_relocalization
        examples/test_global_relocalization.cpp
    )
    target_link_libraries(test_global_relocalization
        slam_engine
    )

    # Localization mode test (FAST-LIO with pre-built map)
    add_executable(test_localization_mode
        examples/test_localization_mode.cpp
    )
    target_link_libraries(test_localization_mode
        slam_engine
    )

    # Progressive localization test (coverage-based, uses FAST-LIO for accumulation)
    add_executable(test_progressive_localization
        examples/test_progressive_localization.cpp
    )
    target_link_libraries(test_progressive_localization
        slam_engine
    )

    # Livox Mid-360 connection test
    add_executable(test_livox_connection
        examples/test_livox_connection.cpp
    )
    target_link_libraries(test_livox_connection
        slam_engine
    )
    if(WIN32)
        target_link_libraries(test_livox_connection ws2_32)
    endif()

    # Live SLAM with native Livox driver
    add_executable(live_slam
        examples/live_slam.cpp
    )
    target_link_libraries(live_slam
        slam_engine
    )
    if(WIN32)
        target_link_libraries(live_slam ws2_32)
    endif()
    if(HAS_RERUN)
        target_link_libraries(live_slam rerun_sdk)
        target_compile_definitions(live_slam PRIVATE HAS_RERUN)
    endif()

    # Replay SLAM (for offline testing with recorded data)
    add_executable(replay_slam
        examples/replay_slam.cpp
    )
    target_link_libraries(replay_slam
        slam_engine
    )
    if(HAS_RERUN)
        target_link_libraries(replay_slam rerun_sdk)
        target_compile_definitions(replay_slam PRIVATE HAS_RERUN)
    endif()

    # Recording analyzer (diagnose tracking issues)
    add_executable(analyze_recording
        tools/analyze_recording.cpp
    )
    target_include_directories(analyze_recording PRIVATE
        ${EIGEN3_INCLUDE_DIR}
    )

    # Simple SLAM test (no visualization)
    add_executable(test_replay_simple
        examples/test_replay_simple.cpp
    )
    target_link_libraries(test_replay_simple
        slam_engine
    )

    # Minimal test to isolate stack overflow
    add_executable(test_minimal
        examples/test_minimal.cpp
        ${PROJECT_SOURCE_DIR}/slam_engine/include/slam/ikd-Tree/ikd_Tree.cpp
    )
    target_include_directories(test_minimal PRIVATE
        ${PROJECT_SOURCE_DIR}/slam_engine/include
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
    )
    target_link_libraries(test_minimal
        ${PCL_LIBRARIES}
    )
    target_compile_definitions(test_minimal PRIVATE HAS_PCL)

    # Main SLAM application
    add_executable(slam_main
        examples/slam_main.cpp
    )
    target_link_libraries(slam_main
        slam_engine
    )

    # Link Livox SDK2 if available
    if(BUILD_WITH_LIVOX_SDK)
        target_link_libraries(slam_main
            ${LivoxSDK2_LIBRARIES}
        )
        target_include_directories(slam_main PRIVATE
            ${LivoxSDK2_INCLUDE_DIRS}
        )
        target_compile_definitions(slam_main PRIVATE LIVOX_SDK2_ENABLED)
    endif()
endif()

#==============================================================================
# GUI Application
#==============================================================================

if(HAS_IMGUI AND WIN32)
    add_executable(slam_control_gui WIN32
        gui/slam_control_gui.cpp
    )
    target_link_libraries(slam_control_gui
        imgui::imgui
        d3d11
    )
    target_compile_definitions(slam_control_gui PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
    message(STATUS "Building SLAM Control GUI")
endif()

#==============================================================================
# Installation
#==============================================================================

install(DIRECTORY common/include/slam
    DESTINATION include
)

install(DIRECTORY slam_engine/include/slam
    DESTINATION include
)

install(DIRECTORY config
    DESTINATION share/${PROJECT_NAME}
)

#==============================================================================
# Package Configuration
#==============================================================================

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/slam_stackConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/slam_stackConfig.cmake
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/slam_stackConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

#==============================================================================
# Summary
#==============================================================================

message(STATUS "")
message(STATUS "=== SLAM Stack Configuration ===")
message(STATUS "  Version:         ${PROJECT_VERSION}")
message(STATUS "  Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "  Livox SDK2:      ${BUILD_WITH_LIVOX_SDK}")
message(STATUS "  Visualizer:      ${BUILD_VISUALIZER}")
message(STATUS "  Rerun:           ${HAS_RERUN}")
message(STATUS "  ImGui GUI:       ${HAS_IMGUI}")
message(STATUS "  Tests:           ${BUILD_TESTS}")
message(STATUS "  Examples:        ${BUILD_EXAMPLES}")
message(STATUS "================================")
message(STATUS "")
