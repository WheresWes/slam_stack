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

# PCL (optional, for point cloud processing and ikd-tree)
find_package(PCL 1.8 QUIET COMPONENTS common filters)
if(PCL_FOUND)
    message(STATUS "Found PCL: ${PCL_VERSION}")
    include_directories(${PCL_INCLUDE_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    set(HAS_PCL TRUE)
else()
    message(STATUS "PCL not found - ikd-tree features disabled")
    set(HAS_PCL FALSE)
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
    set(IKDTREE_SRC
        ${PROJECT_SOURCE_DIR}/slam_engine/include/slam/ikd-Tree/ikd_Tree.cpp
    )

    # SLAM engine library with PCL support
    add_library(slam_engine ${IKDTREE_SRC})
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
message(STATUS "  Tests:           ${BUILD_TESTS}")
message(STATUS "  Examples:        ${BUILD_EXAMPLES}")
message(STATUS "================================")
message(STATUS "")
