# Minimum CMake version requirement
cmake_minimum_required(VERSION 3.10)

# Project name and language settings
project(MyProject LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the CUDA standard (if required)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Enable GDAL (assuming GDAL is installed in the system)
find_package(GDAL REQUIRED)

# Output directories for compiled binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add include directories for GDAL
include_directories(${GDAL_INCLUDE_DIRS})

# Define the source files (C++ and CUDA)
set(SOURCES
    src/flow_direction_iterative.cpp
    src/flow_direction_parallel.cu
)

# Define the executable and link with GDAL
add_executable(MyProject ${SOURCES})

# Link GDAL library to your project
target_link_libraries(MyProject ${GDAL_LIBRARIES})
