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

# Define the source files for C++ (iterative version)
set(ITERATIVE_SOURCES
    src/flow_direction_iterative.cpp
)

# Define the source files for CUDA (parallel version)
set(PARALLEL_SOURCES
    src/flow_direction_parallel.cu
)

# Create an executable for the iterative (C++) version
add_executable(flow_direction_iterative ${ITERATIVE_SOURCES})

# Link the GDAL library to the C++ iterative executable
target_link_libraries(flow_direction_iterative PRIVATE ${GDAL_LIBRARIES})

# Create an executable for the parallel (CUDA) version
add_executable(flow_direction_parallel ${PARALLEL_SOURCES})

# Link the GDAL library to the CUDA parallel executable
target_link_libraries(flow_direction_parallel PRIVATE ${GDAL_LIBRARIES} ${CUDA_LIBRARIES})

# Set CUDA architecture flags for the parallel executable
#THIS IS FOR MY OLD GPU GTX780 ARCHITECTURE
target_compile_options(flow_direction_parallel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_35,code=sm_35>
)