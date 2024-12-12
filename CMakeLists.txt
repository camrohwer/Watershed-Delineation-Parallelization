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

# Create an executable for the parallel (CUDA) version
add_executable(flow_dir src/flow_direction.cu)

# Link the GDAL library to the CUDA parallel executable
target_link_libraries(flow_dir PRIVATE ${GDAL_LIBRARIES} ${CUDA_LIBRARIES})

# Run the GPU architecture detection program and capture the output
add_executable(flow_accum src/flow_accum.cu)

target_link_libraries(flow_accum PRIVATE ${GDAL_LIBRARIES} ${CUDA_LIBRARIES})

add_executable(stream_ident src/stream_identification.cu)

target_link_libraries(stream_ident PRIVATE ${GDAL_LIBRARIES} ${CUDA_LIBRARIES})

add_executable(watershed_delin src/watershed_delineation.cu)

target_link_libraries(watershed_delin PRIVATE ${GDAL_LIBRARIES} ${CUDA_LIBRARIES})

# Run the GPU architecture detection program and capture the output
execute_process(
    COMMAND ${CMAKE_CUDA_COMPILER} ${CMAKE_SOURCE_DIR}/src/detect_gpu_arch.cu -o detect_gpu_arch
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

execute_process(
    COMMAND ${CMAKE_BINARY_DIR}/detect_gpu_arch
    OUTPUT_VARIABLE GPU_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Check if the GPU_ARCH was detected correctly
if(NOT GPU_ARCH)
    message(FATAL_ERROR "Failed to detect GPU architecture")
endif()

message(STATUS "Detected GPU architecture: compute_${GPU_ARCH}")

# Use the detected architecture in the compile options
foreach(cuda_target flow_dir flow_accum stream_ident watershed_delin)
    target_compile_options(${cuda_target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH}>
        $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    )
endforeach()