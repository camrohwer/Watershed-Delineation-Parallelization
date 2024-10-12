# Watershed Delineation Project

This project implements watershed delineation algorithms using both iterative and parallel approaches using C++ and CUDA respectively. The goal is to efficiently process Digital Elevation Models (DEMs) to perform watershed Delineation.

## Table of Contents

- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [CUDA Setup](#cuda-setup)
- [Building the Project](#building-the-project)
- [Running the Algorithms](#running-the-algorithms)
- [Scripting](#scripting)
- [License](#license)

## Requirements

- CUDA-capable GPU
- Docker (with NVIDIA runtime)
- CMake
- Python (for runtime comparison)

## Project Structure

my_project/  
├── Dockerfile  
├── src/  
│   ├── flow_direction_iterative.cpp  
│   ├── flow_direction_parallel.cu  
├── build/  
├── DEMs/  
│   ├── Output/  
├── CMakeLists.txt  
└── README.md  

- `Dockerfile`: Instructions for building the Docker image.
- `src/`: Contains the source code for the iterative and parallel implementations.
- `build/`: Output directory for the build files.
- `DEMs/`: Directory for input and output raster files.
- `CMakeLists.txt`: Configuration file for CMake.
- `README.md`: This documentation file.

## CUDA Setup

Ensure you have up to date NVIDIA drivers and the correct CUDA version installed. You can check the installed version using:

```bash
nvidia-smi
```

Make sure your Docker image matches the CUDA version you have installed by checking the NVIDIA Dockeer images documentation at:
https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md

If you need to configure the NVIDIA runtime for Docker, on your host device run:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

## Building the Project

1. Build the Docker image:
```bash
docker build -t watershed_delineation .  
```

2. Run the Docker container with GPU support:
```bash
docker run --gpus all -it watershed_delineation  
```
note: There is an included devcontainer.json file if you would like to open the container with GPU support enabled in VSCode

3. Inside the container, navigate to the project directory and create a build directory:
```bash
cd /path/to/project
mkdir build
cd build
```

4. Use CMake to configure the project:
```bash
cmake ..
```

5. Build the target for the iterative algorithm:
```bash
cmake --build . --target flow_direction_iterative  
```

6. Build the target for the parallel algorithm:
```bash
cmake --build . --target flow_direction_parallel  
```

## Running the Algorithms

You can run the alorithms using the following commands:

### Iterative Flow Direction

```bash
./flow_direction_iterative /relative/path/to/input/DEM
```

### Parallel Flow Direction

```bash
./flow_direction_parallel /relative/path/to/input/DEM 
```

## Scripting

Various scripts are included for calculating runtimes, checking outputs, comparing GPU Warp size, etc...

### Python

These can be run from the main project directory using:

```bash
python scripts/script.py 
```

## License

Distributed under the MIT License.
