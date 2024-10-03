# Base image with CUDA support
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    git \
    wget \
    software-properties-common \
    curl \
    libgdal-dev \
    gdal-bin

# Set up GDAL environment
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install Conda for GDAL
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p $HOME/miniconda && \
    rm ~/miniconda.sh && \
    $HOME/miniconda/bin/conda init

# Add Miniconda to the path
ENV PATH=/root/miniconda/bin:$PATH

# Install GDAL via Conda for compatibility
RUN conda install -c conda-forge gdal=3.7.1

# Install GDAL utilities if necessary
RUN apt-get install -y gdal-bin python3-gdal

# Set working directory
WORKDIR /workspace

# Copy your code into the container
COPY . /workspace

# Compile C++ code with GDAL
RUN mkdir build && cd build && cmake .. && make

# Compile CUDA code
RUN nvcc -o flow_direction_parallel src/flow_direction_parallel.cu -lgdal

# Default command
CMD ["/bin/bash"]
