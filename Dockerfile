# Base image with CUDA support
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Set environment variable to noninteractive to avoid timezone prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    git \
    wget \
    software-properties-common \
    curl \
    libgdal-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Add Miniconda to the PATH for the Docker build process
ENV PATH="/opt/conda/bin:$PATH"

# Create a new conda environment with Python 3.10 and install GDAL
RUN conda create -n myenv python=3.10 gdal=3.7.1 -c conda-forge && \
    conda clean -afy

# Set working directory
WORKDIR /workspace

# Copy your code into the container
COPY . /workspace

# Activate the conda environment and run a bash shell
CMD ["bash", "-c", "source activate myenv && exec bash"]