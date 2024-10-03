#include <iostream>
#include <gdal_priv.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void flowDirectionKernel(double* dem, int* flow_dir, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    if (x >= width || y >= height) return;

    int directions[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {1, 1}, {-1, 1}, {1, -1}};
    int best_dir = -1;
    double max_slope = -1;

    for (int i = 0; i < 8; ++i) {
        int nx = x + directions[i][0];
        int ny = y + directions[i][1];
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            double dz = dem[index] - dem[ny * width + nx];
            double slope = dz / sqrt(pow(directions[i][0], 2) + pow(directions[i][1], 2));
            if (slope > max_slope) {
                max_slope = slope;
                best_dir = i;
            }
        }
    }
    flow_dir[index] = best_dir;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <DEM file path>" << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize GDAL
    GDALAllRegister();
    
    // Open the DEM file
    GDALDataset* poDataset = (GDALDataset*)GDALOpen(argv[1], GA_ReadOnly);
    if (poDataset == nullptr) {
        std::cerr << "Error opening DEM file." << std::endl;
        return EXIT_FAILURE;
    }

    // Get raster size
    int width = poDataset->GetRasterXSize();
    int height = poDataset->GetRasterYSize();
    
    // Allocate memory for DEM data
    double* h_dem = new double[width * height]; // DEM data
    int* h_flow_dir = new int[width * height]; // Flow direction data

    // Read DEM data
    GDALRasterBand* poBand = poDataset->GetRasterBand(1);
    poBand->RasterIO(GF_Read, 0, 0, width, height, h_dem, width, height, GDT_Float64, 0, 0);

    // Allocate memory on GPU
    double* d_dem;
    int* d_flow_dir;
    cudaMalloc(&d_dem, width * height * sizeof(double));
    cudaMalloc(&d_flow_dir, width * height * sizeof(int));

    // Transfer DEM data to GPU
    cudaMemcpy(d_dem, h_dem, width * height * sizeof(double), cudaMemcpyHostToDevice);

    // Define grid and block size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    flowDirectionKernel<<<numBlocks, threadsPerBlock>>>(d_dem, d_flow_dir, width, height);

    // Transfer results back to host
    cudaMemcpy(h_flow_dir, d_flow_dir, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_dem);
    cudaFree(d_flow_dir);

    // Clean up
    delete[] h_dem;
    delete[] h_flow_dir;
    GDALClose(poDataset);

    return 0;
}
