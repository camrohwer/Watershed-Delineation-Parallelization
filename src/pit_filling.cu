#include <iostream>
#include <gdal_priv.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <vector>
#include <limits.h>

struct PitCell {
    float elevation;
    int index;
};

__global__ void identifyPits(const float* dem, PitCell* pitCells, int* numPits, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 1 || x >= width || y <= 1 || y >= height) return; // skip boundary 

    int idx = y * width + x;
    float curElev = dem[idx];

    bool isPit = true;

    for (int dy = -1; dy <= 1; dy++){
        for (int dx = -1; dx <= 1; dx++){
            if (dx == 0 && dy == 0) continue; //skip current pixel

            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height){
                //get neightbour index
                int ni = ny * width + nx;
                if (dem[ni] <= curElev){
                    isPit = false;
                    break;
                }
            }
        }
        if (!isPit) break;
    }

    // if pit, store elev
    if (isPit){
        int count = atomicAdd(numPits, 1);
        pitCells[count].elevation = curElev;
        pitCells[count].index = idx;
    }
}
int main(int argc, char* argv[]){
    if (argc < 2){
        std::cout << "Please provide a filepath for input" << std::endl;
        return -1;
    }

    // register drivers to open raster data
    GDALAllRegister();
    
    // Open DEM dataset
    const char* input = argv[1];
    GDALDataset* demDataset  = (GDALDataset*) GDALOpen(input, GA_ReadOnly);

    if (demDataset == NULL) {
        std::cerr << "Error opening DEM file." << std::endl;
        return -1;
    }

    //Raster size to use with Malloc and device mem
    int width = demDataset->GetRasterXSize();
    int height = demDataset->GetRasterYSize();
    float *demData = (float *)CPLMalloc(sizeof(float) * width * height);

    //populate demData dynamically allocated memory
    CPLErr err = demDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, demData, width, height, GDT_Float32, 0, 0);
    if (err != CE_None){
        std::cerr << "Error reading DEM data: " << CPLGetLastErrorMsg() << std::endl;
        GDALClose(demDataset);
        return -1;
    }

    //allocating device mem
    float *d_dem;
    PitCell *d_pitCells;

    if (cudaMalloc(&d_dem, sizeof(float) * width * height) != cudaSuccess) {
        std::cerr << "Error allocating memory for DEM on device." << std::endl;
        return -1;
    }

    if (cudaMalloc(&d_pitCells, sizeof(PitCell) * width * height) != cudaSuccess) {
        std::cerr << "Error allocating memory for pit cells on device." << std::endl;
        cudaFree(d_dem); // Free already allocated memory
        return -1;
    }

    //copy DEM data to device
    cudaError_t memcpy_err = cudaMemcpy(d_dem, demData, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    if (memcpy_err != cudaSuccess){
        std::cerr << "Error copying data to device: " << cudaGetErrorString(memcpy_err) << std::endl;
        return -1;
    }

    int* d_numPits;
    int numPits = 0;

    cudaMalloc(&d_numPits, sizeof(int));
    cudaMemcpy(d_numPits, &numPits, sizeof(int), cudaMemcpyHostToDevice);

    //define grid and block size
    dim3 blockSize(8,8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    identifyPits<<<gridSize, blockSize>>>(d_dem, d_pitCells, d_numPits, width, height);
    cudaDeviceSynchronize();

    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess){
        std::cerr << "Kernel Launch failed: " << cudaGetErrorString(kernel_err) << std::endl;
        return -1;
    }

    // copy pit count from device to host
    cudaMemcpy(&numPits, d_numPits, sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<PitCell> hostPits(numPits);
    cudaMemcpy(hostPits.data(), d_pitCells, numPits * sizeof(PitCell), cudaMemcpyDeviceToHost);

    cudaFree(d_dem);
    cudaFree(d_pitCells);
    cudaFree(d_numPits);

    //place pits in PQ
    std::priority_queue<int> pq;
    for (const auto& pit : hostPits){
        pq.push(pit.elevation);
    }

    //OUPUT FOR TESTING
    int j = 0;
    std::cout << "Pits sorted by elevation" << std::endl;
    while(!pq.empty()){
        j++;
        pq.pop();
    }
    std::cout << "Size of PQ: " << j <<std::endl;
}