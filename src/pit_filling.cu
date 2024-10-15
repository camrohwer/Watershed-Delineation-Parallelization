#include <iostream>
#include <gdal_priv.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <vector>

__global__ void identifyPits(const float* dem, int* pits, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;    

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
                if (dem[ni] < curElev){
                    isPit = false;
                    break;
                }
            }
        }
        if (!isPit) break;
    }

    // if pit, store elev
    if (isPit){
        pits[idx] = curElev;
    } else {
        pits[idx] = -1; //-1 is non pit
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
        return -1;
    }

    //allocating device mem
    float *d_dem;
    int *d_pits;

    cudaMalloc(&d_dem, sizeof(float) * width * height);
    if (cudaMalloc(&d_dem, sizeof(float) * width * height) != cudaSuccess) {
        std::cerr << "Error allocating memory for DEM on device." << std::endl;
        return -1;
    }

    cudaMalloc(&d_pits, sizeof(int) * width * height);
    if (cudaMalloc(&d_pits, sizeof(int) * width * height) != cudaSuccess) {
        std::cerr << "Error allocating memory for flow direction on device." << std::endl;
        cudaFree(d_dem); // Free already allocated memory
        return -1;
    }

    //copy DEM data to device
    cudaError_t memcpy_err = cudaMemcpy(d_dem, demData, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    if (memcpy_err != cudaSuccess){
        std::cerr << "Error copying data to device: " << cudaGetErrorString(memcpy_err) << std::endl;
        return -1;
    }

    //define grid and block size
    dim3 blockSize(8,8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    identifyPits<<<gridSize, blockSize>>>(d_dem, d_pits, width, height);
    cudaDeviceSynchronize();

    //copy pits back to host
    int size = width * height;
    std::vector<int> pits(size);
    cudaMemcpy(pits.data(), d_pits, size * sizeof(int), cudaMemcpyDeviceToHost);

    size_t pitsSize = pits.size();
    std::cout << "Number of pits: " << pitsSize << std::endl;

    cudaFree(d_dem);
    cudaFree(d_pits);

    //place pits in PQ
    std::priority_queue<int> pq;

    for (int i = 0; i < size; ++i){
        if (pits[i] != -1){
            pq.push(pits[i]);
        }
    }

    //OUPUT FOR TESTING
    std::cout << "Pits sorted by elevation" << std::endl;
    while(!pq.empty()){
        //std::cout << pq.top() << std::endl;
        pq.pop();
    }
}