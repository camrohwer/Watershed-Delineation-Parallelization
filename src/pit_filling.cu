#include <iostream>
#include <gdal_priv.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <vector>
#include <limits.h>
#include <float.h>

#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
// ./pit_filling ../../DEMs/092F.tif ../../DEMs/Output/092F_filled.tif
struct PitCell {
    float elevation;
    int index;
};

__global__ void pitFilling(PitCell* pits, int numPits, float* dem, const int width, const int height, const float hc){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPits) return;

    PitCell pit = pits[idx];
    int x = pit.index % width;
    int y = pit.index / width;

    float lowestNeighbour = FLT_MAX;
    
    for (int dy = -1; dy <= 1; dy++){
        for (int dx = -1; dx <= 1; dx++){
            if (dx == 0 && dy == 0) continue; //skip current pixel

            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height){
                float n = dem[ny * width + nx];

                if (n < lowestNeighbour) {
                    lowestNeighbour = n;
                }
            }
        }
    }
    if (lowestNeighbour != FLT_MAX) {
        dem[pit.index] = lowestNeighbour + hc;
    }
}
__global__ void identifyPits(const float* dem, PitCell* pitCells, int* numPits, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return; // skip boundary 

     __shared__ float sharedDem[BLOCK_DIM_Y + 2][BLOCK_DIM_X + 2];

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    
    if (x < width && y < height){
        sharedDem[ty][tx] = dem[y * width + x];

        //left padding 
        if (threadIdx.x == 0 && x > 0){
            sharedDem[ty][0] = dem[y * width + (x - 1)];
        }

        //right padding
        if (threadIdx.x == blockDim.x - 1 && x < width - 1) {
            sharedDem[ty][tx + 1] = dem[y * width + (x + 1)];
        }

        //top padding
        if (threadIdx.y == 0 && y > 0){ 
            sharedDem[0][tx] = dem[(y - 1) * width + x];
        }

        //bottom padding
        if (threadIdx.y == blockDim.y - 1 && y < height - 1){
            sharedDem[ty + 1][tx] = dem[(y + 1) * width + x];
        }
    }
    __syncthreads();

    float curElev = sharedDem[ty][tx];
    bool isPit = true;

    const int offsetX[8] = { -1, 0, 1, 0, -1, 1, 1, -1 };
    const int offsetY[8] = { 0, -1, 0, 1, -1, -1, 1, 1 };

    for (int i = 0; i < 8; i++){
        float neigbourElev = sharedDem[ty + offsetY[i]][tx + offsetX[i]];

        if (neigbourElev <= curElev){
            isPit = false;
            break;
        }
    }
    // if pit, store elev
    if (isPit){
        int count = atomicAdd(numPits, 1);
        pitCells[count].elevation = curElev;
        pitCells[count].index = y * width + x;
    }
}
int main(int argc, char* argv[]){
    if (argc < 3){
        std::cout << "Please provide a filepath for input and output raster" << std::endl;
        return -1;
    }

    // register drivers to open raster data
    GDALAllRegister();
    
    // Open DEM dataset
    const char* input = argv[1];
    GDALDataset* demDataset  = (GDALDataset*) GDALOpen(input, GA_ReadOnly);

    if (demDataset == nullptr){
        std::cerr << "Error opening DEM file" << std::endl;
        return -1;
    }

    const char* projection = demDataset->GetProjectionRef();
    double geoTransform[6];

    if (demDataset->GetGeoTransform(geoTransform) != CE_None){
        std::cerr << "Error reading geo-transform" << std::endl;
        GDALClose(demDataset);
        return -1;
    }

    //create output raster for pit filling
    const char *outputFilename = argv[2];
    //Geotiff Driver
    GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    //32int Empty raster with same dims as input
    GDALDataset *pitFillingDataset = poDriver->Create(outputFilename,
                                                    demDataset->GetRasterXSize(),
                                                    demDataset->GetRasterYSize(),
                                                    1, GDT_Float32, NULL);

    pitFillingDataset->SetProjection(projection);
    pitFillingDataset->SetGeoTransform(geoTransform);

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
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
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

    if (numPits > 0){
        std::cout << "Number of pits to be filled: " << numPits << std::endl;

        pitFilling<<<(numPits + 255) / 256, 256>>>(d_pitCells, numPits, d_dem, width, height, 0.0001f);
        cudaDeviceSynchronize();

        cudaError_t fillKernelErr = cudaGetLastError();
        if (fillKernelErr != cudaSuccess){
            std::cerr << "Pit Filling Kernel Failed: " << cudaGetErrorString(fillKernelErr) << std::endl;
            return -1;
        }
    }

    cudaMemcpy(demData, d_dem, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    // Write pit filling data to the output dataset
    err = pitFillingDataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height, demData, width, height, GDT_Float32, 0, 0);
    if (err != CE_None) {
        std::cerr << "Error writing pit filling data: " << CPLGetLastErrorMsg() << std::endl;
        return -1;
    }

    cudaFree(d_dem);
    cudaFree(d_pitCells);
    cudaFree(d_numPits);
    CPLFree(demData);
    GDALClose(demDataset);
    GDALClose(pitFillingDataset);

    std::cout << "Pit filling completed and output written to " << outputFilename << std::endl;
    return 0;
}