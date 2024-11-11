#include <iostream>
#include <gdal_priv.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define THREADCELLS 4
#define BLOCK_SIZE 16

__constant__ int offsetX[9] = {0, -1, 0, 1, 1, 1, 0, -1, -1};
__constant__ int offsetY[9] = {0, -1, -1, -1, 0, 1, 1, 1, 0};
__constant__ int direction[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8}; 


//two nested for loops - handle in the neighborhood sequentially
//fourth array - takes place of the cells - used for writing final to? cumulative?
__global__ void flowAccumKernel(int* gpuAccum, int* gpuOldFlow, int* gpuNewFlow, int * flowDir, bool* gpuStop, int N, int M){
    int i = THREADCELLS * (blockIdx.y * BLOCK_SIZE + threadIdx.y);
    int j = THREADCELLS * (blockIdx.x * BLOCK_SIZE + threadIdx.x);

    for (int r = i; r < i + THREADCELLS && r < N; r++){
        for (int s = j; s < j + THREADCELLS && s < M; s++){
            int currFlow = gpuOldFlow[r * M + s];
            if (currFlow > 0){
                gpuOldFlow[r * M + s] = 0;
                int cellFlowDir = flowDir[r * M + s]; 
                int k = offsetX[cellFlowDir];
                int l = offsetY[cellFlowDir];

                int newR = r + l;
                int newS = s + k;
                if (newR >= 0 && newR < N && newS >= 0 && newS < M) {
                    atomicAdd(&gpuNewFlow[newR * M + newS], currFlow);
                    atomicAdd(&gpuAccum[newR * M + newS], currFlow);
                }
            } 
        }
    }
}

int main(int argc, char* argv[]){

    //checks for input file passed as arg
    if (argc < 3){
        std::cout << "Please provide a filepath for input and output raster" << std::endl;
        return -1;
    }
    // register drivers to open raster data
    GDALAllRegister();
    
    // Open DEM dataset
    const char* input = argv[1];
    GDALDataset* D8Dataset  = (GDALDataset*) GDALOpen(input, GA_ReadOnly);

    if (D8Dataset == NULL) {
        std::cerr << "Error opening flow direction file." << std::endl;
        return -1;
    }

    //create output raster for flow accumulation
    const char *outputFilename = argv[2];

    //Geotiff Driver
    GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    //32int Empty raster with same dims as input
    GDALDataset *flowAccumDataset = poDriver->Create(outputFilename,
                                                    D8Dataset->GetRasterXSize(),
                                                    D8Dataset->GetRasterYSize(),
                                                    1, GDT_Int32, NULL);

    //Raster size to use with Malloc and device mem
    int width = D8Dataset->GetRasterXSize();
    int height = D8Dataset->GetRasterYSize();
    bool *stopFlag = new bool;
    int *flowDir = (int *)CPLMalloc(sizeof(int) * width * height);

    //populate demData dynamically allocated memory
    CPLErr err = D8Dataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, flowDir, width, height, GDT_Int32, 0, 0);
    if (err != CE_None){
        std::cerr << "Error reading DEM data: " << CPLGetLastErrorMsg() << std::endl;
        return -1;
    }

    //allocating device mem
    int *d_oldFlow, *d_newFlow, *d_flowDir, *d_accum;
    bool *d_stopFlag;
    
    // Allocate memory for d_oldFlow on device
    if (cudaMalloc(&d_oldFlow, width * height * sizeof(int)) != cudaSuccess) {
        std::cerr << "Error allocating memory for Old Flow on device" << std::endl;
        return -1;
    }
    // Allocate memory for d_newFlow on device
    if (cudaMalloc(&d_newFlow, width * height * sizeof(int)) != cudaSuccess) {
        std::cerr << "Error allocating memory for New Flow on device" << std::endl;
        return -1;
    }
    // Allocate memory for d_flowDir on device
    if (cudaMalloc(&d_flowDir, width * height * sizeof(int)) != cudaSuccess) {
        std::cerr << "Error allocating memory for Flow Direction on device" << std::endl;
        return -1;
    }
      // Allocate memory for d_flowDir on device
    if (cudaMalloc(&d_accum, width * height * sizeof(int)) != cudaSuccess) {
        std::cerr << "Error allocating memory for Flow Direction on device" << std::endl;
        return -1;
    }
    // Allocate memory for d_stopFlag on device
    if (cudaMalloc(&d_stopFlag, sizeof(bool)) != cudaSuccess) {
        std::cerr << "Error allocating memory for Stop Flag on device" << std::endl;
        return -1;
    }
        
    //copy flow direction data to device
    cudaError_t memcpy_err_flowDir = cudaMemcpy(d_flowDir, flowDir, sizeof(int) * width * height, cudaMemcpyHostToDevice);
    if (memcpy_err_flowDir != cudaSuccess){
        std::cerr << "Error copying data to device: " << cudaGetErrorString(memcpy_err_flowDir) << std::endl;
        return -1;
    }
    int* hostOldFlow = new int[width * height];
    for (int i = 0; i < width * height; ++i) {
        hostOldFlow[i] = 1;
    }

    int* hostNewFlow = new int [width*height];
    for (int i = 0; i < width * height; ++i) {
        hostNewFlow[i] = 0;
    }

    cudaMemcpy(d_oldFlow, hostOldFlow, width * height * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_newFlow, hostNewFlow, width * height * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_accum, 0, sizeof(int) * width * height);   

    //define grid and block size
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    //do{
    for(int x = 0; x < 3; x++){
        printf("Iteration: %d\n", x);
        *stopFlag = false;
        cudaMemcpy(d_stopFlag, stopFlag, sizeof(bool), cudaMemcpyHostToDevice);
        flowAccumKernel<<<gridSize, blockSize>>>(d_accum, d_oldFlow, d_newFlow, d_flowDir, d_stopFlag, height, width);
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess){
            std::cerr << "Error launching kernel: " << cudaGetErrorString(kernelErr) << std::endl;
            return -1;
        }
        cudaDeviceSynchronize();
        cudaMemcpy(stopFlag, d_stopFlag, sizeof(bool), cudaMemcpyDeviceToHost);
        int *temp = d_oldFlow;
        d_oldFlow = d_newFlow;
        d_newFlow = temp;

        cudaMemset(d_newFlow, 0, sizeof(int) * width * height);
        
    //} while (*stopFlag);
    }
    int *hostflowAccumulationData = (int *)CPLMalloc(sizeof(int) * width * height);
    cudaDeviceSynchronize();
    cudaMemcpy(hostflowAccumulationData, d_accum, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
    err = flowAccumDataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height,
        hostflowAccumulationData, width, height, GDT_Int32, 0, 0);
    if (err != CE_None) {
        std::cerr << "Error writing flow accumulation data: " << CPLGetLastErrorMsg() << std::endl;
        return -1;
    }
    
    cudaFree(d_oldFlow);
    cudaFree(d_newFlow);
    cudaFree(d_flowDir);
    cudaFree(d_stopFlag);
    cudaFree(d_accum);
    CPLFree(flowDir);
    GDALClose(D8Dataset);
    GDALClose(flowAccumDataset);
    return 0;
}