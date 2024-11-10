#include <iostream>
#include <gdal_priv.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 16

__global__ void flowAccumKernel(int* gpuOldFlow, int* gpuNewFlow, int * flowDir, bool* gpuStop, int N, int M){
    //__shared__ int sharedOldFlow[BLOCK_SIZE][BLOCK_SIZE];
    //*gpuStop = false;
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.x + threadIdx.y;
    
    if (x >= M || y >= N){
        return;
    }
    int idx = y * M + x;
    //load flow data into shared memory
    //sharedOldFlow[x][y] = gpuOldFlow[idx];
    //__syncthreads();

    int flow = gpuOldFlow[y* M + x];
    //printf("(%d, %d) - %d  ",x,y, flow);
    if (flow <= 0 || flowDir[idx] == 0) return;
    if (flow > 0){
        gpuOldFlow[y* M + x] = 0;
        int dir = flowDir[idx];

        int targetX = x;
        int targetY = y;
        switch (dir){
            case 1: //northwest
                targetY -=1;
                targetX -=1;
                break;
            case 2: //north
                targetY -=1;
                break;
            case 3: //northeast
                targetY -=1;
                targetX +=1;
                break;
            case 4: //east
                targetX += 1;
                break;
            case 5: //southeast
                targetY +=1;
                targetX +=1;
                break;
            case 6: //south
                targetY +=1;
                break;
            case 7: //southwest
                targetY +=1;
                targetX -=1;
                break;
            case 8: //west
                targetX -=1;
                break;
        }
        //printf("--%d, %d--", targetX, targetY);

        if (targetX >= 0 && targetX < M && targetY >= 0 && targetY < N){
            atomicAdd(&gpuNewFlow[targetY * M + targetX], flow);
            *gpuStop = true;
            //printf("adjustment");
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
    //const char *outputFilename = "../../DEMs/Output/parallel_flow_accum.tif"; //TODO should fix abs paths
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
    float *flowDir = (float *)CPLMalloc(sizeof(float) * width * height);
    

    //populate demData dynamically allocated memory
    CPLErr err = D8Dataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, flowDir, width, height, GDT_Float32, 0, 0);
    if (err != CE_None){
        std::cerr << "Error reading DEM data: " << CPLGetLastErrorMsg() << std::endl;
        return -1;
    }

    //allocating device mem
    int *d_oldFlow, *d_newFlow, *d_flowDir;
    bool *d_stopFlag;

    
    cudaMalloc(&d_oldFlow, width * height * sizeof(int));
    cudaMalloc(&d_newFlow, width * height * sizeof(int));
    cudaMalloc(&d_flowDir, width * height * sizeof(int));
    cudaMalloc(&d_stopFlag, sizeof(bool));
    
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

    cudaMemcpy(d_oldFlow, hostOldFlow, width * height * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_newFlow, 0, sizeof(int) * width *height);
    //define grid and block size
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);


    //call kernel 
    do{
        *stopFlag = false;
        cudaMemcpy(d_stopFlag, stopFlag, sizeof(bool), cudaMemcpyHostToDevice);
        flowAccumKernel<<<gridSize, blockSize>>>(d_oldFlow, d_newFlow, d_flowDir, d_stopFlag, height, width);
        cudaDeviceSynchronize();
        cudaMemcpy(stopFlag, d_stopFlag, sizeof(bool), cudaMemcpyDeviceToHost);
        int *temp = d_oldFlow;
        d_oldFlow = d_newFlow;
        d_newFlow = temp;

        cudaMemset(d_newFlow, 0, sizeof(int) * width * height);
    } while (*stopFlag);
    int *hostflowAccumulationData = (int *)CPLMalloc(sizeof(int) * width * height);
    cudaDeviceSynchronize();
    cudaMemcpy(hostflowAccumulationData, d_oldFlow, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
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
    CPLFree(flowDir);
    GDALClose(D8Dataset);
    GDALClose(flowAccumDataset);
    return 0;


}