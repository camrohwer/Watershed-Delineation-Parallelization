#include <iostream>
#include <gdal_priv.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <float.h>

#define FLOW_NODATA -1
#define HEIGHT_CONST 0.1f
#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8

__constant__ int offsetX[8] = { -1, 0, 1, 0, -1, 1, 1, -1 };
__constant__ int offsetY[8] = { 0, -1, 0, 1, -1, -1, 1, 1 };
__constant__ int direction[8] = { 8, 2, 4, 6, 1, 3, 5, 7 }; 

__global__ void pitFillFlowDirectionKernel(const float* dem, int* flow_dir, int* numPits, int width, int height, float hc) {
    //unique thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return; //skip boundary 
    
    __shared__ float sharedDem[BLOCK_DIM_X + 2][BLOCK_DIM_Y + 2]; //extra column padding

    //local indices for use with padded shared
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    //copy center pixel to shared
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
    //top left corner
    if (threadIdx.x == 0 && threadIdx.y == 0 && x > 0 && y > 0) {
    sharedDem[0][0] = dem[(y - 1) * width + (x - 1)];
    }
    //top right corner
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0 && x < width - 1 && y > 0) {
        sharedDem[0][tx + 1] = dem[(y - 1) * width + (x + 1)];
    }
    //bottom left corner
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1 && x > 0 && y < height - 1) {
        sharedDem[ty + 1][0] = dem[(y + 1) * width + (x - 1)];
    }
    // bottom right corner
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1 && x < width - 1 && y < height - 1) {
        sharedDem[ty + 1][tx + 1] = dem[(y + 1) * width + (x + 1)];
    }
    __syncthreads();

    float centre = sharedDem[ty][tx]; //get dem value at current pixel
    float lowest = centre;
    int dir = FLOW_NODATA;
    float lowestNeighbour = FLT_MAX;
    bool isPit = true;

    //pit check
    for (int i = 0; i < 8; i++) {
        float neighbourElev = sharedDem[ty + offsetY[i]][tx + offsetX[i]];
        if (neighbourElev <= centre){
            isPit = false;
            break;
        }else{
            if (neighbourElev < lowestNeighbour){
                lowestNeighbour = neighbourElev;
            }
        }
    }

    //keep track of pitcount in shared to reduce writes to global
    __shared__ int localPits;
    if (threadIdx.x == 0 && threadIdx.y == 0) localPits = 0;
    __syncthreads();

    if (isPit){
        atomicAdd(&localPits, 1);
        //update sharedDem with pit filling step for flow direction calculation
        sharedDem[ty][tx] = lowestNeighbour + hc;
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0){
        atomicAdd(numPits, localPits);
    }

    for (int i = 0; i < 8; i++) {
        float neighbour = sharedDem[ty + offsetY[i]][tx + offsetX[i]];

        if (neighbour < lowest){
            lowest = neighbour;
            dir = direction[i];
        }
    }
    if (dir != FLOW_NODATA){
        flow_dir[y * width + x] = dir;
    }
}

void cleanup(float* demData, int* flowDirData, float* d_demData, int* d_flowDirData, int* d_numPits) {
    //helped function for cleaup of dynamically allocated memory
    if (demData) CPLFree(demData);
    if (flowDirData) CPLFree(flowDirData);
    if (d_demData) cudaFree(d_demData);
    if (d_flowDirData) cudaFree(d_flowDirData);
    if (d_numPits) cudaFree(d_numPits);
}

int main(int argc, char* argv[]) {
    //checks for input file passed as arg
    if (argc < 3){
        std::cout << "Please provide a filepath for input and output raster" << std::endl;
        return -1;
    }

    // register drivers to open raster data
    GDALAllRegister();
    
    // Open DEM dataset
    const char* input = argv[1];
    GDALDataset* demDataset  = (GDALDataset*) GDALOpen(input, GA_ReadOnly);

    if (demDataset == nullptr) {
        std::cerr << "Error opening DEM file." << std::endl;
        return -1;
    }

    //get projection from input raster
    const char* projection = demDataset->GetProjectionRef();
    if (projection == nullptr){
        std::cerr << "Error: Could not retrieve projection from DEM Dataset." << std::endl;
        GDALClose(demDataset);
        return -1;
    }
    double geoTransform[6];

    if (demDataset->GetGeoTransform(geoTransform) != CE_None){
        std::cerr << "Error reading geo-transform" << std::endl;
        GDALClose(demDataset);
        return -1;
    }

    //create output raster for flow direction
    const char *outputFilename = argv[2];
    //Geotiff Driver
    GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    //32int Empty raster with same dims as input
    GDALDataset *flowDirDataset = poDriver->Create(outputFilename,
                                                    demDataset->GetRasterXSize(),
                                                    demDataset->GetRasterYSize(),
                                                    1, GDT_Int32, NULL);

    flowDirDataset->SetProjection(projection);
    flowDirDataset->SetGeoTransform(geoTransform);

    //Raster size to use with Malloc and device mem
    int width = demDataset->GetRasterXSize();
    int height = demDataset->GetRasterYSize();
    float *demData = (float *)CPLMalloc(sizeof(float) * width * height);

    //populate demData dynamically allocated memory
    CPLErr err = demDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, demData, width, height, GDT_Float32, 0, 0);
    if (err != CE_None){
        std::cerr << "Error reading DEM data: " << CPLGetLastErrorMsg() << std::endl;
        cleanup(demData, nullptr, nullptr, nullptr, nullptr);
        return -1;
    }

    int *flowDirData = (int *)CPLMalloc(sizeof(int) * width * height);

    //create cuda stream
    cudaStream_t stream;
    cudaError_t stream_err =cudaStreamCreate(&stream);
    if (stream_err != cudaSuccess){
        std::cerr << "Error creating CUDA stream: " <<cudaGetErrorString(stream_err) << std::endl;
        cleanup(demData, flowDirData, nullptr, nullptr, nullptr);
        return -1;
    }

    int numPits = 0;
    int* d_numPits; 
    if (cudaMalloc(&d_numPits, sizeof(int)) != cudaSuccess){
        std::cerr << "Error allocationg memory for Pit Count on device." << std::endl;
        cleanup(demData, flowDirData, nullptr, nullptr, d_numPits);
        return -1;
    }
    cudaMemset(d_numPits, 0, sizeof(int));

    //allocating device mem
    float *d_demData;
    if (cudaMalloc(&d_demData, sizeof(float) * width * height) != cudaSuccess) {
        std::cerr << "Error allocating memory for DEM on device." << std::endl;
        cleanup(demData, flowDirData, d_demData, nullptr, d_numPits);
        return -1;
    }

    int *d_flowDirData;
    if (cudaMalloc(&d_flowDirData, sizeof(int) * width * height) != cudaSuccess) {
        std::cerr << "Error allocating memory for flow direction on device." << std::endl;
        cleanup(demData, flowDirData, d_demData, d_flowDirData, d_numPits);
        return -1;
    }

    //copy DEM data to device
    cudaError_t memcpy_err = cudaMemcpyAsync(d_demData, demData, sizeof(int) * width * height, cudaMemcpyHostToDevice);
    if (memcpy_err != cudaSuccess){
        std::cerr << "Error copying data to device: " << cudaGetErrorString(memcpy_err) << std::endl;
        return -1;
    }

    //define grid and block size
    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); //dynamic grid allocation based on input size

    // Launch the CUDA kernel
    pitFillFlowDirectionKernel<<<gridSize, blockSize, 0, stream>>>(d_demData, d_flowDirData, d_numPits, width, height, HEIGHT_CONST);

    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess){
        std::cerr << "Cuda kernel launch error: " << cudaGetErrorString(kernel_err) << std::endl;
        return -1;
    }

    // Copy flow direction data back to host
    memcpy_err = cudaMemcpyAsync(flowDirData, d_flowDirData, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
    if (memcpy_err != cudaSuccess){
        std::cerr << "Error copying data to device: " << cudaGetErrorString(memcpy_err) << std::endl;
        return -1;
    }

    //copy numpits back to device
    memcpy_err = cudaMemcpyAsync(&numPits, d_numPits, sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (memcpy_err != cudaSuccess){
        std::cerr << "Error copying data to device: " << cudaGetErrorString(memcpy_err) << std::endl;
        return -1;
    }

    cudaStreamSynchronize(stream);
    
    // Write flow direction data to the output dataset
    err = flowDirDataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height, flowDirData, width, height, GDT_Int32, 0, 0);
    if (err != CE_None) {
        std::cerr << "Error writing flow direction data: " << CPLGetLastErrorMsg() << std::endl;
        return -1;
    }

    // Cleanup
    cleanup(demData, flowDirData, d_demData, d_flowDirData, d_numPits);
    GDALClose(demDataset);
    GDALClose(flowDirDataset);

    std::cout << "Flow direction calculated and saved to " << outputFilename << std::endl;
    return 0;
}
