#include <gdal_priv.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8

__constant__ int offsetX[9] = { 0,  -1,  0,  1, 1, 1, 0, -1, -1};
__constant__ int offsetY[9] = { 0,  -1, -1, -1, 0, 1, 1,  1,  0};
__constant__ int direction[9] = { 0, 1,  2,  3, 4, 5, 6,  7,  8}; 
/*
1: (-1,-1) 2: (0 ,-1) 3: (1,-1)
8: (-1, 0) 0: (0 , 0) 4: (1 ,0)
7: (-1, 1) 6: (0 , 1) 5: (1 ,1)
*/

__device__ int reverseDir(int dir){
                    //  0, 1, 2, 3, 4, 5, 6, 7, 8
    int reverseDir[] = {0, 5, 8, 7, 8, 1, 2, 3, 4};
    return reverseDir[dir];
}
__global__ void streamIdentification(const int* flowAccum, int* stream, int threshold, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return; //skip boundary

    if (flowAccum[idx] >= threshold){
        stream[idx] = 1;
    }else{
        stream[idx] = 0;
    }
}

__global__ void endpointIdentification(const int* flowDir, const int* streams, int* endpoints, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return; //skip boundary

    if (streams[idx] == 1 && flowDir[idx] == 0){
        int hasDownstream = 0;
        int flowDirection = flowDir[idx];
        for (int i = 1; i < 9; i++){
            int nx = x + offsetX[flowDirection];
            int ny = y + offsetY[flowDirection];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nIdx = ny * width + nx;

                if (streams[nIdx] == 1 && flowDir[nIdx] == reverseDir(i)){
                    hasDownstream = 1;
                }
            }
        }
        endpoints[idx] = (hasDownstream == 0) ? 1 : 0;
    } else {
        endpoints[idx] = 0;
    }
}

GDALDataset* openRaster(const char* path, GDALAccess access){
    GDALDataset* dataset = (GDALDataset*) GDALOpen(path, access);
    if (dataset == nullptr){
        std::cerr << "Error opening file" << path << std::endl;
    }
    return dataset;
}

void cleanup(int* hostFlowAccumData, int* hostFlowDirData, int* hostStreamData, 
             int* deviceFlowAccumData, int* deviceFlowDirData, int* deviceStreamData, 
             GDALDataset* flowAccumDataset, GDALDataset* flowDirDataset, GDALDataset* streamDataset, GDALDataset* endpointDataset) {
    // Free host memory if allocated
    if (hostFlowAccumData) CPLFree(hostFlowAccumData);
    if (hostFlowDirData) CPLFree(hostFlowDirData);
    if (hostStreamData) CPLFree(hostStreamData);

    // Free device memory if allocated
    if (deviceFlowAccumData) cudaFree(deviceFlowAccumData);
    if (deviceFlowDirData) cudaFree(deviceFlowDirData);
    if (deviceStreamData) cudaFree(deviceStreamData);

    // Close GDAL datasets if opened
    if (flowAccumDataset) GDALClose(flowAccumDataset);
    if (flowDirDataset) GDALClose(flowDirDataset);
    if (streamDataset) GDALClose(streamDataset);
    if (endpointDataset) GDALClose(endpointDataset);
}


int main(int argc, char* argv[]){
    //ARGS : FlowAccum, FlowDir, StreamData, EndpointData
    if (argc < 5){
        std::cout << "Please provide a filepath for input and output raster" << std::endl;
        return -1;
    }

    // register drivers to open raster data
    GDALAllRegister();
    
    // open input datasets
    GDALDataset* flowAccumDataset = (GDALDataset*) GDALOpen(argv[1], GA_ReadOnly);
    GDALDataset* flowDirDataset = (GDALDataset*) GDALOpen(argv[2], GA_ReadOnly);
    if (!flowAccumDataset || !flowDirDataset) {
        cleanup(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, flowAccumDataset, flowDirDataset, nullptr, nullptr);
        return -1;
    }

    //get projection from input raster
    const char* projection = flowAccumDataset->GetProjectionRef();
    if (projection == nullptr){
        std::cerr << "Error: Could not retrieve projection from Flow Accumulation Dataset." << std::endl;
        cleanup(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, flowAccumDataset, flowDirDataset, nullptr, nullptr);
        return -1;
    }
    double geoTransform[6];

    if (flowAccumDataset->GetGeoTransform(geoTransform) != CE_None){
        std::cerr << "Error reading geo-transform" << std::endl;
        cleanup(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, flowAccumDataset, flowDirDataset, nullptr, nullptr);
        return -1;
    }

    int width = flowAccumDataset->GetRasterXSize();
    int height = flowAccumDataset->GetRasterYSize();

    //create output dataset
    const char *outputFilename1 = argv[3];
    GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset *streamDataset = poDriver->Create(outputFilename1, width, height, 1, GDT_Int32, NULL);
    if (!streamDataset){
        std::cerr << "Error creating output dataset" << std::endl;
        cleanup(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, flowAccumDataset, flowDirDataset, nullptr, nullptr);
        return -1;
    }
    streamDataset->SetProjection(projection);
    streamDataset->SetGeoTransform(geoTransform);

    const char *outputFilename2 = argv[4];
    GDALDataset *endpointDataset = poDriver->Create(outputFilename2, width, height, 1, GDT_Int32, NULL);
    if (!endpointDataset){
        std::cerr << "Error creating output dataset" << std::endl;
        cleanup(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, flowAccumDataset, flowDirDataset, streamDataset, nullptr);
        return -1;
    }
    endpointDataset->SetProjection(projection);
    endpointDataset->SetGeoTransform(geoTransform);

    int *flowAccumData = (int *)CPLMalloc(sizeof(int) * width * height);
    int *flowDirData = (int *)CPLMalloc(sizeof(int) * width * height);
    int *streamData = (int *)CPLMalloc(sizeof(int) * width * height);
    int* endpointData = (int*)CPLMalloc(sizeof(int) * width * height);

    if (!flowAccumData || !flowDirData || !streamData || !endpointData) {
        std::cerr << "Memory allocation error on host." << std::endl;
        cleanup(flowAccumData, flowDirData, streamData, nullptr, nullptr, nullptr, flowAccumDataset, flowDirDataset, streamDataset, endpointDataset);
        return -1;
    }

    // Read data into host memory
    if (flowAccumDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, flowAccumData, width, height, GDT_Int32, 0, 0) != CE_None){
        std::cerr << "Error reading Flow Accumulation data." << std::endl;
        cleanup(flowAccumData, flowDirData, streamData, nullptr, nullptr, nullptr, flowAccumDataset, flowDirDataset, streamDataset, endpointDataset);
        return -1;
    }

    if (flowDirDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, flowDirData, width, height, GDT_Int32, 0, 0) != CE_None){
        std::cerr << "Error reading Flow Direction data." << std::endl;
        cleanup(flowAccumData, flowDirData, streamData, nullptr, nullptr, nullptr, flowAccumDataset, flowDirDataset, streamDataset, endpointDataset);
        return -1;
    }

    int *d_flowAccumData, *d_flowDirData, *d_streamData, *d_endpointData;

    if (cudaMalloc(&d_flowAccumData, sizeof(int) * width * height) != cudaSuccess ||
        cudaMalloc(&d_flowDirData, sizeof(int) * width * height) != cudaSuccess ||
        cudaMalloc(&d_streamData, sizeof(int) * width * height) != cudaSuccess ||
        cudaMalloc(&d_endpointData, sizeof(int) * width * height) != cudaSuccess) {
        std::cerr << "Error allocation memory for data on device" << std::endl;
        cleanup(flowAccumData, flowDirData, streamData, d_flowAccumData, d_flowDirData, d_streamData, flowAccumDataset, flowDirDataset, streamDataset, endpointDataset);
        return -1;
    }

    if (cudaMemcpy(d_flowAccumData, flowAccumData, sizeof(int) * width * height, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_flowDirData, flowDirData, sizeof(int) * width * height, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Error copying data to device." << std::endl;
        cleanup(flowAccumData, flowDirData, streamData, d_flowAccumData, d_flowDirData, d_streamData, flowAccumDataset, flowDirDataset, streamDataset, endpointDataset);
        return -1;
    }
    

    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); //dynamic grid allocation based on input size
    int threshold = 300;

    std::cout << "Stream Identification Kernel Launched" << std::endl;
    streamIdentification<<<gridSize, blockSize>>>(d_flowAccumData, d_streamData, threshold, width, height);
    if (cudaGetLastError() != cudaSuccess){
        std::cerr << "Cuda kernel launch error." << std::endl;
        cleanup(flowAccumData, flowDirData, streamData, d_flowAccumData, d_flowDirData, d_streamData, flowAccumDataset, flowDirDataset, streamDataset, endpointDataset);
        return -1;
    }
    cudaDeviceSynchronize();
    std::cout << "Stream Identification Kernel Finished" << std::endl;

    std::cout << "Endpoint Identification Kernel Launched" << std::endl;
    endpointIdentification<<<gridSize, blockSize>>>(d_flowDirData, d_streamData, d_endpointData, width, height);
    if (cudaGetLastError() != cudaSuccess){
        std::cerr << "Cuda kernel launch error." << std::endl;
        cleanup(flowAccumData, flowDirData, streamData, d_flowAccumData, d_flowDirData, d_streamData, flowAccumDataset, flowDirDataset, streamDataset, endpointDataset);
        return -1;
    }
    cudaDeviceSynchronize();
    std::cout << "Enpoint Identification Kernel Finished" << std::endl;

    //copy results back to host
    if (cudaMemcpy(streamData, d_streamData, sizeof(int) * width * height, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(endpointData, d_endpointData, sizeof(int) * width * height, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Error copying Stream data back from device." << std::endl;
        cleanup(flowAccumData, flowDirData, streamData, d_flowAccumData, d_flowDirData, d_streamData, flowAccumDataset, flowDirDataset, streamDataset, endpointDataset);
        return -1;
    }

    // write results to output raster
    if (streamDataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height, streamData, width, height, GDT_Int32, 0, 0) != CE_None ||
        endpointDataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height, endpointData, width, height, GDT_Int32, 0, 0) != CE_None){
        std::cerr << "Error writing stream data" << std::endl;
        cleanup(flowAccumData, flowDirData, streamData, d_flowAccumData, d_flowDirData, d_streamData, flowAccumDataset, flowDirDataset, streamDataset, endpointDataset);
        return -1;
    }

    std::cout << "Stream data calculated and saved to: " << outputFilename1 << " and " <<outputFilename2 << std::endl;

    //final resource cleanup
    cleanup(flowAccumData, flowDirData, streamData, d_flowAccumData, d_flowDirData, d_streamData, flowAccumDataset, flowDirDataset, streamDataset, endpointDataset);
    return 0;
}