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

__global__ void neighbourIndicePrecompute(int* flowDir, int* nI, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return; //skip boundary

    int flowIdx = -1;
    int flowDirection = flowDir[idx];
    int nx = x + offsetX[flowDirection];
    int ny = y + offsetY[flowDirection];
    int nIdx = ny * width + nx;

    if (nIdx != idx) {
        flowIdx = nIdx;
    }

    nI[idx] = flowIdx;
}

__global__ void assignWatershedIDs(const int* endpoints, int* labels, int* idCounter, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return; //skip boundary

    if (endpoints[idx] == 1){
        labels[idx] = atomicAdd(idCounter, 1);
    } else {
        labels[idx] = -1;
    }
}

__global__ void watershedLabelProp(const int* flowDir, const int* downstreamIdx, int* labels, int* modified, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return; //skip boundary

    if (labels[idx] == -1){
        int nIdx = downstreamIdx[idx];
        if (labels[nIdx] != -1){
            labels[idx] = downstreamIdx[nIdx];
            atomicOr(modified, 1);
        }
    }
}

void cleanup(GDALDataset* flowDirDataset, GDALDataset* endpointDataset, GDALDataset* watershedDataset, int* flowDirData, 
            int* endpointData, int* watershedData, int* neighbourIndices, int *d_flowDirData, int *d_endpointData, 
            int *d_watershedData, int *d_neighbourIndices, int *d_idCounter, int *d_stopFlag){
    if (flowDirDataset) GDALClose(flowDirDataset);
    if (endpointDataset) GDALClose(endpointDataset);
    if (watershedDataset) GDALClose(watershedDataset);

    if (flowDirData) CPLFree(flowDirData);
    if (endpointData) CPLFree(endpointData);
    if (watershedData) CPLFree(watershedData);
    if (watershedData) CPLFree(watershedData);

    if (d_flowDirData) cudaFree(d_flowDirData);
    if (d_endpointData) cudaFree(d_endpointData);
    if (d_watershedData) cudaFree(d_watershedData);
    if (d_neighbourIndices) cudaFree(d_neighbourIndices);
    if (d_idCounter) cudaFree(d_idCounter);
    if (d_stopFlag) cudaFree(d_stopFlag);
}
int main(int argc, char* argv[]){
    //ARGS: FlowDir, Endpoints, Watersheds
    if (argc < 4){
        std::cout << "Please provide necessary filepaths" << std::endl;
        return -1;
    }

    // register drivers to open raster data
    GDALAllRegister();

        // open input datasets
    GDALDataset* flowDirDataset = (GDALDataset*) GDALOpen(argv[1], GA_ReadOnly);
    GDALDataset* endpointDataset = (GDALDataset*) GDALOpen(argv[2], GA_ReadOnly);
    if (!flowDirDataset || !endpointDataset) {
        cleanup(flowDirDataset, endpointDataset, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        return -1;
    }

    //get projection from input raster
    const char* projection = flowDirDataset->GetProjectionRef();
    if (projection == nullptr){
        std::cerr << "Error: Could not retrieve projection from Flow Accumulation Dataset." << std::endl;
        cleanup(flowDirDataset, endpointDataset, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        return -1;
    }
    double geoTransform[6];

    if (flowDirDataset->GetGeoTransform(geoTransform) != CE_None){
        std::cerr << "Error reading geo-transform" << std::endl;
        cleanup(flowDirDataset, endpointDataset, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        return -1;
    }

    int width = flowDirDataset->GetRasterXSize();
    int height = flowDirDataset->GetRasterYSize();

    //create output dataset
    const char *outputFilename = argv[3];
    GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset *watershedDataset = poDriver->Create(outputFilename, width, height, 1, GDT_Int32, NULL);
    if (!watershedDataset){
        std::cerr << "Error creating output dataset" << std::endl;
        cleanup(flowDirDataset, endpointDataset, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        return -1;
    }
    watershedDataset->SetProjection(projection);
    watershedDataset->SetGeoTransform(geoTransform);

    int *flowDirData = (int *)CPLMalloc(sizeof(int) * width * height);
    int *endpointData = (int *)CPLMalloc(sizeof(int) * width * height);
    int *watershedData = (int *)CPLMalloc(sizeof(int) * width * height);
    int *neighbourIndices = (int *)CPLMalloc(sizeof(int) * width * height);

    if (!flowDirData || !endpointData || !watershedData || !neighbourIndices){
        std::cerr << "Memory allocation error on host." << std::endl;
        cleanup(flowDirDataset, endpointDataset, watershedDataset, flowDirData, endpointData, watershedData, neighbourIndices, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        return -1;
    }

    if (flowDirDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, flowDirData, width, height, GDT_Int32, 0, 0) != CE_None){
        std::cerr << "Error reading Flow Direction data." << std::endl;
        cleanup(flowDirDataset, endpointDataset, watershedDataset, flowDirData, endpointData, watershedData, neighbourIndices, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        return -1;
    }

    if (endpointDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, endpointData, width, height, GDT_Int32, 0, 0) != CE_None){
        std::cerr << "Error reading endpoint data." << std::endl;
        cleanup(flowDirDataset, endpointDataset, watershedDataset, flowDirData, endpointData, watershedData, neighbourIndices, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        return -1;
    }

    int *d_flowDirData, *d_endpointData, *d_watershedData, *d_neighbourIndices, *d_idCounter, *d_stopFlag;
    int idCounter = 0;

    if (cudaMalloc(&d_flowDirData, sizeof(int) * width * height) != cudaSuccess ||
        cudaMalloc(&d_endpointData, sizeof(int) * width * height) != cudaSuccess ||
        cudaMalloc(&d_watershedData, sizeof(int) * width * height) != cudaSuccess ||
        cudaMalloc(&d_neighbourIndices, sizeof(int) * width * height) != cudaSuccess ||
        cudaMalloc(&d_idCounter, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&d_stopFlag, sizeof(int)) != cudaSuccess) {
            
            std::cerr << "Error allocating memory on device" << std::endl;
            cleanup(flowDirDataset, endpointDataset, watershedDataset, flowDirData, endpointData, watershedData, neighbourIndices, d_flowDirData, d_endpointData, d_watershedData, d_neighbourIndices, d_idCounter, d_stopFlag);
            return -1;
    }

    if (cudaMemcpy(d_flowDirData, flowDirData, sizeof(int) * width * height, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_endpointData, endpointData, sizeof(int) * width * height, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_idCounter, &idCounter, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Error copying data to device." << std::endl;
        cleanup(flowDirDataset, endpointDataset, watershedDataset, flowDirData, endpointData, watershedData, neighbourIndices, d_flowDirData, d_endpointData, d_watershedData, d_neighbourIndices, d_idCounter, d_stopFlag);
        return -1;
    }
    
    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); //dynamic grid allocation based on input size

    // precompute downstream neighbour indices
    std::cout << "Launching neighbour index precomputation Kernel" <<std::endl;
    neighbourIndicePrecompute<<<gridSize, blockSize>>>(d_flowDirData, d_neighbourIndices, width, height);
    if (cudaGetLastError() != cudaSuccess){
        std::cerr << "Cuda kernel launch error." << std::endl;
        cleanup(flowDirDataset, endpointDataset, watershedDataset, flowDirData, endpointData, watershedData, neighbourIndices, d_flowDirData, d_endpointData, d_watershedData, d_neighbourIndices, d_idCounter, d_stopFlag);
        return -1;
    }
    cudaDeviceSynchronize();
    std::cout << "Finished neighbour index precomputation Kernel" <<std::endl;

    // assign watershed labels
    std::cout << "Launching Watershed Label Assignment Kernel" <<std::endl;
    assignWatershedIDs<<<gridSize, blockSize>>>(d_endpointData, d_watershedData, d_idCounter, width, height);
    if (cudaGetLastError() != cudaSuccess){
        std::cerr << "Cuda kernel launch error." << std::endl;
        cleanup(flowDirDataset, endpointDataset, watershedDataset, flowDirData, endpointData, watershedData, neighbourIndices, d_flowDirData, d_endpointData, d_watershedData, d_neighbourIndices, d_idCounter, d_stopFlag);
        return -1;
    }
    cudaDeviceSynchronize();

    // pull-based label propogation
    int x = 0;
    int *stopFlag = new int(0);
    do{
        printf("Kernel iteration: %d\n", x++ + 1);
        *stopFlag = 0;
        cudaMemcpy(d_stopFlag, stopFlag, sizeof(int), cudaMemcpyHostToDevice);
        watershedLabelProp<<<gridSize, blockSize>>>(d_flowDirData, d_neighbourIndices, d_watershedData, d_stopFlag, height, width);
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess){
            std::cerr << "Error launching kernel: " << cudaGetErrorString(kernelErr) << std::endl;
            return -1;
        }
        cudaDeviceSynchronize();
        cudaMemcpy(stopFlag, d_stopFlag, sizeof(int), cudaMemcpyDeviceToHost);

        if (x == 500){
            break;
        }
    } while(*stopFlag != 0);

    if (cudaMemcpy(watershedData, d_watershedData, sizeof(int) * width * height, cudaMemcpyDeviceToHost) != cudaSuccess){
        std::cerr << "Error copying back delineated watersheds" << std::endl;
        cleanup(flowDirDataset, endpointDataset, watershedDataset, flowDirData, endpointData, watershedData, neighbourIndices, d_flowDirData, d_endpointData, d_watershedData, d_neighbourIndices, d_idCounter, d_stopFlag);
        return -1;
    }

    if (watershedDataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height, watershedData, width, height, GDT_Int32, 0, 0) != CE_None){
        std::cerr << "Error writing watershed data to tif file" << std::endl;
        cleanup(flowDirDataset, endpointDataset, watershedDataset, flowDirData, endpointData, watershedData, neighbourIndices, d_flowDirData, d_endpointData, d_watershedData, d_neighbourIndices, d_idCounter, d_stopFlag);
        return -1;
    }


    std::cout << "Watersheds delineated and writted to: " << outputFilename << std::endl;
    cleanup(flowDirDataset, endpointDataset, watershedDataset, flowDirData, endpointData, watershedData, neighbourIndices, d_flowDirData, d_endpointData, d_watershedData, d_neighbourIndices, d_idCounter, d_stopFlag);
    return 0;
}