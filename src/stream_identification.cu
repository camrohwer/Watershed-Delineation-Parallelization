#include <gdal_priv.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8

__constant__ int offsetX[8] = { -1, 0, 1, 0, -1, 1, 1, -1 };
__constant__ int offsetY[8] = { 0, -1, 0, 1, -1, -1, 1, 1 };

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

    if (streams[idx] == 1){
        int hasDownstream = 0;

        for (int i = 0; i < 8; i++){
            int nx = x + offsetX[i];
            int ny = y + offsetY[i];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nIdx = ny * width + nx;
                //check if neighbour flows into cell
                hasDownstream = 1;
                break;
            }
        }
        endpoints[idx] = (hasDownstream == 0) ? 1 : 0;
    } else {
        endpoints[idx] = 0;
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
    
    // Open flow accumulation dataset
    const char* input = argv[1];
    GDALDataset* flowDataset  = (GDALDataset*) GDALOpen(input, GA_ReadOnly);

    if (flowDataset == nullptr) {
        std::cerr << "Error opening Flow Accumulation file." << std::endl;
        return -1;
    }

    //get projection from input raster
    const char* projection = flowDataset->GetProjectionRef();
    if (projection == nullptr){
        std::cerr << "Error: Could not retrieve projection from DEM Dataset." << std::endl;
        GDALClose(flowDataset);
        return -1;
    }
    double geoTransform[6];

    if (flowDataset->GetGeoTransform(geoTransform) != CE_None){
        std::cerr << "Error reading geo-transform" << std::endl;
        GDALClose(flowDataset);
        return -1;
    }

    //create output raster for flow direction
    const char *outputFilename = argv[2];
    //Geotiff Driver
    GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    //32int Empty raster with same dims as input
    GDALDataset *streamDataset = poDriver->Create(outputFilename,
                                                    flowDataset->GetRasterXSize(),
                                                    flowDataset->GetRasterYSize(),
                                                    1, GDT_Int32, NULL);

    streamDataset->SetProjection(projection);
    streamDataset->SetGeoTransform(geoTransform);

    //Raster size to use with Malloc and device mem
    int width = flowDataset->GetRasterXSize();
    int height = flowDataset->GetRasterYSize();
    int *flowData = (int *)CPLMalloc(sizeof(int) * width * height);

    //populate flowData dynamically allocated memory
    CPLErr err = flowDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, flowData, width, height, GDT_Int32, 0, 0);
    if (err != CE_None){
        std::cerr << "Error reading Flow Accumulation data: " << CPLGetLastErrorMsg() << std::endl;
        CPLFree(flowData);
        return -1;
    }
    int *streamData = (int *)CPLMalloc(sizeof(int) * width * height);

    int *d_flowData;
    if (cudaMalloc(&d_flowData, sizeof(int) * width * height) != cudaSuccess){
        std::cerr << "Error allocation memory for Flow Accumulation data on device" << std::endl;
        CPLFree(flowData);
        CPLFree(streamData);
        return -1;
    }
    cudaError_t memcpy_err = cudaMemcpy(d_flowData, flowData, sizeof(int) * width * height, cudaMemcpyHostToDevice);
    if (memcpy_err != cudaSuccess){
        std::cerr << "Error copying flow accumulation data to device" << std::endl;
        CPLFree(flowData);
        CPLFree(streamData);
        cudaFree(d_flowData);
        return -1;
    }

    int* d_streamData;
    if (cudaMalloc(&d_streamData, sizeof(int) * width * height) != cudaSuccess){
        std::cerr << "Errory allocation stream data on device" << std::endl;
        CPLFree(flowData);
        CPLFree(streamData);
        cudaFree(d_flowData);
        return -1;
    }

    dim3 blockSize(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y); //dynamic grid allocation based on input size

    int threshold = 300;

    std::cout << "Kernel Launch" << std::endl;
    streamIdentification<<<gridSize, blockSize>>>(d_flowData, d_streamData, threshold, width, height);
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess){
        std::cerr << "Cuda kernel launch error: " << cudaGetErrorString(kernel_err) << std::endl;
        return -1;
    }
    cudaDeviceSynchronize();
    std::cout << "Kernel Finished" << std::endl;

    memcpy_err = cudaMemcpy(streamData, d_streamData, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
    if (memcpy_err != cudaSuccess){
        std::cerr << "Error copying stream data back from device" << std::endl;
        CPLFree(flowData);
        CPLFree(streamData);
        cudaFree(d_flowData);
        cudaFree(d_streamData);
        return -1;
    }

    err = streamDataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height, streamData, width, height, GDT_Int32, 0, 0);
    if (err != CE_None){
        std::cerr << "Error writing stream data" << std::endl;
        CPLFree(flowData);
        CPLFree(streamData);
        cudaFree(d_flowData);
        return -1;
    }

    CPLFree(flowData);
    CPLFree(streamData);
    cudaFree(d_flowData);
    GDALClose(flowDataset);
    GDALClose(streamDataset);

    std::cout << "Stream data calculated and saved to: " << outputFilename << std::endl;
    return 0;
}