#include <iostream>
#include <gdal_priv.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define THREADCELLS 4
#define BLOCK_SIZE 16

__constant__ int offsetX[9] = {0, -1,  0,  1,  1,  1,  0, -1, -1};
__constant__ int offsetY[9] = {0, -1, -1, -1,  0,  1,  1,  1,  0};
                            // 0,  1,  2,  3,  4,  5,  6,  7,  8 


//Each thread handles a THREADCELLS x THREADCELLS neighbourhood
__global__ void flowAccumKernel(int* gpuAccum, int* gpuOldFlow, int* gpuNewFlow, const int * flowDir, int* gpuStop, const int N, const int M){
    int i = THREADCELLS * (blockIdx.y * blockDim.y + threadIdx.y);
    int j = THREADCELLS * (blockIdx.x * blockDim.x + threadIdx.x);

    for (int r = i; r < i + THREADCELLS && r < N; r++){
        for (int s = j; s < j + THREADCELLS && s < M; s++){
            int idx = r * M + s;
            int curFlow = gpuOldFlow[idx];
            if (curFlow > 0){
                gpuOldFlow[idx] = 0;
                int cellFlowDir = flowDir[idx]; 
                if (cellFlowDir == 0) continue;
                int newR = r + offsetY[cellFlowDir];
                int newS = s + offsetX[cellFlowDir];

                int valid = (newR >= 0 && newR < N && newS >= 0 && newS < M);
                int new_idx = newR * M + newS;
                atomicAdd(&gpuNewFlow[new_idx], valid * curFlow);
                atomicAdd(&gpuAccum[new_idx], valid * curFlow);
                atomicOr(gpuStop, valid * 1);
            } 
        }
    }
}

int main(int argc, char* argv[]){
    // FlowDir, FlowAccum
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

    //Get projection info
    const char* projection = D8Dataset->GetProjectionRef();
    if (projection == nullptr){
        std::cerr<< "Error: Could not retrieve projection information" << std::endl;
        GDALClose(D8Dataset);
        return -1;
    }

    double geoTransform[6];
    if (D8Dataset->GetGeoTransform(geoTransform) != CE_None){
        std::cerr << "Error reading geo-transform" << std::endl;
        GDALClose(D8Dataset);
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

    //Set projection of output
    flowAccumDataset->SetProjection(projection);
    flowAccumDataset->SetGeoTransform(geoTransform);

    //Raster size to use with Malloc and device mem
    int width = D8Dataset->GetRasterXSize();
    int height = D8Dataset->GetRasterYSize();
    int *flowDir = (int *)CPLMalloc(sizeof(int) * width * height);

    //populate demData dynamically allocated memory
    CPLErr err = D8Dataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, flowDir, width, height, GDT_Int32, 0, 0);
    if (err != CE_None){
        std::cerr << "Error reading DEM data: " << CPLGetLastErrorMsg() << std::endl;
        return -1;
    }

    //allocating device mem
    int *d_oldFlow, *d_newFlow, *d_flowDir, *d_accum, *d_stopFlag;
    
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
    if (cudaMalloc(&d_stopFlag, sizeof(int)) != cudaSuccess) {
        std::cerr << "Error allocating memory for Stop Flag on device" << std::endl;
        return -1;
    }
        
    //copy flow direction data to device
    cudaError_t memcpy_err_flowDir = cudaMemcpyAsync(d_flowDir, flowDir, sizeof(int) * width * height, cudaMemcpyHostToDevice);
    if (memcpy_err_flowDir != cudaSuccess){
        std::cerr << "Error copying data to device: " << cudaGetErrorString(memcpy_err_flowDir) << std::endl;
        return -1;
    }
    int* hostOldFlow = new int[width * height];
    int* hostNewFlow = new int [width*height];
    for (int i = 0; i < width * height; ++i) hostOldFlow[i] = 1, hostNewFlow[i] = 0;

    cudaMemcpyAsync(d_oldFlow, hostOldFlow, width * height * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_newFlow, hostNewFlow, width * height * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemsetAsync(d_accum, 0, sizeof(int) * width * height);   

    //define grid and block size
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int x = 0;
    int *stopFlag = new int(0);

    do{
        printf("Kernel iteration: %d\n", x++ + 1);
        *stopFlag = 0;
        cudaMemcpyAsync(d_stopFlag, stopFlag, sizeof(int), cudaMemcpyHostToDevice);

        flowAccumKernel<<<gridSize, blockSize, 0, stream>>>(d_accum, d_oldFlow, d_newFlow, d_flowDir, d_stopFlag, height, width);

        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr != cudaSuccess){
            std::cerr << "Error launching kernel: " << cudaGetErrorString(kernelErr) << std::endl;
            return -1;
        }
        cudaMemcpyAsync(stopFlag, d_stopFlag, sizeof(int), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(stream);

        int *temp = d_oldFlow;
        d_oldFlow = d_newFlow;
        d_newFlow = temp;
        cudaMemsetAsync(d_newFlow, 0, sizeof(int) * width * height);

        //early termination to prevent infite looping if error in flow direction data
        if (x == 100000){
            break;
        }
    } while (*stopFlag != 0);

    int *hostflowAccumulationData = (int *)CPLMalloc(sizeof(int) * width * height);
    cudaMemcpyAsync(hostflowAccumulationData, d_accum, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);

    err = flowAccumDataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height,
        hostflowAccumulationData, width, height, GDT_Int32, 0, 0);
    if (err != CE_None) {
        std::cerr << "Error writing flow accumulation data: " << CPLGetLastErrorMsg() << std::endl;
        return -1;
    }
    
    //perform cleanup
    cudaFree(d_oldFlow); cudaFree(d_newFlow); cudaFree(d_flowDir);
    cudaFree(d_stopFlag); cudaFree(d_accum);
    CPLFree(flowDir);
    GDALClose(D8Dataset); GDALClose(flowAccumDataset);
    delete[] hostOldFlow; delete[] hostNewFlow;
    cudaStreamDestroy(stream);
    return 0;
}