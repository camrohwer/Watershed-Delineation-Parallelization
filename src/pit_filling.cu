#include <iostream>
#include <gdal_priv.h>
#include <cuda_runtime.h>
#include <float.h>

#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
#define HEIGHT_CONST 0.1f

//loop offsets for checking neighbours
__constant__ int offsetX[8] = { -1, 0, 1, 0, -1, 1, 1, -1 };
__constant__ int offsetY[8] = { 0, -1, 0, 1, -1, -1, 1, 1 };

__global__ void identifyAndFillPits(float* dem, int* numPits, int width, int height, float hc){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return; // skip boundary 

    __shared__ float sharedDem[BLOCK_DIM_Y + 2][BLOCK_DIM_X + 2];

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    //load shared dem values 
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
    __syncthreads();

    float curElev = static_cast<float>(sharedDem[ty][tx]);
    bool isPit = true;
    float lowestNeighbour = FLT_MAX;

    for (int i = 0; i < 8; i++){
        int neighbourElev = sharedDem[ty + offsetY[i]][tx + offsetX[i]];

        if (neighbourElev <= curElev){
            isPit = false;
            break;
        }else{
            if (neighbourElev < lowestNeighbour){
                lowestNeighbour = neighbourElev;
            }
        }
    }

    //keep count of pits in local variable, and only have first thread perform atomic to update global count.
    //reduces atomics to global
    __shared__ int localPits;
    if (threadIdx.x == 0 && threadIdx.y == 0) localPits = 0;
    __syncthreads();

    if (isPit){
        atomicAdd(&localPits, 1);
        dem[y * width + x] = static_cast<float>(lowestNeighbour) + hc;
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0){
        atomicAdd(numPits, localPits);
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

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //allocating device mem
    float *d_dem;
    cudaMalloc(&d_dem, sizeof(float) * width * height);

    int* d_numPits;
    cudaMalloc(&d_numPits, sizeof(int));
    cudaMemset(d_numPits, 0, sizeof(int));

    int numPits = 0;
    
    //copy DEM data to device
    cudaMemcpyAsync(d_dem, demData, sizeof(float) * width * height, cudaMemcpyHostToDevice, stream);

    //define grid and block size
    dim3 blockSize(BLOCK_DIM_X,BLOCK_DIM_Y);
    dim3 gridSize((width + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    identifyAndFillPits<<<gridSize, blockSize, 0, stream>>>(d_dem, d_numPits, width, height, HEIGHT_CONST);

    // copy pit count from device to host
    cudaMemcpyAsync(&numPits, d_numPits, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(demData, d_dem, sizeof(float) * width * height, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // Write pit filling data to the output dataset
    err = pitFillingDataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height, demData, width, height, GDT_Float32, 0, 0);
    if (err != CE_None) {
        std::cerr << "Error writing pit filling data: " << CPLGetLastErrorMsg() << std::endl;
        return -1;
    }

    cudaFree(d_dem);
    cudaFree(d_numPits);
    CPLFree(demData);
    GDALClose(demDataset);
    GDALClose(pitFillingDataset);
    cudaStreamDestroy(stream);

    std::cout << "Number of pits filled: " << numPits << std::endl;
    std::cout << "Pit filling completed and output written to " << outputFilename << std::endl;
    return 0;
}