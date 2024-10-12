#include <iostream>
#include <gdal_priv.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

const int FLOW_NODATA = -1;

__global__ void flowDirectionKernel(float* dem, int* flow_dir, int width, int height) {
    //unique thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < 1 || x >= width || y < 1 || y >= height - 1) {
        return; //skip boundary pixels
    }
    
    float centre = dem[idx]; //get dem value at current pixel
    float lowest = centre;
    int dir = FLOW_NODATA;

    for (int dy = -1; dy <= 1; dy++){
        for (int dx = -1; dx <= 1; dx++){
            if (dx == 0 && dy == 0) continue; //skip current pixel

            //find neightbours coords
            int nx = x + dx;
            int ny = y + dy;

            //check only valid pixels
            if (nx >= 0 && nx < width && ny >= 0 && ny < height){
                float n = dem[ny * width + nx]; //get neighbours value

                if (n < lowest){
                    lowest = n;
                    // Determine the direction based on the neighbor's position
                    if (dy == -1 && dx == -1) dir = 1;  // North-West
                    else if (dy == -1 && dx == 0) dir = 2; // North
                    else if (dy == -1 && dx == 1) dir = 3; // North-East
                    else if (dy == 0 && dx == 1) dir = 4;  // East
                    else if (dy == 1 && dx == 1) dir = 5;  // South-East
                    else if (dy == 1 && dx == 0) dir = 6;  // South
                    else if (dy == 1 && dx == -1) dir = 7; // South-West
                    else if (dy == 0 && dx == -1) dir = 8; // West
                }
            }
        }
    }
    //printf("%d\n",dir);
    if (dir != FLOW_NODATA){
        flow_dir[idx] = dir;
    }
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

    if (demDataset == NULL) {
        std::cerr << "Error opening DEM file." << std::endl;
        return -1;
    }

    //create output raster for flow direction
    const char *outputFilename = argv[2];
    //const char *outputFilename = "../../DEMs/Output/parallel_flow_direction.tif"; //TODO should fix abs paths
    //Geotiff Driver
    GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    //32int Empty raster with same dims as input
    GDALDataset *flowDirDataset = poDriver->Create(outputFilename,
                                                    demDataset->GetRasterXSize(),
                                                    demDataset->GetRasterYSize(),
                                                    1, GDT_Int32, NULL);

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
    float *d_demData;
    int *d_flowDirData;

    cudaMalloc(&d_demData, sizeof(float) * width * height);
    if (cudaMalloc(&d_demData, sizeof(float) * width * height) != cudaSuccess) {
        std::cerr << "Error allocating memory for DEM on device." << std::endl;
        return -1;
    }

    cudaMalloc(&d_flowDirData, sizeof(int) * width * height);
    if (cudaMalloc(&d_flowDirData, sizeof(int) * width * height) != cudaSuccess) {
        std::cerr << "Error allocating memory for flow direction on device." << std::endl;
        cudaFree(d_demData); // Free already allocated memory
        return -1;
    }

    //copy DEM data to device
    cudaError_t memcpy_err = cudaMemcpy(d_demData, demData, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    if (memcpy_err != cudaSuccess){
        std::cerr << "Error copying data to device: " << cudaGetErrorString(memcpy_err) << std::endl;
        return -1;
    }

    /* 
    int dim_x = std::atoi(argv[3]);
    int dim_y = std::atoi(argv[4]);
    dim3 blockSize(dim_x,dim_y);
    */

    //define grid and block size
    dim3 blockSize(8,8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    flowDirectionKernel<<<gridSize, blockSize>>>(d_demData, d_flowDirData, width, height);
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess){
        std::cerr << "Cuda kernel launch error: " << cudaGetErrorString(kernel_err) << std::endl;
        return -1;
    }

    // Copy flow direction data back to host
    int *flowDirData = (int *)CPLMalloc(sizeof(int) * width * height);
    cudaDeviceSynchronize();
    cudaMemcpy(flowDirData, d_flowDirData, sizeof(int) * width * height, cudaMemcpyDeviceToHost);

    // Write flow direction data to the output dataset
    err = flowDirDataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height,
                                                     flowDirData, width, height, GDT_Int32, 0, 0);
    if (err != CE_None) {
        std::cerr << "Error writing flow direction data: " << CPLGetLastErrorMsg() << std::endl;
        return -1;
    }

    // Cleanup
    CPLFree(demData);
    CPLFree(flowDirData);
    cudaFree(d_demData);
    cudaFree(d_flowDirData);
    GDALClose(demDataset);
    GDALClose(flowDirDataset);

    std::cout << "Flow direction calculated and saved to " << outputFilename << std::endl;

    return 0;
}
