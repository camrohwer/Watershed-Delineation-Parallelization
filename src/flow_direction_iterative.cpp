#include <gdal_priv.h>
#include <cpl_conv.h> // for CPLMalloc()

#include <iostream>
#include <vector>
#include <cmath>

//raster from https://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdem_mnec/

const int FLOW_NODATA = -1;

//valid neighbour check
bool isValidNeighbor(int x, int y, int width, int height) {
    return (x >= 0 && x < width && y >= 0 && y < height);
}

// Function to calculate flow direction using D8 method
void calculateFlowDirection(GDALDataset *demDataset, GDALDataset *flowDirDataset) {
    int width = demDataset->GetRasterXSize();
    int height = demDataset->GetRasterYSize();
    //Allocate space for Input raster as vector 
    float *demData = (float *)CPLMalloc(sizeof(float) * width * height);
    //Allocate space for flow direction vector
    int *flowDirData = (int *)CPLMalloc(sizeof(int) * width * height);

    // Read DEM data from demDataset to demData
    CPLErr err;
    err = demDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height,
                                            demData, width, height, GDT_Float32, 0, 0);
    if (err != CE_None){
        std::cerr << "Error reading DEM data: " << CPLGetLastErrorMsg() << std::endl;
        exit(EXIT_FAILURE);
    }

     // Initialize flow direction data
    for (int i = 0; i < width * height; ++i) {
        flowDirData[i] = FLOW_NODATA;  // Initialize with nodata value
    }

    // Calculate flow direction
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float center = demData[y * width + x];
            //if center pixel is invalid we dont need to check neightbours
            if (center == FLOW_NODATA) continue;
            // Check neighbors
            float lowest = center;
            int dir = FLOW_NODATA;

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue; // Skip the center pixel

                    float neighbor = demData[(y + dy) * width + (x + dx)];

                    if (neighbor < lowest && neighbor != FLOW_NODATA) {
                        lowest = neighbor;
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

            // Assign the flow direction
            //printf("%d\n", dir);
            if (dir != FLOW_NODATA){
                flowDirData[y * width + x] = dir;
            }
        }
    }

    // Write flow direction data to the output dataset
    err = flowDirDataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height,
                                                flowDirData, width, height, GDT_Int32, 0, 0);

    if (err != CE_None){
        std::cerr << "Error reading DEM data: " << CPLGetLastErrorMsg() << std::endl;
        exit(EXIT_FAILURE);
    }

    // Cleanup
    CPLFree(demData);
    CPLFree(flowDirData);
}

int main(int argc, const char* argv[]) {
    //check that input file is passed as arg
    if (argc != 2){
        std::cout << "Please provide a filepath for input raster" << std::endl;
        return -1;
    }
    // Register all drivers to be able to open Raster data
    GDALAllRegister();

    //Open DEM Dataset
    const char* input = argv[1];
    GDALDataset* demDataset  = (GDALDataset*) GDALOpen(input, GA_ReadOnly);

    if (demDataset == NULL) {
        std::cerr << "Failed to open DEM" << std::endl;
        return -1;

    }

    //create output raster for flow direction
    const char *outputFilename = "../../DEMs/Output/iterative_flow_direction.tif"; //TODO fix abs paths
    //Get Geotiff driver
    GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    //Create 32Int empty raster with same dimensions as input
    GDALDataset *flowDirDataset = poDriver->Create(outputFilename,
                                                    demDataset->GetRasterXSize(),
                                                    demDataset->GetRasterYSize(),
                                                    1, GDT_Int32, NULL);

     // Calculate flow direction
    calculateFlowDirection(demDataset, flowDirDataset);

    // Cleanup
    GDALClose(demDataset);
    GDALClose(flowDirDataset);

    std::cout << "Flow direction calculated and saved to " << outputFilename << std::endl;

    return 0;
}