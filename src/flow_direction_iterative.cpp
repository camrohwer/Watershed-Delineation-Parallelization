#include <gdal_priv.h>
#include <cpl_conv.h> // for CPLMalloc()

#include <iostream>
#include <vector>
#include <cmath>

//raster from https://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdem_mnec/

const int FLOW_NODATA = -1;

// Function to calculate flow direction using D8 method
void calculateFlowDirection(GDALDataset *demDataset, GDALDataset *flowDirDataset) {
    int width = demDataset->GetRasterXSize();
    int height = demDataset->GetRasterYSize();
    float *demData = (float *)CPLMalloc(sizeof(float) * width * height);
    float *flowDirData = (float *)CPLMalloc(sizeof(float) * width * height);

    // Read DEM data
    demDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height,
                                            demData, width, height, GDT_Float32, 0, 0);

     // Initialize flow direction data
    for (int i = 0; i < width * height; ++i) {
        flowDirData[i] = FLOW_NODATA;  // Initialize with nodata value
    }

    // Calculate flow direction
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float center = demData[y * width + x];

            // Check neighbors
            float lowest = center;
            int direction = FLOW_NODATA;

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue; // Skip the center pixel

                    float neighbor = demData[(y + dy) * width + (x + dx)];

                    if (neighbor < lowest) {
                        lowest = neighbor;
                        // Determine the direction based on the neighbor's position
                        if (dy == -1 && dx == -1) direction = 1;  // North-West
                        else if (dy == -1 && dx == 0) direction = 2; // North
                        else if (dy == -1 && dx == 1) direction = 3; // North-East
                        else if (dy == 0 && dx == 1) direction = 4;  // East
                        else if (dy == 1 && dx == 1) direction = 5;  // South-East
                        else if (dy == 1 && dx == 0) direction = 6;  // South
                        else if (dy == 1 && dx == -1) direction = 7; // South-West
                        else if (dy == 0 && dx == -1) direction = 8; // West
                    }
                }
            }

            // Assign the flow direction
            flowDirData[y * width + x] = direction;
        }
    }

    // Write flow direction data to the output dataset
    flowDirDataset->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, width, height,
                                                flowDirData, width, height, GDT_Int32, 0, 0);

    // Cleanup
    CPLFree(demData);
    CPLFree(flowDirData);
}

int main(int argc, const char* argv[]) {
    if (argc != 2){
        std::cout << "Please provide a filename for input raster" << std::endl;
        return -1;
    }
    GDALAllRegister();

    //Open DEM
    const char* input = argv[1];
    GDALDataset* demDataset  = (GDALDataset*) GDALOpen(input, GA_ReadOnly);

    if (demDataset == NULL) {
        std::cout << "Failed to open DEM" << std::endl;
        return -1;

    }

    //create output raster for flow direction

    const char *outputFilename = "flow_direction.tif";
    GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
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