from osgeo import gdal
import numpy as np

#NOT WORKING YET
#NEED TO FIX EDGE CASES IN FLOW DIRECITON CALCULATION BEFORE FINISH IMPLEMENTING

def load_raster(path):
    dataset = gdal.Open(path)
    band = dataset.GetRasterBand(1) #band 1 is default raster band
    array = band.ReadAsArray() #raster as np array
    return array, dataset

#open rasters
raster_iterative_path = "./DEMs/Output/iterative_flow_direction.tif"
raster_iterative, dataset_iterative = load_raster(raster_iterative_path)

raster_parallel_path = "./DEMs/Output/parallel_flow_direction.tif"
raster_parallel, dataset_parallel = load_raster(raster_parallel_path)

#check dimensions
if raster_parallel.shape != raster_iterative.shape:
    raise ValueError("Rasters have different dimesnsions")

difference = raster_iterative != raster_parallel
num_differences = np.sum(difference)
print(f"Number of differing pixels: {num_differences}")

if num_differences > 0:
    indices_of_difference = np.argwhere(difference)
    for index in indices_of_difference[:1000]:  # Show first 5 differences
        print(f"Index {index}: Iterative = {raster_iterative[tuple(index)]}, Parallel = {raster_parallel[tuple(index)]}")

driver = gdal.GetDriverByName("GTiff")
output_path = "./DEMs/Output/diff.tif" 
out_raster = driver.Create(output_path, dataset_iterative.RasterXSize, dataset_iterative.RasterYSize, 1, gdal.GDT_Byte)

# set geo transform and projection for output
out_raster.SetGeoTransform(dataset_iterative.GetGeoTransform())
out_raster.SetProjection(dataset_iterative.GetProjection()) 

#write diff to raster
out_band = out_raster.GetRasterBand(1)
out_band.WriteArray(difference.astype(np.uint8))
out_band.SetNoDataValue(0)
out_band.FlushCache()

#close datasets
dataset_iterative = None
dataset_parallel = None
out_raster = None

print(f"Diff raster written to {output_path}")
