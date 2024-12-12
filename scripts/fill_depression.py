import whitebox_workflows as wbw
import sys
import os
wbe = wbw.WbEnvironment()

def fill_depressions(input, output):
    input_dem = wbe.read_raster(input)
    dem_depfill = wbe.fill_depressions(dem=input_dem, flat_increment=0.001)
    wbe.write_raster(dem_depfill, output, compress=True)

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    fill_depressions(input_path, output_path)
    