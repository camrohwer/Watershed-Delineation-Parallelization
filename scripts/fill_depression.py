import whitebox_workflows as wbw
import sys
import os
wbe = wbw.WbEnvironment()

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    script_dir = os.path.dirname(os.path.realpath(__file__))  
    source_path = os.path.join(script_dir, "../DEMs", input_file)  
    dest_dir = os.path.join(script_dir, "../DEMs/Output")
    dest_path = os.path.join(dest_dir, output_file)

    input_dem = wbe.read_raster(source_path)
    dem_depfill = wbe.fill_depressions(dem=input_dem, flat_increment=0.001)
    wbe.write_raster(dem_depfill, dest_path, compress=True)


