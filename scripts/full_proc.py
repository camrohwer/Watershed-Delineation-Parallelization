import subprocess
import time
from datetime import datetime
import os
import sys
from fill_depression import fill_depressions

#python3 ./scripts/full_proc.py ./DEMs/092H.tif 

output_directory = "./DEMs/Output"

def run_executable(command):
    start_time = time.time()
    try:
        # Call the executable
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elapsed_time = time.time() - start_time
        return elapsed_time, result.stdout.decode(), result.stderr.decode()
    except subprocess.CalledProcessError as e:
        return None, e.stdout.decode(), e.stderr.decode()

def main():
    dem_path = sys.argv[1] #get input path from arg
    filename, ext = os.path.splitext(os.path.basename(dem_path)) #extract input filename without ext

    filled_path = preprocess(dem_path, filename)
    flow_dir_path = flow_dir(filled_path, filename)
    flow_accum_path = flow_accum(flow_dir_path, filename)
    endpoints_path = stream_identification(flow_accum_path, flow_dir_path, filename)
    watershed_path = watershed_delineation(flow_dir_path, endpoints_path, filename)

    print(f"Delineated watershed written to {watershed_path}")

def preprocess(dem_path, filename):
    #uses whitebox to fill depressions
    output = os.path.join(output_directory, filename + "_filled.tif")

    start_time = datetime.now()
    fill_depressions(dem_path, output)
    end_time = datetime.now()

    print(f"Depression & Pit filling took {end_time - start_time}")

    return output

def flow_dir(input, filename):
    output = os.path.join(output_directory, filename + "_dir.tif")
    time_dir, output_dir, error_dir = run_executable(["./build/bin/flow_dir", input, output])

    if time_dir is not None:
        print(f"Flow direction calculation took {time_dir} seconds")
    else:
        print(f"Flow direction error: {error_pit}")

    return output

def flow_accum(input, filename):
    output = os.path.join(output_directory, filename + "_accum.tif")
    time_accum, output_accum, error_accum = run_executable(["./build/bin/flow_accum", input, output])

    if time_accum is not None:
        print(f"Flow accumulation calculation took {time_accum} seconds")
    else:
        print(f"Flow accumulation error: {error_accum}")

    return output

def stream_identification(flow_accum, flow_dir, filename):
    output1 = os.path.join(output_directory, filename + "_streams.tif")
    output2 = os.path.join(output_directory, filename + "_endpoints.tif")
    time_stream, output_stream, error_stream = run_executable(["./build/bin/stream_ident", flow_accum, flow_dir, output1, output2])

    if time_stream is not None:
        print(f"Stream identification took {time_stream} seconds")
    else:
        print(f"Stream identification error: {error_stream}")

    return output2

def watershed_delineation(flow_dir, endpoints, filename):
    output = os.path.join(output_directory, filename + "_watersheds.tif")
    time_watershed, output_watershed, error_watershed = run_executable(["./build/bin/watershed_delin", flow_dir, endpoints, output])

    if time_watershed is not None:
        print(f"Watershed delineation took {time_watershed} seconds")
    else:
        print(f"Watershed delineation error: {error_watershed}")

    return output

if __name__ == "__main__":
    main()