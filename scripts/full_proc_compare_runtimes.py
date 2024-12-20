import subprocess
import time
from datetime import datetime
import os
import sys
import csv
import matplotlib.pyplot as plt
from fill_depression import fill_depressions

# Constants
output_directory = "./DEMs/Output"
runs_per_dem = 5

# List of DEMs to process
dem_list = [
    "./DEMs/092H.tif",
 ]
# Helper function

def run_executable(command):
    start_time = time.time()
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elapsed_time = time.time() - start_time
        return elapsed_time, result.stdout.decode(), result.stderr.decode()
    except subprocess.CalledProcessError as e:
        return None, e.stdout.decode(), e.stderr.decode()

# Processing functions

def preprocess(dem_path, filename):
    output = os.path.join(output_directory, filename + "_filled.tif")

    if not os.path.exists(output):
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
        print(f"Flow direction error: {error_dir}")

    return time_dir, output

def flow_accum(input, filename):
    output = os.path.join(output_directory, filename + "_accum.tif")
    time_accum, output_accum, error_accum = run_executable(["./build/bin/flow_accum", input, output])

    if time_accum is not None:
        print(f"Flow accumulation calculation took {time_accum} seconds")
    else:
        print(f"Flow accumulation error: {error_accum}")

    return time_accum, output

def stream_identification(flow_accum, flow_dir, filename):
    output1 = os.path.join(output_directory, filename + "_streams.tif")
    output2 = os.path.join(output_directory, filename + "_endpoints.tif")
    time_stream, output_stream, error_stream = run_executable(["./build/bin/stream_ident", flow_accum, flow_dir, output1, output2])

    if time_stream is not None:
        print(f"Stream identification took {time_stream} seconds")
    else:
        print(f"Stream identification error: {error_stream}")

    return time_stream, output2

def watershed_delineation(flow_dir, endpoints, filename):
    output = os.path.join(output_directory, filename + "_watersheds.tif")
    time_watershed, output_watershed, error_watershed = run_executable(["./build/bin/watershed_delin", flow_dir, endpoints, output])

    if time_watershed is not None:
        print(f"Watershed delineation took {time_watershed} seconds")
    else:
        print(f"Watershed delineation error: {error_watershed}")

    return time_watershed, output

# Main function

def main():
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # CSV setup
    csv_file = os.path.join(output_directory, "runtimes.csv")
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["DEM", "Run", "Flow Direction", "Flow Accumulation", "Stream Identification", "Watershed Delineation"])

        # Process each DEM
        for dem_path in dem_list:
            filename, ext = os.path.splitext(os.path.basename(dem_path))
            filled_path = preprocess(dem_path, filename)

            for run in range(1, runs_per_dem + 1):
                print(f"Processing {filename}, Run {run}")

                # Steps excluding depression filling
                time_dir, flow_dir_path = flow_dir(filled_path, filename)
                time_accum, flow_accum_path = flow_accum(flow_dir_path, filename)
                time_stream, endpoints_path = stream_identification(flow_accum_path, flow_dir_path, filename)
                time_watershed, watershed_path = watershed_delineation(flow_dir_path, endpoints_path, filename)

                # Write runtimes to CSV
                writer.writerow([filename, run, time_dir, time_accum, time_stream, time_watershed])

    # Plot runtimes
    plot_runtimes(csv_file)

def plot_runtimes(csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)

    steps = ["Flow Direction", "Flow Accumulation", "Stream Identification", "Watershed Delineation"]

    # Bar chart comparison
    avg_runtimes = {}

    for step in steps:
        avg_runtimes[step] = df[step].mean()

    plt.figure(figsize=(10, 6))
    plt.bar(avg_runtimes.keys(), avg_runtimes.values())
    plt.title("Average Runtime Comparison of Steps")
    plt.xlabel("Steps")
    plt.ylabel("Average Time (s)")
    plt.savefig(os.path.join(output_directory, "average_runtime_comparison.png"))
    plt.show()

if __name__ == "__main__":
    main()
