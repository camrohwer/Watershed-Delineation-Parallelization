import subprocess
import time
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from fill_depression import fill_depressions  # Assuming this exists

output_directory = "./DEMs/Output"
results_directory = "./Results"  # Directory for saving graph and runtime stats
#input_files = ['092B.tif', '092F.tif', '092G.tif', '092H.tif', '092I.tif']
input_files = ['092B_filled.tif', '092F_filled.tif', '092G_filled.tif', '092H_filled.tif', '092I_filled.tif']
input_directory = "./DEMs"

# Only include the necessary processes in stats
process_runtimes = {
    "Flow Direction": [],
    "Flow Accumulation": [],
    "Stream Identification": []
}

def run_executable(command):
    start_time = time.time()
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elapsed_time = time.time() - start_time
        return elapsed_time, result.stdout.decode(), result.stderr.decode()
    except subprocess.CalledProcessError as e:
        return None, e.stdout.decode(), e.stderr.decode()

'''
def preprocess(dem_path, filename):
    output = os.path.join(output_directory, filename + "_filled.tif")
    fill_depressions(dem_path, output)  # No time tracking
    return output
'''

def flow_dir(input, filename):
    output = os.path.join(output_directory, filename + "_dir.tif")
    elapsed_time, _, _ = run_executable(["./build/bin/flow_dir", input, output])
    return output, elapsed_time

def flow_accum(input, filename):
    output = os.path.join(output_directory, filename + "_accum.tif")
    elapsed_time, _, _ = run_executable(["./build/bin/flow_accum", input, output])
    return output, elapsed_time

def stream_identification(flow_accum, flow_dir, filename):
    output1 = os.path.join(output_directory, filename + "_streams.tif")
    output2 = os.path.join(output_directory, filename + "_endpoints.tif")
    elapsed_time, _, _ = run_executable(["./build/bin/stream_ident", flow_accum, flow_dir, output1, output2])
    return output2, elapsed_time

def watershed_delineation(flow_dir, endpoints, filename):
    output = os.path.join(output_directory, filename + "_watershed.tif")
    elapsed_time, _, _ = run_executable(["./build/bin/watershed", flow_dir, endpoints, output])
    return output

def save_results(average_runtimes):
    # Ensure results directory exists
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    # Save the graph
    graph_path = os.path.join(results_directory, "process_runtimes.png")
    plt.figure(figsize=(10, 6))
    plt.bar(average_runtimes.keys(), average_runtimes.values(), color="skyblue")
    plt.xlabel("Processes")
    plt.ylabel("Average Runtime (seconds)")
    plt.title("Average Runtime of Processes for Multiple Files")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(graph_path)
    print(f"Graph saved to {graph_path}")

    # Save average runtimes to a text file
    results_path = os.path.join(results_directory, "average_runtimes.txt")
    with open(results_path, "w") as file:
        file.write("Average Runtimes:\n")
        for process, avg_time in average_runtimes.items():
            file.write(f"{process}: {avg_time:.2f} seconds\n")
    print(f"Average runtimes saved to {results_path}")

def main():
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file in input_files:
        dem_path = os.path.join(input_directory, file)
        filename, _ = os.path.splitext(os.path.basename(dem_path))

        print(f"\nProcessing file: {file}")

        '''
        # Preprocessing step (Depression Filling, excluded from stats)
        filled_path = preprocess(dem_path, filename)
        print("Depression Filling completed (time not recorded).")
        '''

        # Step 1: Flow Direction
        #flow_dir_path, time_dir = flow_dir(filled_path, filename)
        flow_dir_path, time_dir = flow_dir(dem_path, filename)
        process_runtimes["Flow Direction"].append(time_dir)
        print(f"Flow Direction: {time_dir:.2f} seconds")

        # Step 2: Flow Accumulation
        flow_accum_path, time_accum = flow_accum(flow_dir_path, filename)
        process_runtimes["Flow Accumulation"].append(time_accum)
        print(f"Flow Accumulation: {time_accum:.2f} seconds")

        # Step 3: Stream Identification
        endpoints_path, time_stream = stream_identification(flow_accum_path, flow_dir_path, filename)
        process_runtimes["Stream Identification"].append(time_stream)
        print(f"Stream Identification: {time_stream:.2f} seconds")

    # Compute average runtimes
    average_runtimes = {process: sum(times) / len(times) for process, times in process_runtimes.items()}

    # Save results to the specified directory
    save_results(average_runtimes)

if __name__ == "__main__":
    main()
