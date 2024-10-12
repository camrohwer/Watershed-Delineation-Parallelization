import subprocess
import time
import os
import sys

def status(i, t, bar_len = 50):
    percent = float(i) / t
    cur = int(round(percent * bar_len))
    bar = '#' * cur + '-' * (bar_len - cur)

    sys.stdout.write(f"\r[{bar}] {int(percent * 100)}%")
    sys.stdout.flush()

def run_executable(command):
    start_time = time.time()
    try:
        # Call the executable
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elapsed_time = time.time() - start_time
        return elapsed_time, result.stdout.decode(), result.stderr.decode()
    except subprocess.CalledProcessError as e:
        return None, e.stdout.decode(), e.stderr.decode()

dem_dir = "./DEMs"
output_dir = "./DEMs/Output"
file_list = [
    f for f in os.listdir(dem_dir)
    if os.path.isfile(os.path.join(dem_dir, f)) and not f.startswith('.')
]

num_runs = 1
total_execs = num_runs * len(file_list) * 2
print(f"Number of Runs {num_runs}")

i_runtimes = []
p_runtimes = []

i = 0
for _ in range(num_runs):
    for file in file_list:
        raster_cell = file.replace(".tif", "")
        input = os.path.join(dem_dir, file)
        output_name = raster_cell + "_flow_direction"

        i_output = os.path.join(output_dir, output_name + "_iterative")
        time_i, output_i, error_i = run_executable(["./build/bin/flow_direction_iterative", input, i_output])
        if time_i is not None:
            i_runtimes.append(time_i)
        else:
            print(f"Iterative Error: {error_i}")
            break

        i += 1
        status(i, total_execs)

        p_output = os.path.join(output_dir, output_name + "_parallel")
        time_p, output_p, error_p = run_executable(["./build/bin/flow_direction_parallel", input, p_output])
        if time_p is not None:
            p_runtimes.append(time_p)
        else:
            print(f"Parallel Error: {error_p}")
            break


        i += 1
        status(i, total_execs)

print("")
    
avg_i = sum(i_runtimes) / len(i_runtimes)
print(f"Average runtime of Iterative solution: {avg_i:.4f} seconds")

avg_p = sum(p_runtimes) / len(p_runtimes)
print(f"Average runtime of Parallel solution: {avg_p:.4f} seconds")
