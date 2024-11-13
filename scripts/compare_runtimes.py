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
print(file_list)

num_runs = 5
total_execs = num_runs * len(file_list) * 4
print(f"Number of Runs {num_runs}")

pit_runtimes = []
dir_runtimes = []
pit_dir_runtimes = []
accum_runtimes = []

i = 0
for _ in range(num_runs):
    for file in file_list:
        raster_cell = file.replace(".tif", "")
        input = os.path.join(dem_dir, file)
        output_name = raster_cell

        pit_output = os.path.join(output_dir, output_name + "_filled.tif")
        time_pit, output_pit, error_pit = run_executable(["./build/bin/pit_filling", input, pit_output])
        if time_pit is not None:
            pit_runtimes.append(time_pit)
        else:
            print(f"Pit Error: {error_pit}")
            break

        i += 1
        status(i, total_execs)

        dir_output = os.path.join(output_dir, output_name + "_dir.tif")
        time_dir, output_fdir, error_dir = run_executable(["./build/bin/flow_direction_parallel", input, dir_output])
        if time_dir is not None:
            dir_runtimes.append(time_dir)
        else:
            print(f"Flow Direction Error: {error_dir}")
            break

        i += 1
        status(i, total_execs)

        pit_dir_output = os.path.join(output_dir, output_name + "_filled_dir.tif")
        time_pit_dir, output_pit_dir, error_pit_dir = run_executable(["./build/bin/pit_filling_flow_direction", input, pit_dir_output])
        if time_pit_dir is not None:
            pit_dir_runtimes.append(time_pit_dir)
        else:
            print(f"Flow Direction Error: {error_pit_dir} \n")
            break

        i += 1
        status(i, total_execs)

file_list = [
    f for f in os.listdir(output_dir)
    if os.path.isfile(os.path.join(output_dir, f)) and not f.startswith('.') and f.endswith("filled_dir.tif")
]

for _ in range(num_runs):
    for file in file_list:
        accum_output = os.path.join(output_dir, output_name + "_accum")
        time_accum, output_accum, error_accum = run_executable(["./build/bin/flow_accum", input, accum_output])
        if time_accum is not None:
            accum_runtimes.append(time_accum)
        else:
            print(f"Flow Direction Error: {error_accum}")
            break

        i += 1
        status(i, total_execs)

print("")
    
avg_pit = sum(pit_runtimes) / len(pit_runtimes)
print(f"Average runtime of pit filling: {avg_pit:.4f} seconds\n")

avg_dir = sum(dir_runtimes) / len(dir_runtimes)
print(f"Average runtime of flow direction calculation: {avg_dir:.4f} seconds\n")

avg_pit_dir = sum(pit_dir_runtimes) / len(pit_dir_runtimes)
print(f"Average runtime of pit filling and flow direction combined kernel: {avg_pit_dir:.4f} seconds\n")

avg_accum = sum(accum_runtimes) / len(accum_runtimes)
print(f"Average runtime of flow accumulation calculations: {avg_accum:.4f} seconds")