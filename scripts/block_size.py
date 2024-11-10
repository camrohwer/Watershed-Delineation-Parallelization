import subprocess
import time
import os

build_bin_dir = os.path.abspath("./build/bin")
os.chdir(build_bin_dir)

def run_executable(command):
    start_time = time.time()
    try:
        # Call the executable
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elapsed_time = time.time() - start_time
        return elapsed_time, result.stdout.decode(), result.stderr.decode()
    except subprocess.CalledProcessError as e:
        return None, e.stdout.decode(), e.stderr.decode()

if not os.path.exists("./pit_filling_flow_direction"):
    print("Executable not found!")
    exit(1)

block_sizes = [(2**i, 2**i) for i in range(1,6)]
num_runs = 5
print(f"Number of Runs {num_runs}")

for dim_x, dim_y in block_sizes:
    parallel = ['./pit_filling_flow_direction', '../../DEMs/092G.tif', '../../DEMs/Output/092G_filled.tif' str(dim_x), str(dim_y)]
    
    runtimes = []
    for _ in range(num_runs):
        time1, output1, error1 = run_executable(parallel)
        if time1 is not None:
            runtimes.append(time1)
        else:
            print(f'Parallel error: {error1}')
            break
    if len(runtimes) == num_runs:
        avg = sum(runtimes) / num_runs
        print(f"Average runtime: {avg:.4f} seconds with block size of {dim_x}x{dim_y}")