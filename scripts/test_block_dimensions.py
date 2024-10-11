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

# Define your input file
input_file = '../../DEMs/cdem_dem_092G.tif'

# List of block sizes to test
block_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]

# Run the parallel version with varying block sizes
for block_size in block_sizes:
    block_x, block_y = block_size
    parallel = ['./flow_direction_parallel', input_file, str(block_x), str(block_y)]
    time_parallel, output_parallel, error_parallel = run_executable(parallel)

    if time_parallel is not None:
        print(f'Parallel runtime with block size {block_x}x{block_y}: {time_parallel:.4f} seconds')
    else:
        print(f'Parallel error with block size {block_x}x{block_y}: {error_parallel}')
