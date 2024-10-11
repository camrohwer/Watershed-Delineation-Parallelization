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

# Define your executables and their arguments
iterative = ['./flow_direction_iterative', '../../DEMs/cdem_dem_092G.tif']
parallel = ['./flow_direction_parallel', '../../DEMs/cdem_dem_092G.tif']


# Run the executables and get their runtimes
time1, output1, error1 = run_executable(iterative)
time2, output2, error2 = run_executable(parallel)

# Print the results
if time1 is not None:
    print(f'Iterative runtime: {time1:.4f} seconds')
else:
    print(f'Iterative error: {error1}')

if time2 is not None:
    print(f'Parallel runtime: {time2:.4f} seconds')
else:
    print(f'Parallel error: {error2}')
