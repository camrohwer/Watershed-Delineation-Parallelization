nvidia-smi to check CUDA version  
https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md  
Check that your supported docker build image has the right tag  
sudo nvidia-ctk runtime configure --runtime=docker

docker build -t watershed_delineation .  
docker run --gpus all -it watershed_delineation  

my_project/  
├── Dockerfile  
├── src/  
│   ├── flow_direction_iterative.cpp  
│   ├── flow_direction_parallel.cu  
├── build/  
├── DEMs/  
│   ├── Output/  
├── CMakeLists.txt  
└── README.md  

cmake --build . --target flow_direction_iterative  
./flow_direction_iterative ../../DEMs/cdem_dem_092G.tif

cmake --build . --target flow_direction_parallel  
./flow_direction_parallel ../../DEMs/cdem_dem_092G.tif  
