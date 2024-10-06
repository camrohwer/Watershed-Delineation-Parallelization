my_project/
├── Dockerfile
├── src/
│   ├── flow_direction_iterative.cpp
│   ├── flow_direction_parallel.cu
├── CMakeLists.txt
└── README.md

mkdir -p build
cd build
cmake ..
make