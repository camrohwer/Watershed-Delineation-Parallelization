docker build -t my_project_image .
This will create a docker container with the necessary dependencies to run both iterative and parallel versions of the program

my_project/
├── Dockerfile
├── src/
│   ├── flow_direction_iterative.cpp
│   ├── flow_direction_parallel.cu
├── CMakeLists.txt
└── README.md
