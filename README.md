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
cmake --build . --target flow_direction_parallel  
