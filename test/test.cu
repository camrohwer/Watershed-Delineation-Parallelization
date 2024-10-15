#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

__global__ void hello_from_gpu(){
    printf("Hello from kernel!\n");
}

int main() {
    hello_from_gpu<<<1, 1>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    //wait for kernel to finish
    cudaDeviceSynchronize();
    std::cout << "Hello from host!" <<std::endl;
    return 0;
}
