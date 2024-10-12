#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_from_gpu(){
    printf("Hellow World!\n");
}

int main() {
    hello_from_gpu<<<1, 10>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    //wait for kernel to finish
    cudaDeviceSynchronize();
    return 0;
}
