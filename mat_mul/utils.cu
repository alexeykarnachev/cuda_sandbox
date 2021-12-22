#include "utils.h"

void* Managed::operator new(size_t size) {
    void* ptr;
    cudaMallocManaged(&ptr, size);
    cudaDeviceSynchronize();
    return ptr;
}

void Managed::operator delete(void* ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
}
