#include "utils.h"

void* Managed::operator new(size_t size) {
    void* ptr;
    cudaMallocManaged(&ptr, size);
    cudaDeviceSynchronize();
    return ptr;
};

void Managed::operator delete(void* ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
}

float* create_random_data(size_t n_vals, int mod_div) {
    float* data = new float[n_vals];
    for (size_t i_val = 0; i_val < n_vals; ++i_val) {
        data[i_val] = (float)(rand() % mod_div);
    }
    return data;
}
