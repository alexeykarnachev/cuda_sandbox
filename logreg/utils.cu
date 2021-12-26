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

float randf(float low, float high) {
    return low + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (high - low)));
}

float* create_toy_y(size_t n_samples) {
    float* y = new float[n_samples];
    for (size_t i = 0; i < n_samples; ++i) {
        y[i] = randf(0, 1) > 0.5 ? 1.0 : 0.0;
    }
    return y;
}

float* create_toy_x(float* y, size_t n_samples, size_t n_features) {
    float* x = new float[n_samples * n_features];
    for (size_t i_sample = 0; i_sample < n_samples; ++i_sample) {
        for (size_t i_feature = 0; i_feature < n_features; ++i_feature) {
            float* val = &(x[i_sample * n_features + i_feature]);
            if (i_feature <= n_features / 2) {
                *val = randf(0, 1) > 0.1 ? y[i_sample] : 1 - y[i_sample];
            } else {
                *val = randf(0, 1) > 0.1 ? 1 - y[i_sample] : y[i_sample];
            }
        }
    }
    return x;
}

float* create_toy_w(size_t n_features) {
    float* w = new float[n_features];
    for (size_t i = 0; i < n_features; ++i) {
        w[i] = randf(-1, 1);
    }
    return w;
}

float* create_toy_b() {
    float* b = new float[1];
    b[0] = 1.0;
    return b;
}
