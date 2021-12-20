#include <stdio.h>

class Managed {
   public:
    void* operator new(size_t size) {
        void* ptr;
        cudaMallocManaged(&ptr, size);
        cudaDeviceSynchronize();
        return ptr;
    }

    void operator delete(void* ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

class Data : public Managed {
   public:
    size_t n_values;
    int* data;

    Data(size_t n_values) : n_values(n_values) {
        size_t size = sizeof(int) * n_values;
        cudaMallocManaged(&data, size);
        cudaDeviceSynchronize();
        for (size_t i = 0; i < n_values; ++i) {
            data[i] = i;
        }
    }

    Data(const Data& other) : n_values(other.n_values) {
        cudaFree(data);
        size_t size = sizeof(int) * n_values;
        cudaMallocManaged(&data, size);

        for (size_t i = 0; i < n_values; ++i) {
            data[i] = other.data[i];
        }
        printf("Copy via copy constructor\n");
    }

    ~Data() { cudaFree(data); }

    __host__ __device__ int& operator[](int idx) { return data[idx]; }
};

__global__ void kernel_by_pointer(Data* data) {
    int mean;
    for (size_t i = 0; i < data->n_values; ++i) {
        mean += (*data)[i];
        (*data)[i] *= 2;
    }
    mean /= data->n_values;
    printf("kernel_by_pointer, mean=%d\n", mean);
}

__global__ void kernel_by_ref(Data& data) {
    int mean;
    for (size_t i = 0; i < data.n_values; ++i) {
        mean += data[i];
        data[i] *= 2;
    }
    mean /= data.n_values;
    printf("kernel_by_ref, mean=%d\n", mean);
}

__global__ void kernel_by_value(Data data) {
    int mean;
    for (size_t i = 0; i < data.n_values; ++i) {
        mean += data[i];
        data[i] *= 2;
    }
    mean /= data.n_values;
    printf("kernel_by_value, mean=%d\n", mean);
}

int main() {
    size_t n_values = 256;
    Data* data_p = new Data(n_values);

    kernel_by_pointer<<<1, 1>>>(data_p);
    cudaDeviceSynchronize();

    kernel_by_ref<<<1, 1>>>(*data_p);
    cudaDeviceSynchronize();

    kernel_by_value<<<1, 1>>>(*data_p);
    cudaDeviceSynchronize();

    kernel_by_pointer<<<1, 1>>>(data_p);
    cudaDeviceSynchronize();

    kernel_by_ref<<<1, 1>>>(*data_p);
    cudaDeviceSynchronize();

    kernel_by_value<<<1, 1>>>(*data_p);
    cudaDeviceSynchronize();

    kernel_by_pointer<<<1, 1>>>(data_p);
    cudaDeviceSynchronize();
}
