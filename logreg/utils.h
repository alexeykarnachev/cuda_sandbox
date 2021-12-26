class Managed {
   public:
    void* operator new(size_t size);
    void operator delete(void* ptr);
};

float* create_toy_x(float* y, size_t n_samples, size_t n_features);
float* create_toy_y(size_t n_samples);
float* create_toy_w(size_t n_features);
float* create_toy_b();
