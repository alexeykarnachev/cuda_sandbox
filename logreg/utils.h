class Managed {
   public:
    void* operator new(size_t size);
    void operator delete(void* ptr);
};

float* create_random_data(size_t n_vals, int mod_div);
