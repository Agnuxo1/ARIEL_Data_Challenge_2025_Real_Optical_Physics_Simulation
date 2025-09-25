// Minimal CUDA test
__global__ void test_kernel() {}

extern "C" {
    void test_launch() {
        test_kernel<<<1, 1>>>();
    }
}
