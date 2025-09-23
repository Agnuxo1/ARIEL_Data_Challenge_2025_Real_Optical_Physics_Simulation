#include <cuda_runtime.h>
#include <cufft.h>

__global__ void encodeToComplexField(float* input, cufftComplex* field,
                                     int input_size, int field_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < field_dim * field_dim) {
        int input_idx = idx % input_size;
        field[idx].x = input[input_idx] * cosf(idx * 0.1f);
        field[idx].y = input[input_idx] * sinf(idx * 0.1f);
    }
}

__global__ void applyOpticalMasks(cufftComplex* freq, float* amp_mask,
                                  float* phase_mask, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < dim * dim) {
        float amp = amp_mask[idx];
        float phase = phase_mask[idx];

        float real = freq[idx].x * amp * cosf(phase) -
                    freq[idx].y * amp * sinf(phase);
        float imag = freq[idx].x * amp * sinf(phase) +
                    freq[idx].y * amp * cosf(phase);

        freq[idx].x = real;
        freq[idx].y = imag;
    }
}

__global__ void calculateIntensity(cufftComplex* field, float* intensity, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < dim * dim) {
        float norm = 1.0f / (dim * dim);
        intensity[idx] = (field[idx].x * field[idx].x +
                         field[idx].y * field[idx].y) * norm;
    }
}

__global__ void computeOutput(float* intensity, float* W, float* b,
                              float* output, int input_dim, int output_dim) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(out_idx < output_dim) {
        float sum = b[out_idx];
        for(int i = 0; i < input_dim; ++i) {
            sum += W[out_idx * input_dim + i] * logf(1.0f + intensity[i]);
        }
        output[out_idx] = sum;
    }
}

__global__ void updateOutputLayer(float* W, float* b, const float* grad,
                                 float lr, int out_dim, int in_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < out_dim * in_dim) {
        W[idx] -= lr * grad[idx % out_dim];
    }
    if(idx < out_dim) {
        b[idx] -= lr * grad[idx];
    }
}
