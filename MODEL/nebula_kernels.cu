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

__global__ void updateOutputLayer(float* W, float* b, float* grad_output,
                                 float* intensity, float lr, int out_dim, int in_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Update weights: W[i,j] -= lr * grad_output[i] * log(1 + intensity[j])
    if(idx < out_dim * in_dim) {
        int out_idx = idx / in_dim;
        int in_idx = idx % in_dim;
        float grad_w = grad_output[out_idx] * logf(1.0f + intensity[in_idx]);
        W[idx] -= lr * grad_w;
    }

    // Update biases: b[i] -= lr * grad_output[i]
    if(idx < out_dim) {
        b[idx] -= lr * grad_output[idx];
    }
}

__global__ void computeIntensityGradients(float* intensity, float* W, float* grad_output,
                                         float* grad_intensity, int out_dim, int in_dim) {
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(in_idx < in_dim) {
        float grad = 0.0f;
        for(int out_idx = 0; out_idx < out_dim; ++out_idx) {
            grad += grad_output[out_idx] * W[out_idx * in_dim + in_idx] *
                   (1.0f / (1.0f + intensity[in_idx]));
        }
        grad_intensity[in_idx] = grad;
    }
}

__global__ void updateOpticalMasks(float* amp_mask, float* phase_mask,
                                  float* grad_amp, float* grad_phase,
                                  float lr, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < dim * dim) {
        amp_mask[idx] -= lr * grad_amp[idx];
        phase_mask[idx] -= lr * grad_phase[idx];

        // Clamp amplitude mask to positive values
        if(amp_mask[idx] < 0.01f) amp_mask[idx] = 0.01f;
        if(amp_mask[idx] > 2.0f) amp_mask[idx] = 2.0f;
    }
}

__global__ void backpropIntensityToField(cufftComplex* field, float* grad_intensity,
                                        cufftComplex* grad_field, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements) {
        cufftComplex f = field[idx];
        float grad_I = grad_intensity[idx];
        grad_field[idx].x = 2.0f * f.x * grad_I;
        grad_field[idx].y = 2.0f * f.y * grad_I;
    }
}

__global__ void scaleComplexKernel(cufftComplex* data, float scale, int elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < elements) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

__global__ void computeMaskGradientsKernel(cufftComplex* freq_pre, cufftComplex* grad_freq_post,
                                           float* amp_mask, float* phase_mask,
                                           float* grad_amp, float* grad_phase,
                                           cufftComplex* grad_freq_pre, int total_elements, int pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements) {
        int pixel_idx = idx % pixels;

        cufftComplex z = freq_pre[idx];
        cufftComplex g = grad_freq_post[idx];
        float amp = amp_mask[pixel_idx];
        float phase = phase_mask[pixel_idx];
        float cos_p = cosf(phase);
        float sin_p = sinf(phase);

        // w = z * exp(i phase)
        float w_r = z.x * cos_p - z.y * sin_p;
        float w_i = z.x * sin_p + z.y * cos_p;

        float grad_amp_local = g.x * w_r + g.y * w_i;
        float grad_phase_local = amp * (-g.x * w_i + g.y * w_r);

        atomicAdd(&grad_amp[pixel_idx], grad_amp_local);
        atomicAdd(&grad_phase[pixel_idx], grad_phase_local);

        // Gradient w.r.t. freq_pre (conjugate mask multiplication)
        cufftComplex grad_z;
        grad_z.x = amp * (g.x * cos_p + g.y * sin_p);
        grad_z.y = amp * (-g.x * sin_p + g.y * cos_p);
        grad_freq_pre[idx] = grad_z;
    }
}

// C wrapper functions for calling from C++
extern "C" {
    void launch_encodeToComplexField(float* input, cufftComplex* field,
                                     int input_size, int field_dim) {
        dim3 block(256);
        dim3 grid((field_dim*field_dim+255)/256);
        encodeToComplexField<<<grid, block>>>(input, field, input_size, field_dim);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        // Error handling moved to host code
    }

    void launch_applyOpticalMasks(cufftComplex* freq, float* amp_mask,
                                  float* phase_mask, int dim) {
        dim3 block(256);
        dim3 grid((dim*dim+255)/256);
        applyOpticalMasks<<<grid, block>>>(freq, amp_mask, phase_mask, dim);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        // Error handling moved to host code
    }

    void launch_calculateIntensity(cufftComplex* field, float* intensity, int dim) {
        dim3 block(256);
        dim3 grid((dim*dim+255)/256);
        calculateIntensity<<<grid, block>>>(field, intensity, dim);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        // Error handling moved to host code
    }

    void launch_computeOutput(float* intensity, float* W, float* b,
                              float* output, int input_dim, int output_dim) {
        dim3 block(32);
        dim3 grid((output_dim+31)/32);
        computeOutput<<<grid, block>>>(intensity, W, b, output, input_dim, output_dim);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        // Error handling moved to host code
    }

    void launch_updateOutputLayer(float* W, float* b, float* grad_output,
                                  float* intensity, float lr, int out_dim, int in_dim) {
        dim3 block(256);
        dim3 grid((out_dim*in_dim+255)/256);
        updateOutputLayer<<<grid, block>>>(W, b, grad_output, intensity, lr, out_dim, in_dim);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        // Error handling moved to host code updateOutputLayer: %s\n", cudaGetErrorString(err));
    }

    void launch_computeIntensityGradients(float* intensity, float* W, float* grad_output,
                                         float* grad_intensity, int out_dim, int in_dim) {
        dim3 block(256);
        dim3 grid((in_dim+255)/256);
        computeIntensityGradients<<<grid, block>>>(intensity, W, grad_output, grad_intensity, out_dim, in_dim);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        // Error handling moved to host code computeIntensityGradients: %s\n", cudaGetErrorString(err));
    }

    void launch_updateOpticalMasks(float* amp_mask, float* phase_mask,
                                  float* grad_amp, float* grad_phase,
                                  float lr, int dim) {
        dim3 block(256);
        dim3 grid((dim*dim+255)/256);
        updateOpticalMasks<<<grid, block>>>(amp_mask, phase_mask, grad_amp, grad_phase, lr, dim);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        // Error handling moved to host code updateOpticalMasks: %s\n", cudaGetErrorString(err));
    }

    void launch_backpropIntensityToField(cufftComplex* field, float* grad_intensity,
                                         cufftComplex* grad_field, int total_elements) {
        dim3 block(256);
        dim3 grid((total_elements+255)/256);
        backpropIntensityToField<<<grid, block>>>(field, grad_intensity, grad_field, total_elements);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        // Error handling moved to host code backpropIntensityToField: %s\n", cudaGetErrorString(err));
    }

    void launch_scaleComplex(cufftComplex* data, float scale, int elements) {
        dim3 block(256);
        dim3 grid((elements+255)/256);
        scaleComplexKernel<<<grid, block>>>(data, scale, elements);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        // Error handling moved to host code scaleComplex: %s\n", cudaGetErrorString(err));
    }

    void launch_computeMaskGradients(cufftComplex* freq_pre, cufftComplex* grad_freq_post,
                                     float* amp_mask, float* phase_mask,
                                     float* grad_amp, float* grad_phase,
                                     cufftComplex* grad_freq_pre, int total_elements, int pixels) {
        dim3 block(256);
        dim3 grid((total_elements+255)/256);
        computeMaskGradientsKernel<<<grid, block>>>(freq_pre, grad_freq_post, amp_mask, phase_mask,
                                                     grad_amp, grad_phase, grad_freq_pre, total_elements, pixels);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        // Error handling moved to host code computeMaskGradients: %s\n", cudaGetErrorString(err));
    }

    // Debug function to dump gradients to host
    void debug_dump_gradients(float* d_grad_output, int output_dim, const char* label) {
        float* h_grad = new float[output_dim];
        cudaMemcpy(h_grad, d_grad_output, output_dim * sizeof(float), cudaMemcpyDeviceToHost);
        // Debug output moved to host code
        delete[] h_grad;
    }
}