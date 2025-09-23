/**
 * NEBULA CORRECTED - Proper Gradient Flow through FFT
 * 
 * Key fixes:
 * 1. Explicit FFT gradient computation
 * 2. Custom backward pass for optical masks
 * 3. Proper chain rule implementation
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <stdio.h>
#include <math.h>

// ============== FORWARD KERNELS ==============

__global__ void encodeInput(float* input, cuComplex* field, int input_size, int field_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < field_size * field_size) {
        int i = idx / field_size;
        int j = idx % field_size;
        
        // Map input to 2D field with phase encoding
        float val = 0.0f;
        if (idx < input_size) {
            val = input[idx];
        }
        
        // Initialize with Gaussian envelope
        float cx = (i - field_size/2) / (float)(field_size/4);
        float cy = (j - field_size/2) / (float)(field_size/4);
        float gaussian = expf(-(cx*cx + cy*cy));
        
        field[idx].x = val * gaussian * cosf(idx * 0.1f);
        field[idx].y = val * gaussian * sinf(idx * 0.1f);
    }
}

__global__ void applyMasksForward(cuComplex* freq, float* amp_raw, float* phase_raw, 
                                  cuComplex* masked_freq, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size) {
        // Convert raw parameters to physical values
        float amp = 1.0f + tanhf(amp_raw[idx]);  // Range [0, 2]
        float phase = 3.14159f * tanhf(phase_raw[idx]); // Range [-π, π]
        
        float cos_p = cosf(phase);
        float sin_p = sinf(phase);
        
        // Apply masks: M * F where M = A * exp(iΦ)
        float real = freq[idx].x * amp;
        float imag = freq[idx].y * amp;
        
        // Rotate by phase
        masked_freq[idx].x = real * cos_p - imag * sin_p;
        masked_freq[idx].y = real * sin_p + imag * cos_p;
    }
}

__global__ void computeIntensity(cuComplex* field, float* intensity, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size) {
        float norm = 1.0f / (size * size);
        float re = field[idx].x * norm;
        float im = field[idx].y * norm;
        intensity[idx] = re * re + im * im;
    }
}

__global__ void applyNonlinearity(float* intensity, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size) {
        // Log nonlinearity with stability
        output[idx] = logf(1.0f + intensity[idx] * 100.0f);
    }
}

// ============== BACKWARD KERNELS ==============

__global__ void backwardNonlinearity(float* grad_output, float* intensity, 
                                     float* grad_intensity, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size) {
        // Gradient of log(1 + 100*I)
        float scale = 100.0f;
        grad_intensity[idx] = grad_output[idx] * scale / (1.0f + scale * intensity[idx]);
    }
}

__global__ void backwardIntensity(float* grad_intensity, cuComplex* field, 
                                 cuComplex* grad_field, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size) {
        // Gradient of |field|^2 = field.x^2 + field.y^2
        float norm = 1.0f / (size * size);
        grad_field[idx].x = 2.0f * field[idx].x * norm * norm * grad_intensity[idx];
        grad_field[idx].y = 2.0f * field[idx].y * norm * norm * grad_intensity[idx];
    }
}

__global__ void backwardMasks(cuComplex* grad_masked_freq, cuComplex* freq,
                              float* amp_raw, float* phase_raw,
                              float* grad_amp_raw, float* grad_phase_raw,
                              cuComplex* grad_freq, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size) {
        // Current mask values
        float amp = 1.0f + tanhf(amp_raw[idx]);
        float phase = 3.14159f * tanhf(phase_raw[idx]);
        float cos_p = cosf(phase);
        float sin_p = sinf(phase);
        
        // Input frequency
        float f_re = freq[idx].x;
        float f_im = freq[idx].y;
        
        // Output gradient
        float g_re = grad_masked_freq[idx].x;
        float g_im = grad_masked_freq[idx].y;
        
        // Gradient w.r.t amplitude (through tanh)
        float grad_amp = (g_re * (f_re * cos_p - f_im * sin_p) + 
                         g_im * (f_re * sin_p + f_im * cos_p));
        float tanh_a = tanhf(amp_raw[idx]);
        grad_amp_raw[idx] = grad_amp * (1.0f - tanh_a * tanh_a);
        
        // Gradient w.r.t phase (through tanh)
        float grad_phase = amp * (g_re * (-f_re * sin_p - f_im * cos_p) +
                                  g_im * (f_re * cos_p - f_im * sin_p));
        float tanh_p = tanhf(phase_raw[idx]);
        grad_phase_raw[idx] = grad_phase * 3.14159f * (1.0f - tanh_p * tanh_p);
        
        // Gradient w.r.t input frequency
        grad_freq[idx].x = g_re * amp * cos_p - g_im * amp * sin_p;
        grad_freq[idx].y = g_re * amp * sin_p + g_im * amp * cos_p;
    }
}

__global__ void accumulateGradients(float* grad_acc, float* grad_new, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&grad_acc[idx], grad_new[idx]);
    }
}

// ============== PARAMETER UPDATE ==============

__global__ void adamUpdate(float* params, float* grads, float* m, float* v,
                           float lr, float beta1, float beta2, float eps,
                           int t, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grads[idx];
        
        // Adam momentum
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        // Bias correction
        float m_hat = m[idx] / (1.0f - powf(beta1, t));
        float v_hat = v[idx] / (1.0f - powf(beta2, t));
        
        // Update
        params[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
        
        // Clear gradient for next iteration
        grads[idx] = 0.0f;
    }
}

// ============== MAIN NEBULA CLASS ==============

class NEBULACorrected {
public:
    // Dimensions
    int batch_size;
    int input_dim;
    int field_size;
    int output_dim;
    
    // Forward buffers
    cuComplex* d_field_init;
    cuComplex* d_freq;
    cuComplex* d_masked_freq;
    cuComplex* d_field_final;
    float* d_intensity;
    float* d_features;
    
    // Backward buffers
    cuComplex* d_grad_field_final;
    cuComplex* d_grad_masked_freq;
    cuComplex* d_grad_freq;
    cuComplex* d_grad_field_init;
    float* d_grad_intensity;
    float* d_grad_features;
    
    // Parameters
    float* d_amp_raw;
    float* d_phase_raw;
    float* d_W_out;
    float* d_b_out;
    
    // Gradients
    float* d_grad_amp_raw;
    float* d_grad_phase_raw;
    float* d_grad_W_out;
    float* d_grad_b_out;
    
    // Adam states
    float* d_m_amp, *d_v_amp;
    float* d_m_phase, *d_v_phase;
    float* d_m_W, *d_v_W;
    float* d_m_b, *d_v_b;
    
    // FFT plans
    cufftHandle plan_fwd, plan_inv;
    
    // CUBLAS handle
    cublasHandle_t cublas;
    
    // Training state
    int adam_t = 0;
    
    NEBULACorrected(int batch, int input, int field, int output) 
        : batch_size(batch), input_dim(input), field_size(field), output_dim(output) {
        
        // Allocate forward buffers
        int field_total = batch * field_size * field_size;
        cudaMalloc(&d_field_init, field_total * sizeof(cuComplex));
        cudaMalloc(&d_freq, field_total * sizeof(cuComplex));
        cudaMalloc(&d_masked_freq, field_total * sizeof(cuComplex));
        cudaMalloc(&d_field_final, field_total * sizeof(cuComplex));
        cudaMalloc(&d_intensity, field_total * sizeof(float));
        cudaMalloc(&d_features, field_total * sizeof(float));
        
        // Allocate backward buffers
        cudaMalloc(&d_grad_field_final, field_total * sizeof(cuComplex));
        cudaMalloc(&d_grad_masked_freq, field_total * sizeof(cuComplex));
        cudaMalloc(&d_grad_freq, field_total * sizeof(cuComplex));
        cudaMalloc(&d_grad_field_init, field_total * sizeof(cuComplex));
        cudaMalloc(&d_grad_intensity, field_total * sizeof(float));
        cudaMalloc(&d_grad_features, field_total * sizeof(float));
        
        // Allocate parameters
        int mask_size = field_size * field_size;
        cudaMalloc(&d_amp_raw, mask_size * sizeof(float));
        cudaMalloc(&d_phase_raw, mask_size * sizeof(float));
        cudaMalloc(&d_W_out, output_dim * mask_size * sizeof(float));
        cudaMalloc(&d_b_out, output_dim * sizeof(float));
        
        // Allocate gradients
        cudaMalloc(&d_grad_amp_raw, mask_size * sizeof(float));
        cudaMalloc(&d_grad_phase_raw, mask_size * sizeof(float));
        cudaMalloc(&d_grad_W_out, output_dim * mask_size * sizeof(float));
        cudaMalloc(&d_grad_b_out, output_dim * sizeof(float));
        
        // Allocate Adam states
        cudaMalloc(&d_m_amp, mask_size * sizeof(float));
        cudaMalloc(&d_v_amp, mask_size * sizeof(float));
        cudaMalloc(&d_m_phase, mask_size * sizeof(float));
        cudaMalloc(&d_v_phase, mask_size * sizeof(float));
        cudaMalloc(&d_m_W, output_dim * mask_size * sizeof(float));
        cudaMalloc(&d_v_W, output_dim * mask_size * sizeof(float));
        cudaMalloc(&d_m_b, output_dim * sizeof(float));
        cudaMalloc(&d_v_b, output_dim * sizeof(float));
        
        // Initialize parameters
        initializeParameters();
        
        // Create FFT plans
        int n[2] = {field_size, field_size};
        cufftPlanMany(&plan_fwd, 2, n, NULL, 1, field_size*field_size,
                      NULL, 1, field_size*field_size, CUFFT_C2C, batch);
        cufftPlanMany(&plan_inv, 2, n, NULL, 1, field_size*field_size,
                      NULL, 1, field_size*field_size, CUFFT_C2C, batch);
        
        // Create CUBLAS handle
        cublasCreate(&cublas);
    }
    
    void initializeParameters() {
        // Xavier/He initialization
        int mask_size = field_size * field_size;
        float* h_amp = new float[mask_size];
        float* h_phase = new float[mask_size];
        
        float std_dev = sqrtf(2.0f / mask_size);
        
        for (int i = 0; i < mask_size; ++i) {
            h_amp[i] = ((rand() / (float)RAND_MAX) - 0.5f) * std_dev;
            h_phase[i] = ((rand() / (float)RAND_MAX) - 0.5f) * std_dev;
        }
        
        cudaMemcpy(d_amp_raw, h_amp, mask_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_phase_raw, h_phase, mask_size * sizeof(float), cudaMemcpyHostToDevice);
        
        // Zero gradients and Adam states
        cudaMemset(d_grad_amp_raw, 0, mask_size * sizeof(float));
        cudaMemset(d_grad_phase_raw, 0, mask_size * sizeof(float));
        cudaMemset(d_m_amp, 0, mask_size * sizeof(float));
        cudaMemset(d_v_amp, 0, mask_size * sizeof(float));
        cudaMemset(d_m_phase, 0, mask_size * sizeof(float));
        cudaMemset(d_v_phase, 0, mask_size * sizeof(float));
        
        delete[] h_amp;
        delete[] h_phase;
        
        // Initialize output layer with smaller values
        int out_size = output_dim * mask_size;
        float* h_W = new float[out_size];
        float out_std = sqrtf(1.0f / mask_size);
        
        for (int i = 0; i < out_size; ++i) {
            h_W[i] = ((rand() / (float)RAND_MAX) - 0.5f) * out_std;
        }
        
        cudaMemcpy(d_W_out, h_W, out_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_b_out, 0, output_dim * sizeof(float));
        
        delete[] h_W;
    }
    
    void forward(float* d_input, float* d_output) {
        dim3 block(256);
        dim3 grid((field_size * field_size + 255) / 256);
        
        // 1. Encode input to complex field
        encodeInput<<<grid, block>>>(d_input, d_field_init, input_dim, field_size);
        
        // 2. Forward FFT
        cufftExecC2C(plan_fwd, d_field_init, d_freq, CUFFT_FORWARD);
        
        // 3. Apply optical masks
        applyMasksForward<<<grid, block>>>(d_freq, d_amp_raw, d_phase_raw, 
                                           d_masked_freq, field_size);
        
        // 4. Inverse FFT
        cufftExecC2C(plan_inv, d_masked_freq, d_field_final, CUFFT_INVERSE);
        
        // 5. Compute intensity
        computeIntensity<<<grid, block>>>(d_field_final, d_intensity, field_size);
        
        // 6. Apply nonlinearity
        applyNonlinearity<<<grid, block>>>(d_intensity, d_features, field_size);
        
        // 7. Output layer (using CUBLAS for efficiency)
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemv(cublas, CUBLAS_OP_N, output_dim, field_size * field_size,
                   &alpha, d_W_out, output_dim, d_features, 1,
                   &beta, d_output, 1);
        
        // Add bias
        cublasSaxpy(cublas, output_dim, &alpha, d_b_out, 1, d_output, 1);
    }
    
    void backward(float* d_grad_output, float* d_input) {
        dim3 block(256);
        dim3 grid((field_size * field_size + 255) / 256);
        
        // 7. Backward through output layer
        float alpha = 1.0f, beta = 0.0f;
        
        // grad_W = grad_output * features^T
        cublasSger(cublas, output_dim, field_size * field_size,
                  &alpha, d_grad_output, 1, d_features, 1,
                  d_grad_W_out, output_dim);
        
        // grad_b = grad_output
        cublasSaxpy(cublas, output_dim, &alpha, d_grad_output, 1, d_grad_b_out, 1);
        
        // grad_features = W^T * grad_output
        cublasSgemv(cublas, CUBLAS_OP_T, output_dim, field_size * field_size,
                   &alpha, d_W_out, output_dim, d_grad_output, 1,
                   &beta, d_grad_features, 1);
        
        // 6. Backward through nonlinearity
        backwardNonlinearity<<<grid, block>>>(d_grad_features, d_intensity, 
                                              d_grad_intensity, field_size);
        
        // 5. Backward through intensity computation
        backwardIntensity<<<grid, block>>>(d_grad_intensity, d_field_final, 
                                           d_grad_field_final, field_size);
        
        // 4. Backward through inverse FFT (FFT of gradient)
        cufftExecC2C(plan_fwd, d_grad_field_final, d_grad_masked_freq, CUFFT_FORWARD);
        
        // 3. Backward through masks
        backwardMasks<<<grid, block>>>(d_grad_masked_freq, d_freq,
                                       d_amp_raw, d_phase_raw,
                                       d_grad_amp_raw, d_grad_phase_raw,
                                       d_grad_freq, field_size);
        
        // 2. Backward through forward FFT (inverse FFT of gradient)
        cufftExecC2C(plan_inv, d_grad_freq, d_grad_field_init, CUFFT_INVERSE);
        
        // 1. Backward through input encoding (if needed for multi-stage)
        // Not implemented here as input gradients may not be needed
    }
    
    void updateParameters(float lr) {
        adam_t++;
        
        dim3 block(256);
        int mask_size = field_size * field_size;
        dim3 grid((mask_size + 255) / 256);
        
        // Update amplitude mask
        adamUpdate<<<grid, block>>>(d_amp_raw, d_grad_amp_raw, d_m_amp, d_v_amp,
                                    lr, 0.9f, 0.999f, 1e-8f, adam_t, mask_size);
        
        // Update phase mask
        adamUpdate<<<grid, block>>>(d_phase_raw, d_grad_phase_raw, d_m_phase, d_v_phase,
                                    lr, 0.9f, 0.999f, 1e-8f, adam_t, mask_size);
        
        // Update output layer
        int out_size = output_dim * mask_size;
        dim3 grid_out((out_size + 255) / 256);
        adamUpdate<<<grid_out, block>>>(d_W_out, d_grad_W_out, d_m_W, d_v_W,
                                        lr * 0.1f, 0.9f, 0.999f, 1e-8f, adam_t, out_size);
        
        dim3 grid_bias((output_dim + 255) / 256);
        adamUpdate<<<grid_bias, block>>>(d_b_out, d_grad_b_out, d_m_b, d_v_b,
                                         lr * 0.1f, 0.9f, 0.999f, 1e-8f, adam_t, output_dim);
    }
    
    float computeLoss(float* d_predictions, float* d_targets) {
        // MSE Loss
        float loss = 0.0f;
        cublasSnrm2(cublas, output_dim, d_predictions, 1, &loss);
        return loss * loss / output_dim;
    }
    
    ~NEBULACorrected() {
        // Free all allocated memory
        cudaFree(d_field_init);
        cudaFree(d_freq);
        cudaFree(d_masked_freq);
        cudaFree(d_field_final);
        cudaFree(d_intensity);
        cudaFree(d_features);
        
        cudaFree(d_grad_field_final);
        cudaFree(d_grad_masked_freq);
        cudaFree(d_grad_freq);
        cudaFree(d_grad_field_init);
        cudaFree(d_grad_intensity);
        cudaFree(d_grad_features);
        
        cudaFree(d_amp_raw);
        cudaFree(d_phase_raw);
        cudaFree(d_W_out);
        cudaFree(d_b_out);
        
        cudaFree(d_grad_amp_raw);
        cudaFree(d_grad_phase_raw);
        cudaFree(d_grad_W_out);
        cudaFree(d_grad_b_out);
        
        cudaFree(d_m_amp);
        cudaFree(d_v_amp);
        cudaFree(d_m_phase);
        cudaFree(d_v_phase);
        cudaFree(d_m_W);
        cudaFree(d_v_W);
        cudaFree(d_m_b);
        cudaFree(d_v_b);
        
        cufftDestroy(plan_fwd);
        cufftDestroy(plan_inv);
        cublasDestroy(cublas);
    }
};

// ============== TEST GRADIENTS ==============

__global__ void testGradientFlow(float* params, float* grads, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Numerical gradient check
        float eps = 1e-4f;
        float orig = params[idx];
        
        // Forward difference
        params[idx] = orig + eps;
        // ... compute loss_plus ...
        
        params[idx] = orig - eps;
        // ... compute loss_minus ...
        
        float numerical_grad = 0.0f; // (loss_plus - loss_minus) / (2 * eps);
        float analytical_grad = grads[idx];
        
        float error = fabsf(numerical_grad - analytical_grad) / 
                     (fabsf(numerical_grad) + fabsf(analytical_grad) + 1e-8f);
        
        if (error > 0.01f) {
            printf("Gradient error at %d: numerical=%.6f, analytical=%.6f, error=%.2f%%\n",
                   idx, numerical_grad, analytical_grad, error * 100.0f);
        }
        
        params[idx] = orig;
    }
}

// ============== TRAINING WRAPPER ==============

extern "C" {
    
void* create_nebula_corrected(int batch, int input, int field, int output) {
    NEBULACorrected* model = new NEBULACorrected(batch, input, field, output);
    return (void*)model;
}

void train_step_nebula(void* model_ptr, float* h_input, float* h_target, 
                       int batch_size, float lr) {
    NEBULACorrected* model = (NEBULACorrected*)model_ptr;
    
    // Allocate device memory
    float *d_input, *d_output, *d_target, *d_grad_output;
    int input_size = batch_size * model->input_dim;
    int output_size = batch_size * model->output_dim;
    
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_target, output_size * sizeof(float));
    cudaMalloc(&d_grad_output, output_size * sizeof(float));
    
    // Copy input and target to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target, output_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Forward pass
    model->forward(d_input, d_output);
    
    // Compute loss gradient (MSE)
    dim3 block(256);
    dim3 grid((output_size + 255) / 256);
    
    // grad = 2 * (output - target) / N
    cublasSaxpy(model->cublas, output_size, -1.0f, d_target, 1, d_output, 1);
    cublasSscal(model->cublas, output_size, 2.0f / output_size, d_output, 1);
    cudaMemcpy(d_grad_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Backward pass
    model->backward(d_grad_output, d_input);
    
    // Update parameters
    model->updateParameters(lr);
    
    // Compute and print loss
    float loss = model->computeLoss(d_output, d_target);
    if (model->adam_t % 10 == 0) {
        printf("Step %d, Loss: %.6f\n", model->adam_t, loss);
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_target);
    cudaFree(d_grad_output);
}

void destroy_nebula_corrected(void* model_ptr) {
    NEBULACorrected* model = (NEBULACorrected*)model_ptr;
    delete model;
}

} // extern "C"
