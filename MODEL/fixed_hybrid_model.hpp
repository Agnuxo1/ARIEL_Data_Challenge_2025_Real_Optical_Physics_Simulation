/**
 * FIXED HYBRID QUANTUM-NEBULA MODEL FOR ARIEL DATA CHALLENGE 2025
 *
 * FIXES:
 * 1. Proper data loading and normalization
 * 2. Multi-GPU distribution support
 * 3. Enhanced quantum feature extraction
 * 4. Improved gradient computation
 */

#ifndef FIXED_HYBRID_MODEL_HPP
#define FIXED_HYBRID_MODEL_HPP

#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
#include <Eigen/Dense>
#include <cnpy.h>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>

using namespace std;
using namespace Eigen;

// Forward declarations
extern "C" {
    void launch_encodeToComplexField(float* input, cufftComplex* field,
                                     int input_size, int field_dim);
    void launch_applyOpticalMasks(cufftComplex* freq, float* amp_mask,
                                  float* phase_mask, int dim);
    void launch_calculateIntensity(cufftComplex* field, float* intensity, int dim);
    void launch_computeOutput(float* intensity, float* W, float* b,
                              float* output, int input_dim, int output_dim);
    void launch_updateOutputLayer(float* W, float* b, float* grad_output,
                                  float* intensity, float lr, int out_dim, int in_dim);
    void launch_computeIntensityGradients(float* intensity, float* W, float* grad_output,
                                         float* grad_intensity, int out_dim, int in_dim);
    void launch_computeOpticalGradients(cufftComplex* field, float* grad_intensity,
                                       float* amp_mask, float* phase_mask,
                                       float* grad_amp, float* grad_phase, int dim);
    void launch_updateOpticalMasks(float* amp_mask, float* phase_mask,
                                  float* grad_amp, float* grad_phase,
                                  float lr, int dim);

    // CORRECTED: Add declarations for nonlinearity kernels
    void launch_applyNonlinearity(float* intensity, float* features, int dim);
    void launch_backwardNonlinearity(float* grad_features, float* intensity,
                                    float* grad_intensity, int dim);
}

// Constants
constexpr int SPECTRUM_SIZE = 283;
constexpr int QUANTUM_FEATURES = 128;
constexpr int NEBULA_SIZE = 256;
constexpr int OUTPUT_TARGETS = 6;

// ==================== FIXED QUANTUM PROCESSOR ====================
class FixedQuantumProcessor {
private:
    vector<float> spectrum_mean;
    vector<float> spectrum_std;
    bool data_loaded = false;

public:
    FixedQuantumProcessor() {
        spectrum_mean.resize(SPECTRUM_SIZE, 0.0);
        spectrum_std.resize(SPECTRUM_SIZE, 1.0);
    }

    // Load and compute proper normalization from real data
    void loadData(const string& data_path) {
        cout << "[QUANTUM] Loading calibrated data from: " << data_path << endl;

        try {
            auto data = cnpy::npy_load(data_path + "/data_train.npy");
            float* data_ptr = data.data<float>();

            int n_samples = data.shape[0];
            int n_features = data.shape[1];

            cout << "[QUANTUM] Data shape: " << n_samples << "x" << n_features << endl;

            // Compute real statistics from calibrated data
            for(int f = 0; f < min(n_features, SPECTRUM_SIZE); ++f) {
                float sum = 0.0, sum_sq = 0.0;

                for(int s = 0; s < n_samples; ++s) {
                    float val = data_ptr[s * n_features + f];
                    sum += val;
                    sum_sq += val * val;
                }

                spectrum_mean[f] = sum / n_samples;
                float variance = (sum_sq / n_samples) - (spectrum_mean[f] * spectrum_mean[f]);
                spectrum_std[f] = sqrt(max(variance, 1e-8f));
            }

            data_loaded = true;
            cout << "[QUANTUM] Real data statistics computed successfully" << endl;

        } catch(const exception& e) {
            cout << "[QUANTUM] Error loading data: " << e.what() << endl;
            cout << "[QUANTUM] Using default normalization" << endl;
        }
    }

    // Enhanced feature extraction with more diversity
    vector<float> extractFeatures(const vector<float>& spectrum) {
        vector<float> features(QUANTUM_FEATURES);

        // Normalize input spectrum with REAL statistics
        vector<float> norm_spectrum(spectrum.size());
        for(size_t i = 0; i < spectrum.size() && i < SPECTRUM_SIZE; ++i) {
            norm_spectrum[i] = (spectrum[i] - spectrum_mean[i]) / spectrum_std[i];
        }

        // Feature extraction with more spectral diversity
        for(int i = 0; i < QUANTUM_FEATURES; ++i) {
            if(i < (int)norm_spectrum.size()) {
                // Direct spectral features
                features[i] = norm_spectrum[i];
            } else if(i < 64) {
                // Spectral derivatives (changes between wavelengths)
                int idx = i % norm_spectrum.size();
                int next_idx = (idx + 1) % norm_spectrum.size();
                features[i] = norm_spectrum[next_idx] - norm_spectrum[idx];
            } else {
                // Non-linear combinations for molecular signatures
                int idx1 = (i * 3) % norm_spectrum.size();
                int idx2 = (i * 7) % norm_spectrum.size();
                features[i] = tanh(norm_spectrum[idx1] * norm_spectrum[idx2]);
            }
        }

        return features;
    }

    // CRITICAL: Save normalization parameters for Kaggle
    void saveNormalization(const string& path) {
        ofstream file(path);
        file << "# ARIEL Quantum Processor Normalization\n";
        file << spectrum_mean.size() << "\n";

        for(size_t i = 0; i < spectrum_mean.size(); ++i) {
            file << spectrum_mean[i] << " " << spectrum_std[i] << "\n";
        }

        file.close();
        cout << "[QUANTUM] Normalization saved to: " << path << endl;
    }

    // CRITICAL: Load normalization parameters for Kaggle
    void loadNormalization(const string& path) {
        ifstream file(path);
        if(!file.is_open()) {
            cout << "[QUANTUM] Warning: Could not load normalization from " << path << endl;
            return;
        }

        string line;
        getline(file, line); // Skip comment

        int size;
        file >> size;

        spectrum_mean.resize(size);
        spectrum_std.resize(size);

        for(int i = 0; i < size; ++i) {
            file >> spectrum_mean[i] >> spectrum_std[i];
        }

        file.close();
        data_loaded = true;
        cout << "[QUANTUM] Normalization loaded from: " << path << endl;
    }
};

// ==================== MULTI-GPU NEBULA PROCESSOR ====================
class MultiGPUNEBULAProcessor {
private:
    vector<int> gpu_ids;
    int current_gpu = 0;

    // Device buffers for each GPU
    vector<float*> d_input;
    vector<cufftComplex*> d_field;
    vector<cufftComplex*> d_freq;
    vector<float*> d_intensity;
    vector<float*> d_features;  // CORRECTED: Buffer for nonlinearity output
    vector<float*> d_output;

    // Learnable parameters (replicated on each GPU)
    vector<float*> d_amplitude_mask;
    vector<float*> d_phase_mask;
    vector<float*> d_W_output;
    vector<float*> d_b_output;

    // Gradient buffers
    vector<float*> d_grad_intensity;
    vector<float*> d_grad_features;  // CORRECTED: Gradient buffer for features
    vector<float*> d_grad_amplitude;
    vector<float*> d_grad_phase;

    // CRITICAL NEW: Additional gradient buffers for corrected backprop
    vector<cufftComplex*> d_grad_field;
    vector<cufftComplex*> d_grad_freq;

    // FFT plans for each GPU
    vector<cufftHandle> plan_fwd;
    vector<cufftHandle> plan_inv;

public:
    MultiGPUNEBULAProcessor() {
        // Initialize all 3 GPUs
        gpu_ids = {0, 1, 2};

        int gpu_count;
        cudaGetDeviceCount(&gpu_count);
        cout << "[MULTI-GPU] Found " << gpu_count << " GPUs" << endl;

        // Limit to available GPUs
        if(gpu_count < 3) {
            gpu_ids.resize(gpu_count);
            for(int i = 0; i < gpu_count; ++i) gpu_ids[i] = i;
        }

        initializeGPUs();
    }

    ~MultiGPUNEBULAProcessor() {
        cleanupGPUs();
    }

    void initializeGPUs() {
        d_input.resize(gpu_ids.size());
        d_field.resize(gpu_ids.size());
        d_freq.resize(gpu_ids.size());
        d_intensity.resize(gpu_ids.size());
        d_features.resize(gpu_ids.size());  // CORRECTED: Resize features buffer
        d_output.resize(gpu_ids.size());
        d_amplitude_mask.resize(gpu_ids.size());
        d_phase_mask.resize(gpu_ids.size());
        d_W_output.resize(gpu_ids.size());
        d_b_output.resize(gpu_ids.size());
        d_grad_intensity.resize(gpu_ids.size());
        d_grad_features.resize(gpu_ids.size());  // CORRECTED: Resize features gradient buffer
        d_grad_amplitude.resize(gpu_ids.size());
        d_grad_phase.resize(gpu_ids.size());

        // CRITICAL NEW: Resize new gradient buffers
        d_grad_field.resize(gpu_ids.size());
        d_grad_freq.resize(gpu_ids.size());

        plan_fwd.resize(gpu_ids.size());
        plan_inv.resize(gpu_ids.size());

        for(size_t g = 0; g < gpu_ids.size(); ++g) {
            cudaSetDevice(gpu_ids[g]);
            cout << "[MULTI-GPU] Initializing GPU " << gpu_ids[g] << endl;

            // Allocate memory on each GPU
            cudaMalloc(&d_input[g], QUANTUM_FEATURES * sizeof(float));
            cudaMalloc(&d_field[g], NEBULA_SIZE * NEBULA_SIZE * sizeof(cufftComplex));
            cudaMalloc(&d_freq[g], NEBULA_SIZE * NEBULA_SIZE * sizeof(cufftComplex));
            cudaMalloc(&d_intensity[g], NEBULA_SIZE * NEBULA_SIZE * sizeof(float));
            cudaMalloc(&d_features[g], NEBULA_SIZE * NEBULA_SIZE * sizeof(float));  // CORRECTED: Features buffer
            cudaMalloc(&d_output[g], OUTPUT_TARGETS * sizeof(float));

            cudaMalloc(&d_amplitude_mask[g], NEBULA_SIZE * NEBULA_SIZE * sizeof(float));
            cudaMalloc(&d_phase_mask[g], NEBULA_SIZE * NEBULA_SIZE * sizeof(float));
            cudaMalloc(&d_W_output[g], OUTPUT_TARGETS * NEBULA_SIZE * NEBULA_SIZE * sizeof(float));
            cudaMalloc(&d_b_output[g], OUTPUT_TARGETS * sizeof(float));

            cudaMalloc(&d_grad_intensity[g], NEBULA_SIZE * NEBULA_SIZE * sizeof(float));
            cudaMalloc(&d_grad_features[g], NEBULA_SIZE * NEBULA_SIZE * sizeof(float));  // CORRECTED: Features gradient
            cudaMalloc(&d_grad_amplitude[g], NEBULA_SIZE * NEBULA_SIZE * sizeof(float));
            cudaMalloc(&d_grad_phase[g], NEBULA_SIZE * NEBULA_SIZE * sizeof(float));

            // CRITICAL NEW: Allocate new gradient buffers
            cudaMalloc(&d_grad_field[g], NEBULA_SIZE * NEBULA_SIZE * sizeof(cufftComplex));
            cudaMalloc(&d_grad_freq[g], NEBULA_SIZE * NEBULA_SIZE * sizeof(cufftComplex));

            // Initialize parameters on each GPU
            initializeMasksOnGPU(g);

            // Create FFT plans
            cufftPlan2d(&plan_fwd[g], NEBULA_SIZE, NEBULA_SIZE, CUFFT_C2C);
            cufftPlan2d(&plan_inv[g], NEBULA_SIZE, NEBULA_SIZE, CUFFT_C2C);
        }

        cout << "[MULTI-GPU] Initialized " << gpu_ids.size() << " GPUs successfully" << endl;
    }

    void initializeMasksOnGPU(int gpu_idx) {
        cudaSetDevice(gpu_ids[gpu_idx]);

        vector<float> h_amp(NEBULA_SIZE * NEBULA_SIZE);
        vector<float> h_phase(NEBULA_SIZE * NEBULA_SIZE);
        vector<float> h_W(OUTPUT_TARGETS * NEBULA_SIZE * NEBULA_SIZE);
        vector<float> h_b(OUTPUT_TARGETS, 0.0);

        default_random_engine gen(1337 + gpu_idx);
        normal_distribution<float> dist(0.0, 0.1);

        for(int i = 0; i < NEBULA_SIZE * NEBULA_SIZE; ++i) {
            h_amp[i] = 1.0 + dist(gen);
            h_phase[i] = dist(gen);
        }

        for(int i = 0; i < OUTPUT_TARGETS * NEBULA_SIZE * NEBULA_SIZE; ++i) {
            h_W[i] = dist(gen);
        }

        cudaMemcpy(d_amplitude_mask[gpu_idx], h_amp.data(),
                  h_amp.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_phase_mask[gpu_idx], h_phase.data(),
                  h_phase.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W_output[gpu_idx], h_W.data(),
                  h_W.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_output[gpu_idx], h_b.data(),
                  h_b.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    vector<float> process(const vector<float>& quantum_features) {
        // Use round-robin GPU assignment
        int gpu_idx = current_gpu % gpu_ids.size();
        current_gpu++;

        cudaSetDevice(gpu_ids[gpu_idx]);

        // Upload features
        cudaMemcpy(d_input[gpu_idx], quantum_features.data(),
                  quantum_features.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Optical processing pipeline
        launch_encodeToComplexField(d_input[gpu_idx], d_field[gpu_idx],
                                   QUANTUM_FEATURES, NEBULA_SIZE);

        cufftExecC2C(plan_fwd[gpu_idx], d_field[gpu_idx], d_freq[gpu_idx], CUFFT_FORWARD);

        launch_applyOpticalMasks(d_freq[gpu_idx], d_amplitude_mask[gpu_idx],
                               d_phase_mask[gpu_idx], NEBULA_SIZE);

        cufftExecC2C(plan_inv[gpu_idx], d_freq[gpu_idx], d_field[gpu_idx], CUFFT_INVERSE);

        launch_calculateIntensity(d_field[gpu_idx], d_intensity[gpu_idx], NEBULA_SIZE);

        // CORRECTED: Apply nonlinearity before output computation
        launch_applyNonlinearity(d_intensity[gpu_idx], d_features[gpu_idx], NEBULA_SIZE);

        launch_computeOutput(d_features[gpu_idx], d_W_output[gpu_idx], d_b_output[gpu_idx],
                           d_output[gpu_idx], NEBULA_SIZE*NEBULA_SIZE, OUTPUT_TARGETS);

        // Download results
        vector<float> output(OUTPUT_TARGETS);
        cudaMemcpy(output.data(), d_output[gpu_idx],
                  OUTPUT_TARGETS * sizeof(float), cudaMemcpyDeviceToHost);

        return output;
    }

    void updateParameters(const vector<float>& grad_output, float lr) {
        // Update parameters on all GPUs in parallel
        for(size_t g = 0; g < gpu_ids.size(); ++g) {
            cudaSetDevice(gpu_ids[g]);

            // Update output layer
            launch_updateOutputLayer(d_W_output[g], d_b_output[g],
                                    const_cast<float*>(grad_output.data()),
                                    d_intensity[g], lr, OUTPUT_TARGETS,
                                    NEBULA_SIZE*NEBULA_SIZE);

            // CORRECTED: Compute gradients w.r.t features (not intensity directly)
            launch_computeIntensityGradients(d_features[g], d_W_output[g],
                                           const_cast<float*>(grad_output.data()),
                                           d_grad_features[g], OUTPUT_TARGETS,
                                           NEBULA_SIZE*NEBULA_SIZE);

            // CORRECTED: Backward through nonlinearity layer
            launch_backwardNonlinearity(d_grad_features[g], d_intensity[g],
                                       d_grad_intensity[g], NEBULA_SIZE);

            // Simplified gradient computation (matching original implementation)
            launch_computeOpticalGradients(d_field[g], d_grad_intensity[g],
                                         d_amplitude_mask[g], d_phase_mask[g],
                                         d_grad_amplitude[g], d_grad_phase[g], NEBULA_SIZE);

            // Update optical masks with computed gradients
            launch_updateOpticalMasks(d_amplitude_mask[g], d_phase_mask[g],
                                    d_grad_amplitude[g], d_grad_phase[g],
                                    lr, NEBULA_SIZE);
        }

        // Synchronize all GPUs
        for(size_t g = 0; g < gpu_ids.size(); ++g) {
            cudaSetDevice(gpu_ids[g]);
            cudaDeviceSynchronize();
        }
    }

    // CRITICAL: Save GPU parameters for Kaggle inference
    void saveParameters(const string& path) {
        cout << "[NEBULA] Saving GPU parameters to: " << path << endl;

        // Download parameters from GPU 0 (all GPUs have same parameters)
        cudaSetDevice(gpu_ids[0]);

        vector<float> h_amp(NEBULA_SIZE * NEBULA_SIZE);
        vector<float> h_phase(NEBULA_SIZE * NEBULA_SIZE);
        vector<float> h_W(OUTPUT_TARGETS * NEBULA_SIZE * NEBULA_SIZE);
        vector<float> h_b(OUTPUT_TARGETS);

        cudaMemcpy(h_amp.data(), d_amplitude_mask[0],
                  h_amp.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_phase.data(), d_phase_mask[0],
                  h_phase.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_W.data(), d_W_output[0],
                  h_W.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b.data(), d_b_output[0],
                  h_b.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // Save to binary file
        ofstream file(path, ios::binary);
        file.write((char*)h_amp.data(), h_amp.size() * sizeof(float));
        file.write((char*)h_phase.data(), h_phase.size() * sizeof(float));
        file.write((char*)h_W.data(), h_W.size() * sizeof(float));
        file.write((char*)h_b.data(), h_b.size() * sizeof(float));
        file.close();

        cout << "[NEBULA] Parameters saved successfully" << endl;
    }

    // CRITICAL: Load GPU parameters for Kaggle inference
    void loadParameters(const string& path) {
        cout << "[NEBULA] Loading GPU parameters from: " << path << endl;

        ifstream file(path, ios::binary);
        if(!file.is_open()) {
            cout << "[NEBULA] Warning: Could not load parameters from " << path << endl;
            return;
        }

        vector<float> h_amp(NEBULA_SIZE * NEBULA_SIZE);
        vector<float> h_phase(NEBULA_SIZE * NEBULA_SIZE);
        vector<float> h_W(OUTPUT_TARGETS * NEBULA_SIZE * NEBULA_SIZE);
        vector<float> h_b(OUTPUT_TARGETS);

        file.read((char*)h_amp.data(), h_amp.size() * sizeof(float));
        file.read((char*)h_phase.data(), h_phase.size() * sizeof(float));
        file.read((char*)h_W.data(), h_W.size() * sizeof(float));
        file.read((char*)h_b.data(), h_b.size() * sizeof(float));
        file.close();

        // Upload to all GPUs
        for(size_t g = 0; g < gpu_ids.size(); ++g) {
            cudaSetDevice(gpu_ids[g]);

            cudaMemcpy(d_amplitude_mask[g], h_amp.data(),
                      h_amp.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_phase_mask[g], h_phase.data(),
                      h_phase.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_W_output[g], h_W.data(),
                      h_W.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b_output[g], h_b.data(),
                      h_b.size() * sizeof(float), cudaMemcpyHostToDevice);
        }

        cout << "[NEBULA] Parameters loaded successfully" << endl;
    }

    void cleanupGPUs() {
        for(size_t g = 0; g < gpu_ids.size(); ++g) {
            cudaSetDevice(gpu_ids[g]);

            if(d_input[g]) cudaFree(d_input[g]);
            if(d_field[g]) cudaFree(d_field[g]);
            if(d_freq[g]) cudaFree(d_freq[g]);
            if(d_intensity[g]) cudaFree(d_intensity[g]);
            if(d_output[g]) cudaFree(d_output[g]);
            if(d_amplitude_mask[g]) cudaFree(d_amplitude_mask[g]);
            if(d_phase_mask[g]) cudaFree(d_phase_mask[g]);
            if(d_W_output[g]) cudaFree(d_W_output[g]);
            if(d_b_output[g]) cudaFree(d_b_output[g]);
            if(d_grad_intensity[g]) cudaFree(d_grad_intensity[g]);
            if(d_grad_amplitude[g]) cudaFree(d_grad_amplitude[g]);
            if(d_grad_phase[g]) cudaFree(d_grad_phase[g]);

            cufftDestroy(plan_fwd[g]);
            cufftDestroy(plan_inv[g]);
        }
    }
};

// ==================== FIXED HYBRID MODEL ====================
class FixedHybridArielModel {
private:
    FixedQuantumProcessor quantum_stage;
    MultiGPUNEBULAProcessor nebula_stage;

public:
    FixedHybridArielModel() {
        cout << "[FIXED-HYBRID] Initialized Enhanced Quantum-NEBULA model" << endl;
        cout << "  - Multi-GPU support: 3x RTX 3080" << endl;
        cout << "  - Enhanced quantum features: " << QUANTUM_FEATURES << endl;
        cout << "  - NEBULA optical size: " << NEBULA_SIZE << "x" << NEBULA_SIZE << endl;
        cout << "  - Output targets: " << OUTPUT_TARGETS << endl;
    }

    void loadData(const string& data_path) {
        quantum_stage.loadData(data_path);
    }

    vector<float> forward(const vector<float>& spectrum) {
        // Stage 1: Enhanced quantum processing with real normalization
        auto quantum_features = quantum_stage.extractFeatures(spectrum);

        // Stage 2: Multi-GPU NEBULA processing
        auto predictions = nebula_stage.process(quantum_features);

        // Physical unit conversion
        predictions[0] = predictions[0] * 1000.0 + 100.0;  // CO2 ppm
        predictions[1] = predictions[1] * 100.0 + 50.0;    // H2O %
        predictions[2] = predictions[2] * 10.0 + 5.0;      // CH4 ppm
        predictions[3] = predictions[3] * 1.0 + 0.5;       // NH3 ppm
        predictions[4] = predictions[4] * 1000.0 + 1500.0; // Temperature K
        predictions[5] = predictions[5] * 1.5 + 1.0;       // Radius Jupiter radii

        return predictions;
    }

    float trainBatch(const vector<vector<float>>& spectra,
                    const vector<vector<float>>& targets) {
        float total_loss = 0.0;

        for(size_t i = 0; i < spectra.size(); ++i) {
            auto predictions = forward(spectra[i]);

            float loss = 0.0;
            vector<float> grad_output(OUTPUT_TARGETS);

            for(int j = 0; j < OUTPUT_TARGETS; ++j) {
                float diff = predictions[j] - targets[i][j];
                loss += diff * diff;
                grad_output[j] = 2.0 * diff / spectra.size();
            }

            total_loss += loss;

            // Multi-GPU parameter update
            nebula_stage.updateParameters(grad_output, 0.001f);
        }

        return total_loss / spectra.size();
    }

    // CRITICAL: Save checkpoint for Kaggle inference
    void saveCheckpoint(const string& path) {
        cout << "[CHECKPOINT] Saving model to: " << path << endl;

        // Save NEBULA processor GPU parameters
        string nebula_path = path + "_nebula.bin";
        nebula_stage.saveParameters(nebula_path);

        // Save quantum processor normalization
        string quantum_path = path + "_quantum.txt";
        quantum_stage.saveNormalization(quantum_path);

        cout << "[CHECKPOINT] Model saved successfully" << endl;
    }

    // CRITICAL: Load checkpoint for Kaggle inference
    void loadCheckpoint(const string& path) {
        cout << "[CHECKPOINT] Loading model from: " << path << endl;

        // Load NEBULA processor GPU parameters
        string nebula_path = path + "_nebula.bin";
        nebula_stage.loadParameters(nebula_path);

        // Load quantum processor normalization
        string quantum_path = path + "_quantum.txt";
        quantum_stage.loadNormalization(quantum_path);

        cout << "[CHECKPOINT] Model loaded successfully" << endl;
    }

    void generateSubmission(const string& test_path, const string& output_path) {
        cout << "[SUBMISSION] Generating predictions with multi-GPU model..." << endl;

        ofstream submission(output_path);
        submission << "planet_ID,CO2,H2O,CH4,NH3,temperature,radius" << endl;

        // Process test samples
        for(int i = 0; i < 1000; ++i) {  // Assuming 1000 test samples
            // Create synthetic test spectrum (replace with real loading)
            vector<float> spectrum(SPECTRUM_SIZE);
            for(int j = 0; j < SPECTRUM_SIZE; ++j) {
                spectrum[j] = 0.5 + 0.1 * sin(j * 0.1 + i * 0.01);
            }

            auto predictions = forward(spectrum);

            submission << i << ",";
            for(int j = 0; j < OUTPUT_TARGETS; ++j) {
                submission << predictions[j];
                if(j < OUTPUT_TARGETS-1) submission << ",";
            }
            submission << endl;
        }

        submission.close();
        cout << "[SUBMISSION] Generated: " << output_path << endl;
    }
};

#endif // FIXED_HYBRID_MODEL_HPP