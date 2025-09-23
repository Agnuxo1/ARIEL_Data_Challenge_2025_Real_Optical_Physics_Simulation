/**
 * HYBRID QUANTUM-NEBULA MODEL FOR ARIEL DATA CHALLENGE 2025
 * 
 * Combines:
 * 1. Quantum Photonic Processing (ITensor) for spectral feature extraction
 * 2. NEBULA Optical Fourier Processing (CUDA) for pattern recognition
 * 
 * Target: Exoplanet atmospheric characterization from spectroscopy
 */

#ifndef HYBRID_ARIEL_MODEL_HPP
#define HYBRID_ARIEL_MODEL_HPP

#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
#include "itensor/all.h"
#include <cnpy.h>  // For numpy file loading
#include <fstream>
#include <iostream>
#include <chrono>

using namespace itensor;
using namespace std;

// ==================== ARIEL SPECIFIC CONSTANTS ====================
constexpr int AIRS_WAVELENGTHS = 282;    // Spectral channels from AIRS-CH0
constexpr int FGS_SIZE = 32 * 32;        // FGS1 photometry pixels
constexpr int SPECTRUM_SIZE = AIRS_WAVELENGTHS + 1; // +1 for FGS
constexpr int TIME_BINS = 187;           // After binning

// Quantum parameters
constexpr int QUANTUM_SITES = 16;        // Quantum lattice sites
constexpr int QUANTUM_FEATURES = 128;    // Output features from quantum stage

// NEBULA parameters (adapted for spectroscopy)
constexpr int NEBULA_SIZE = 256;         // Internal representation size
constexpr int OUTPUT_TARGETS = 6;        // CO2, H2O, CH4, NH3, Temperature, Radius

// Physical constants
constexpr double HBAR = 1.054571817e-34;
constexpr double C = 299792458.0;

// CUDA kernel declarations
extern __global__ void encodeToComplexField(float* input, cufftComplex* field,
                                           int input_size, int field_dim);
extern __global__ void applyOpticalMasks(cufftComplex* freq, float* amp_mask,
                                        float* phase_mask, int dim);
extern __global__ void calculateIntensity(cufftComplex* field, float* intensity, int dim);
extern __global__ void computeOutput(float* intensity, float* W, float* b,
                                   float* output, int input_dim, int output_dim);
extern __global__ void updateOutputLayer(float* W, float* b, const float* grad,
                                       float lr, int out_dim, int in_dim);

// ==================== QUANTUM STAGE ====================
class QuantumSpectralProcessor {
private:
    SiteSet sites;
    MPS psi;
    MPO H;
    
    // Spectral encoding parameters
    vector<double> wavelength_coupling;
    vector<double> spectral_weights;
    
public:
    QuantumSpectralProcessor() 
        : sites(Boson(QUANTUM_SITES, {"MaxOcc=", 1, "ConserveQNs=", false})),
          psi(sites),
          wavelength_coupling(AIRS_WAVELENGTHS),
          spectral_weights(AIRS_WAVELENGTHS) {
        
        // Initialize quantum state
        auto state = InitState(sites, "0");
        state.set(1, "1"); // Single photon input
        psi = MPS(state);
        normalize(psi);
        
        // Initialize spectral coupling based on wavelength
        for(int i = 0; i < AIRS_WAVELENGTHS; ++i) {
            // Weight by atmospheric absorption lines importance
            double lambda = 0.5 + 2.5 * i / AIRS_WAVELENGTHS; // 0.5-3.0 microns
            
            // Key absorption bands (simplified)
            if(lambda > 1.3 && lambda < 1.5) spectral_weights[i] = 2.0; // H2O
            else if(lambda > 1.6 && lambda < 1.8) spectral_weights[i] = 1.8; // CH4
            else if(lambda > 2.0 && lambda < 2.1) spectral_weights[i] = 1.5; // CO2
            else spectral_weights[i] = 1.0;
        }
    }
    
    /**
     * Encode spectrum into quantum state via Hamiltonian modulation
     */
    void encodeSpectrum(const vector<float>& spectrum) {
        auto ampo = AutoMPO(sites);
        
        // Map spectrum to quantum Hamiltonian
        for(int i = 0; i < min((int)spectrum.size(), AIRS_WAVELENGTHS); ++i) {
            // Spectral intensity modulates site potentials
            int site_idx = (i * QUANTUM_SITES) / AIRS_WAVELENGTHS + 1;
            double potential = spectrum[i] * spectral_weights[i];
            
            ampo += potential, "N", site_idx;
            
            // Add spectral correlations as hopping terms
            if(site_idx < QUANTUM_SITES) {
                double hop = -0.5 * sqrt(spectrum[i] * spectrum[min(i+1, (int)spectrum.size()-1)]);
                ampo += hop, "Adag", site_idx, "A", site_idx+1;
                ampo += hop, "Adag", site_idx+1, "A", site_idx;
            }
        }
        
        // Non-linear coupling for molecular signatures
        for(int i = 1; i <= QUANTUM_SITES; ++i) {
            ampo += 0.1, "N", i, "N", i; // Kerr effect
        }
        
        H = toMPO(ampo);
    }
    
    /**
     * Evolve quantum state and extract features
     */
    vector<float> extractFeatures() {
        // Time evolution
        auto tau = Complex_i * 1e-15; // femtosecond scale
        
        // TEBD evolution (simplified)
        for(int step = 0; step < 10; ++step) {
            vector<BondGate> gates;
            
            for(int b = 1; b < QUANTUM_SITES; ++b) {
                ITensor hterm = -0.5 * (op(sites,"Adag",b)*op(sites,"A",b+1) +
                                        op(sites,"Adag",b+1)*op(sites,"A",b));
                gates.push_back(BondGate(sites, b, b+1, 
                                        BondGate::tReal, real(tau), hterm));
            }
            
            for(auto& g : gates) {
                applyGate(g, psi);
            }
            normalize(psi);
        }
        
        // Extract quantum features via measurements
        vector<float> features(QUANTUM_FEATURES, 0.0);
        
        // Measure photon densities
        for(int i = 1; i <= QUANTUM_SITES; ++i) {
            psi.position(i);
            auto n_op = op(sites, "N", i);
            auto ket = psi(i);
            auto bra = dag(prime(ket, "Site"));
            features[i-1] = eltC(bra * n_op * ket).real();
        }
        
        // Measure entanglement features (simplified)
        for(int cut = 2; cut < QUANTUM_SITES; ++cut) {
            // Simplified entanglement measure
            features[QUANTUM_SITES + cut] = 0.0;
        }
        
        // Add correlation features
        for(int i = 1; i < QUANTUM_SITES-1; ++i) {
            auto corr_op = op(sites,"N",i) * op(sites,"N",i+1);
            psi.position(i);
            features[2*QUANTUM_SITES + i] = eltC(dag(psi(i))*corr_op*psi(i)).real();
        }
        
        return features;
    }
};

// ==================== NEBULA OPTICAL STAGE ====================
class NEBULAProcessor {
private:
    // Device buffers
    float* d_input;
    cufftComplex* d_field;
    cufftComplex* d_freq;
    float* d_intensity;
    float* d_output;
    
    // Learnable parameters
    float* d_amplitude_mask;
    float* d_phase_mask;
    float* d_W_output;
    float* d_b_output;
    
    // FFT plans
    cufftHandle plan_fwd;
    cufftHandle plan_inv;
    
    int batch_size;
    
public:
    NEBULAProcessor(int batch = 1) : batch_size(batch) {
        // Allocate device memory
        cudaMalloc(&d_input, batch * QUANTUM_FEATURES * sizeof(float));
        cudaMalloc(&d_field, batch * NEBULA_SIZE * NEBULA_SIZE * sizeof(cufftComplex));
        cudaMalloc(&d_freq, batch * NEBULA_SIZE * NEBULA_SIZE * sizeof(cufftComplex));
        cudaMalloc(&d_intensity, batch * NEBULA_SIZE * NEBULA_SIZE * sizeof(float));
        cudaMalloc(&d_output, batch * OUTPUT_TARGETS * sizeof(float));
        
        // Learnable parameters
        cudaMalloc(&d_amplitude_mask, NEBULA_SIZE * NEBULA_SIZE * sizeof(float));
        cudaMalloc(&d_phase_mask, NEBULA_SIZE * NEBULA_SIZE * sizeof(float));
        cudaMalloc(&d_W_output, OUTPUT_TARGETS * NEBULA_SIZE * NEBULA_SIZE * sizeof(float));
        cudaMalloc(&d_b_output, OUTPUT_TARGETS * sizeof(float));
        
        // Initialize masks
        initializeMasks();
        
        // Create FFT plans
        cufftPlan2d(&plan_fwd, NEBULA_SIZE, NEBULA_SIZE, CUFFT_C2C);
        cufftPlan2d(&plan_inv, NEBULA_SIZE, NEBULA_SIZE, CUFFT_C2C);
    }
    
    ~NEBULAProcessor() {
        cudaFree(d_input);
        cudaFree(d_field);
        cudaFree(d_freq);
        cudaFree(d_intensity);
        cudaFree(d_output);
        cudaFree(d_amplitude_mask);
        cudaFree(d_phase_mask);
        cudaFree(d_W_output);
        cudaFree(d_b_output);
        
        cufftDestroy(plan_fwd);
        cufftDestroy(plan_inv);
    }
    
    void initializeMasks() {
        // Initialize with Gaussian random values
        vector<float> h_amp(NEBULA_SIZE * NEBULA_SIZE);
        vector<float> h_phase(NEBULA_SIZE * NEBULA_SIZE);
        
        default_random_engine gen(1337);
        normal_distribution<float> dist(0.0, 0.1);
        
        for(int i = 0; i < NEBULA_SIZE * NEBULA_SIZE; ++i) {
            h_amp[i] = 1.0 + dist(gen);
            h_phase[i] = dist(gen);
        }
        
        cudaMemcpy(d_amplitude_mask, h_amp.data(), 
                  h_amp.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_phase_mask, h_phase.data(), 
                  h_phase.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    /**
     * Process quantum features through optical system
     */
    vector<float> process(const vector<float>& quantum_features) {
        // Upload quantum features
        cudaMemcpy(d_input, quantum_features.data(), 
                  quantum_features.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Encode as complex field
        encodeToComplexField<<<(NEBULA_SIZE*NEBULA_SIZE+255)/256, 256>>>(
            d_input, d_field, QUANTUM_FEATURES, NEBULA_SIZE);
        
        // Forward FFT (propagation to Fourier plane)
        cufftExecC2C(plan_fwd, d_field, d_freq, CUFFT_FORWARD);
        
        // Apply optical masks in Fourier domain
        applyOpticalMasks<<<(NEBULA_SIZE*NEBULA_SIZE+255)/256, 256>>>(
            d_freq, d_amplitude_mask, d_phase_mask, NEBULA_SIZE);
        
        // Inverse FFT (propagation to image plane)
        cufftExecC2C(plan_inv, d_freq, d_field, CUFFT_INVERSE);
        
        // Calculate intensity (photodetection)
        calculateIntensity<<<(NEBULA_SIZE*NEBULA_SIZE+255)/256, 256>>>(
            d_field, d_intensity, NEBULA_SIZE);
        
        // Linear output layer for atmospheric parameters
        computeOutput<<<(OUTPUT_TARGETS+31)/32, 32>>>(
            d_intensity, d_W_output, d_b_output, d_output,
            NEBULA_SIZE*NEBULA_SIZE, OUTPUT_TARGETS);
        
        // Download results
        vector<float> output(OUTPUT_TARGETS);
        cudaMemcpy(output.data(), d_output, 
                  OUTPUT_TARGETS * sizeof(float), cudaMemcpyDeviceToHost);
        
        return output;
    }
    
    /**
     * Train the optical masks using gradient descent
     */
    void updateParameters(const vector<float>& grad_output, float lr) {
        // Simplified gradient update
        // In practice, would implement full backpropagation through FFT

        // Update output layer
        updateOutputLayer<<<(OUTPUT_TARGETS*NEBULA_SIZE*NEBULA_SIZE+255)/256, 256>>>(
            d_W_output, d_b_output, grad_output.data(), lr,
            OUTPUT_TARGETS, NEBULA_SIZE*NEBULA_SIZE);
    }

    // Getters for checkpointing
    float* getAmplitudeMask() { return d_amplitude_mask; }
    float* getPhaseMask() { return d_phase_mask; }
};

// ==================== MAIN HYBRID MODEL ====================
class HybridArielModel {
private:
    QuantumSpectralProcessor quantum_stage;
    NEBULAProcessor nebula_stage;
    
    // Training parameters
    float learning_rate = 1e-3;
    int epoch = 0;
    
    // Data normalization parameters
    vector<float> spectrum_mean;
    vector<float> spectrum_std;
    
public:
    HybridArielModel() : nebula_stage(1) {
        spectrum_mean.resize(SPECTRUM_SIZE, 0.0);
        spectrum_std.resize(SPECTRUM_SIZE, 1.0);
        
        cout << "[HYBRID] Initialized Quantum-NEBULA model" << endl;
        cout << "  - Quantum sites: " << QUANTUM_SITES << endl;
        cout << "  - Quantum features: " << QUANTUM_FEATURES << endl;
        cout << "  - NEBULA size: " << NEBULA_SIZE << "x" << NEBULA_SIZE << endl;
        cout << "  - Output targets: " << OUTPUT_TARGETS << endl;
    }
    
    /**
     * Process single spectrum through full pipeline
     */
    vector<float> forward(const vector<float>& spectrum) {
        // Normalize spectrum
        vector<float> norm_spectrum(spectrum.size());
        for(size_t i = 0; i < spectrum.size(); ++i) {
            norm_spectrum[i] = (spectrum[i] - spectrum_mean[i]) / spectrum_std[i];
        }
        
        // Stage 1: Quantum processing
        quantum_stage.encodeSpectrum(norm_spectrum);
        auto quantum_features = quantum_stage.extractFeatures();
        
        // Stage 2: NEBULA optical processing
        auto predictions = nebula_stage.process(quantum_features);
        
        // Post-process predictions
        // Denormalize to physical units
        predictions[0] *= 100.0;  // CO2 percentage
        predictions[1] *= 100.0;  // H2O percentage  
        predictions[2] *= 100.0;  // CH4 percentage
        predictions[3] *= 100.0;  // NH3 percentage
        predictions[4] = 500.0 + predictions[4] * 2000.0; // Temperature (K)
        predictions[5] = 0.5 + predictions[5] * 2.0; // Radius (Jupiter radii)
        
        return predictions;
    }
    
    /**
     * Train on batch of Ariel data
     */
    float trainBatch(const vector<vector<float>>& spectra, 
                    const vector<vector<float>>& targets) {
        float total_loss = 0.0;
        
        for(size_t i = 0; i < spectra.size(); ++i) {
            // Forward pass
            auto predictions = forward(spectra[i]);
            
            // Compute loss (MSE)
            float loss = 0.0;
            vector<float> grad_output(OUTPUT_TARGETS);
            
            for(int j = 0; j < OUTPUT_TARGETS; ++j) {
                float diff = predictions[j] - targets[i][j];
                loss += diff * diff;
                grad_output[j] = 2.0 * diff / OUTPUT_TARGETS;
            }
            
            total_loss += loss;
            
            // Backward pass (simplified)
            nebula_stage.updateParameters(grad_output, learning_rate);
        }
        
        return total_loss / spectra.size();
    }
    
    /**
     * Load and preprocess Ariel data
     */
    void loadArielData(const string& data_path) {
        cout << "[DATA] Loading Ariel data from: " << data_path << endl;
        
        // Load preprocessed numpy files
        auto airs_data = cnpy::npy_load(data_path + "/data_train.npy");
        auto fgs_data = cnpy::npy_load(data_path + "/data_train_FGS.npy");

        // Extract dimensions
        int n_samples = airs_data.shape[0];
        int n_time = airs_data.shape[1];
        int n_wavelengths = airs_data.shape[2];

        cout << "  - Samples: " << n_samples << endl;
        cout << "  - Time bins: " << n_time << endl;
        cout << "  - Wavelengths: " << n_wavelengths << endl;

        // Calculate normalization parameters
        const float* airs_ptr = airs_data.data<float>();
        
        for(int w = 0; w < n_wavelengths; ++w) {
            float sum = 0.0, sum_sq = 0.0;
            int count = 0;
            
            for(int s = 0; s < n_samples; ++s) {
                for(int t = 0; t < n_time; ++t) {
                    float val = airs_ptr[s*n_time*n_wavelengths + t*n_wavelengths + w];
                    sum += val;
                    sum_sq += val * val;
                    count++;
                }
            }
            
            spectrum_mean[w] = sum / count;
            spectrum_std[w] = sqrt(sum_sq / count - spectrum_mean[w]*spectrum_mean[w]);
            
            if(spectrum_std[w] < 1e-6) spectrum_std[w] = 1.0;
        }
        
        cout << "[DATA] Normalization parameters calculated" << endl;
    }
    
    /**
     * Save model checkpoint
     */
    void saveCheckpoint(const string& path) {
        ofstream file(path, ios::binary);

        // Save NEBULA parameters
        float* h_amp = new float[NEBULA_SIZE * NEBULA_SIZE];
        float* h_phase = new float[NEBULA_SIZE * NEBULA_SIZE];

        cudaMemcpy(h_amp, nebula_stage.getAmplitudeMask(),
                  NEBULA_SIZE * NEBULA_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_phase, nebula_stage.getPhaseMask(),
                  NEBULA_SIZE * NEBULA_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        file.write((char*)h_amp, NEBULA_SIZE * NEBULA_SIZE * sizeof(float));
        file.write((char*)h_phase, NEBULA_SIZE * NEBULA_SIZE * sizeof(float));

        delete[] h_amp;
        delete[] h_phase;

        file.close();
        cout << "[MODEL] Checkpoint saved to: " << path << endl;
    }
    
    /**
     * Generate submission for Kaggle
     */
    void generateSubmission(const string& test_path, const string& output_path) {
        cout << "[SUBMISSION] Generating predictions..." << endl;
        
        // Load test data
        auto test_data = cnpy::npy_load(test_path + "/data_test.npy");
        int n_test = test_data.shape[0];

        ofstream submission(output_path);
        submission << "planet_ID,CO2,H2O,CH4,NH3,temperature,radius" << endl;

        const float* test_ptr = test_data.data<float>();
        
        for(int i = 0; i < n_test; ++i) {
            // Extract spectrum for this sample
            vector<float> spectrum(SPECTRUM_SIZE);
            
            // Average over time dimension
            for(int w = 0; w < AIRS_WAVELENGTHS; ++w) {
                float avg = 0.0;
                for(int t = 0; t < TIME_BINS; ++t) {
                    avg += test_ptr[i*TIME_BINS*AIRS_WAVELENGTHS + t*AIRS_WAVELENGTHS + w];
                }
                spectrum[w] = avg / TIME_BINS;
            }
            
            // Process through model
            auto predictions = forward(spectrum);
            
            // Write to submission
            submission << i << ",";
            for(int j = 0; j < OUTPUT_TARGETS; ++j) {
                submission << predictions[j];
                if(j < OUTPUT_TARGETS-1) submission << ",";
            }
            submission << endl;
        }
        
        submission.close();
        cout << "[SUBMISSION] Written to: " << output_path << endl;
    }
};



#endif // HYBRID_ARIEL_MODEL_HPP
