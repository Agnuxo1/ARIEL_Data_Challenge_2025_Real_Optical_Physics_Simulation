/**
 * SIMPLE ARIEL TRAINER - CPU ONLY VERSION
 * For testing and debugging the training process
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <iomanip>
#include <cnpy.h>

using namespace std;

// Simple model configuration
constexpr int N_WAVELENGTHS = 283;
constexpr int N_TIME_BINS = 187;
constexpr int OUTPUT_TARGETS = 566;  // 283 wavelengths + 283 sigmas
constexpr int N_TRAINING_SAMPLES = 1100;

class SimpleArielModel {
private:
    vector<vector<float>> weights;
    vector<float> bias;
    vector<float> spectrum_mean;
    vector<float> spectrum_std;
    
public:
    SimpleArielModel() {
        // Initialize weights randomly
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float> dis(0.0, 0.1);
        
        weights.resize(OUTPUT_TARGETS, vector<float>(N_WAVELENGTHS));
        bias.resize(OUTPUT_TARGETS);
        
        for(int i = 0; i < OUTPUT_TARGETS; ++i) {
            for(int j = 0; j < N_WAVELENGTHS; ++j) {
                weights[i][j] = dis(gen);
            }
            bias[i] = dis(gen);
        }
        
        // Initialize normalization
        spectrum_mean.resize(N_WAVELENGTHS, 0.0);
        spectrum_std.resize(N_WAVELENGTHS, 1.0);
        
        cout << "Simple ARIEL Model initialized with " << OUTPUT_TARGETS << " outputs" << endl;
    }
    
    vector<float> forward(const vector<float>& spectrum) {
        vector<float> output(OUTPUT_TARGETS);
        
        // Normalize spectrum
        vector<float> norm_spectrum(N_WAVELENGTHS);
        for(int i = 0; i < N_WAVELENGTHS; ++i) {
            norm_spectrum[i] = (spectrum[i] - spectrum_mean[i]) / (spectrum_std[i] + 1e-8);
        }
        
        // Forward pass
        for(int i = 0; i < OUTPUT_TARGETS; ++i) {
            output[i] = bias[i];
            for(int j = 0; j < N_WAVELENGTHS; ++j) {
                output[i] += weights[i][j] * norm_spectrum[j];
            }
        }
        
        // Post-process for submission format
        // First 283 values: wavelength predictions (wl_1 to wl_283)
        // Next 283 values: sigma predictions (sigma_1 to sigma_283)
        
        // Normalize wavelength predictions to reasonable range
        for(int i = 0; i < 283; ++i) {
            output[i] = 0.4 + tanh(output[i]) * 0.2;  // Range 0.4-0.6
        }
        
        // Normalize sigma predictions to reasonable range
        for(int i = 283; i < OUTPUT_TARGETS; ++i) {
            output[i] = 0.01 + abs(tanh(output[i])) * 0.02;  // Range 0.01-0.03
        }
        
        return output;
    }
    
    float trainBatch(const vector<vector<float>>& spectra, const vector<vector<float>>& targets, float lr) {
        float total_loss = 0.0;
        
        for(size_t i = 0; i < spectra.size(); ++i) {
            // Forward pass
            vector<float> predictions = forward(spectra[i]);
            
            // Compute loss (MSE) - only on first 6 targets for atmospheric parameters
            float loss = 0.0;
            vector<float> grad_output(OUTPUT_TARGETS, 0.0);
            
            // Only train on atmospheric parameters (first 6 targets)
            for(int j = 0; j < min(6, (int)targets[i].size()); ++j) {
                float diff = predictions[j] - targets[i][j];
                loss += diff * diff;
                grad_output[j] = 2.0 * diff / 6.0;  // Only update first 6 outputs
            }
            
            total_loss += loss;
            
            // Update weights (simplified gradient descent)
            for(int k = 0; k < OUTPUT_TARGETS; ++k) {
                for(int j = 0; j < N_WAVELENGTHS; ++j) {
                    weights[k][j] -= lr * grad_output[k] * spectra[i][j];
                }
                bias[k] -= lr * grad_output[k];
            }
        }
        
        return total_loss / spectra.size();
    }
    
    void generateSubmission(const string& test_path, const string& output_path) {
        cout << "Generating submission..." << endl;
        
        // Load test data
        auto test_data = cnpy::npy_load(test_path + "/data_test.npy");
        auto planet_ids = cnpy::npy_load(test_path + "/test_planet_ids.npy");
        
        float* test_data_ptr = test_data.data<float>();
        int* planet_ids_ptr = planet_ids.data<int>();
        
        int n_test_planets = test_data.shape[0];
        cout << "Test planets: " << n_test_planets << endl;
        
        // Create submission file
        ofstream submission(output_path);
        
        // Write header
        submission << "planet_id";
        for(int i = 1; i <= 283; ++i) {
            submission << ",wl_" << i;
        }
        for(int i = 1; i <= 283; ++i) {
            submission << ",sigma_" << i;
        }
        submission << endl;
        
        // Generate predictions
        for(int i = 0; i < n_test_planets; ++i) {
            if(i % 100 == 0) {
                cout << "Processing planet " << i+1 << "/" << n_test_planets << endl;
            }
            
            // Extract spectrum (average over time)
            vector<float> spectrum(N_WAVELENGTHS);
            for(int j = 0; j < N_WAVELENGTHS; ++j) {
                spectrum[j] = 0.0;
                for(int t = 0; t < N_TIME_BINS; ++t) {
                    spectrum[j] += test_data_ptr[i * N_TIME_BINS * N_WAVELENGTHS + t * N_WAVELENGTHS + j];
                }
                spectrum[j] /= N_TIME_BINS;
            }
            
            // Get predictions
            vector<float> predictions = forward(spectrum);
            
            // Write to submission
            submission << planet_ids_ptr[i];
            for(int j = 0; j < OUTPUT_TARGETS; ++j) {
                submission << "," << fixed << setprecision(6) << predictions[j];
            }
            submission << endl;
        }
        
        submission.close();
        cout << "Submission saved to: " << output_path << endl;
    }
};

int main(int argc, char* argv[]) {
    cout << "=================================" << endl;
    cout << "SIMPLE ARIEL TRAINER (CPU ONLY)" << endl;
    cout << "=================================" << endl;
    
    // Parse arguments
    string data_path = "./calibrated_data";
    string output_path = "./simple_output";
    int epochs = 100;
    
    for(int i = 1; i < argc; i += 2) {
        if(i + 1 < argc) {
            if(string(argv[i]) == "--data") {
                data_path = argv[i + 1];
            } else if(string(argv[i]) == "--output") {
                output_path = argv[i + 1];
            } else if(string(argv[i]) == "--epochs") {
                epochs = atoi(argv[i + 1]);
            }
        }
    }
    
    cout << "Configuration:" << endl;
    cout << "  Data path: " << data_path << endl;
    cout << "  Output path: " << output_path << endl;
    cout << "  Epochs: " << epochs << endl;
    cout << "  Output targets: " << OUTPUT_TARGETS << endl;
    cout << "=================================" << endl;
    
    // Create output directory
    system(("mkdir " + output_path).c_str());
    
    // Initialize model
    SimpleArielModel model;
    
    // Load training data
    cout << "Loading training data..." << endl;
    auto train_data = cnpy::npy_load(data_path + "/data_train.npy");
    auto targets_data = cnpy::npy_load(data_path + "/targets_train.npy");
    
    float* train_data_ptr = train_data.data<float>();
    float* targets_data_ptr = targets_data.data<float>();
    
    int n_train_planets = train_data.shape[0];
    cout << "Training planets: " << n_train_planets << endl;
    
    // Prepare training data
    vector<vector<float>> spectra(n_train_planets, vector<float>(N_WAVELENGTHS));
    vector<vector<float>> targets(n_train_planets, vector<float>(6));
    
    for(int i = 0; i < n_train_planets; ++i) {
        // Extract spectrum (average over time)
        for(int j = 0; j < N_WAVELENGTHS; ++j) {
            spectra[i][j] = 0.0;
            for(int t = 0; t < N_TIME_BINS; ++t) {
                spectra[i][j] += train_data_ptr[i * N_TIME_BINS * N_WAVELENGTHS + t * N_WAVELENGTHS + j];
            }
            spectra[i][j] /= N_TIME_BINS;
        }
        
        // Extract targets
        for(int j = 0; j < 6; ++j) {
            targets[i][j] = targets_data_ptr[i * 6 + j];
        }
    }
    
    cout << "Starting training..." << endl;
    
    // Training loop
    float learning_rate = 0.001;
    for(int epoch = 0; epoch < epochs; ++epoch) {
        auto start = chrono::high_resolution_clock::now();
        
        float loss = model.trainBatch(spectra, targets, learning_rate);
        
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(end - start);
        
        cout << "Epoch " << epoch + 1 << "/" << epochs 
             << " | Loss: " << fixed << setprecision(6) << loss
             << " | Time: " << duration.count() << "s" << endl;
        
        // Decay learning rate
        learning_rate *= 0.98;
        
        // Save checkpoint every 10 epochs
        if((epoch + 1) % 10 == 0) {
            cout << "Checkpoint saved at epoch " << epoch + 1 << endl;
        }
    }
    
    cout << "Training complete!" << endl;
    
    // Generate submission
    model.generateSubmission(data_path, output_path + "/submission.csv");
    
    cout << "=================================" << endl;
    cout << "SIMPLE TRAINING COMPLETE!" << endl;
    cout << "=================================" << endl;
    
    return 0;
}
