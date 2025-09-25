/**
 * DATA LOADER FOR ARIEL DATA CHALLENGE 2025
 * Handles calibrated spectroscopic data loading
 */

#ifndef ARIEL_DATA_LOADER_HPP
#define ARIEL_DATA_LOADER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cnpy.h>
#include <random>
#include <algorithm>
#include <numeric>

struct ArielSample {
    std::vector<float> airs_spectrum;  // 283 wavelengths
    std::vector<float> fgs_photometry; // FGS1 data
    std::vector<float> targets;        // CO2, H2O, CH4, NH3, T, R
    int planet_id;
};

class ArielDataLoader {
private:
    // Lazy loading data
    std::string data_path;
    std::vector<int> train_indices;
    std::vector<int> val_indices;
    std::vector<std::vector<float>> all_targets;

    // Data dimensions
    int n_samples, n_time, n_wavelengths, n_pixels;

    // Legacy (deprecated)
    std::vector<ArielSample> train_data;
    std::vector<ArielSample> val_data;
    std::vector<ArielSample> test_data;

    std::mt19937 rng;
    
public:
    ArielDataLoader(const std::string& data_path, float val_split = 0.2) 
        : rng(42) {
        
        std::cout << "[LOADER] Loading Ariel data from: " << data_path << std::endl;
        
        // Load preprocessed numpy arrays
        loadTrainingData(data_path, val_split);
        loadTestData(data_path);
    }
    
    void loadTrainingData(const std::string& path, float val_split) {
        // Store paths for lazy loading
        data_path = path;

        // Load only metadata
        auto airs = cnpy::npy_load(path + "/data_train.npy");
        auto targets_npy = cnpy::npy_load(path + "/targets_train.npy");

        n_samples = airs.shape[0];
        n_time = airs.shape[1];
        n_wavelengths = airs.shape[2];
        n_pixels = airs.shape[3];

        std::cout << "  Training samples: " << n_samples << std::endl;
        std::cout << "  Time bins: " << n_time << std::endl;
        std::cout << "  Wavelengths: " << n_wavelengths << std::endl;

        // Store targets in memory (small)
        float* targets_ptr = targets_npy.data<float>();
        all_targets.resize(n_samples);
        for(int s = 0; s < n_samples; ++s) {
            all_targets[s].resize(6);
            for(int t = 0; t < 6; ++t) {
                all_targets[s][t] = targets_ptr[s * 6 + t];
            }
        }

        // Create sample indices for train/val split
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(42));

        int n_val = static_cast<int>(n_samples * val_split);
        train_indices.assign(indices.begin(), indices.end() - n_val);
        val_indices.assign(indices.end() - n_val, indices.end());

        std::cout << "  Train: " << train_indices.size() << " | Val: " << val_indices.size() << std::endl;
    }

    // Load single sample on-demand (lazy loading)
    ArielSample loadSample(int sample_idx) {
        // Load data files (this should be cached for efficiency)
        static auto airs = cnpy::npy_load(data_path + "/data_train.npy");
        static auto fgs = cnpy::npy_load(data_path + "/data_train_FGS.npy");

        float* airs_ptr = airs.data<float>();
        float* fgs_ptr = fgs.data<float>();

        ArielSample sample;
        sample.planet_id = sample_idx;

        // Extract spectrum for this sample only (efficient)
        sample.airs_spectrum.resize(n_wavelengths);
        for(int w = 0; w < n_wavelengths; ++w) {
            float sum = 0.0;
            // Use first time step, average over spatial pixels
            int t = 0;
            for(int p = 0; p < n_pixels; ++p) {
                int idx = sample_idx * n_time * n_wavelengths * n_pixels +
                         t * n_wavelengths * n_pixels +
                         w * n_pixels + p;
                sum += airs_ptr[idx];
            }
            sample.airs_spectrum[w] = sum / n_pixels;
        }

        // FGS photometry (simplified)
        sample.fgs_photometry.resize(1);
        sample.fgs_photometry[0] = fgs_ptr[sample_idx * n_time * 32 * 32]; // Just first value

        // Targets (already in memory)
        sample.targets = all_targets[sample_idx];

        return sample;
    }
    
    void loadTestData(const std::string& path) {
        // Similar to training data but without targets
        auto airs = cnpy::npy_load(path + "/data_test.npy");
        
        // Process test samples...
        // (Similar code to training data processing)
    }
    
    std::vector<std::vector<float>> loadTargets(const std::string& csv_path) {
        std::vector<std::vector<float>> targets;
        std::ifstream file(csv_path);
        std::string line;
        
        // Skip header
        std::getline(file, line);
        
        while(std::getline(file, line)) {
            std::vector<float> row;
            size_t pos = 0;
            
            // Skip planet_ID
            pos = line.find(',');
            line = line.substr(pos + 1);
            
            // Parse 6 target values
            for(int i = 0; i < 6; ++i) {
                pos = line.find(',');
                float val = std::stof(line.substr(0, pos));
                row.push_back(val);
                if(pos != std::string::npos) {
                    line = line.substr(pos + 1);
                }
            }
            
            targets.push_back(row);
        }
        
        return targets;
    }
    
    // Batch generator for training (lazy loading)
    class BatchIterator {
    private:
        ArielDataLoader* loader;
        std::vector<int>& indices;
        size_t batch_size;
        size_t current_idx;
        std::mt19937 rng;

    public:
        BatchIterator(ArielDataLoader* l, std::vector<int>& idx, size_t bs)
            : loader(l), indices(idx), batch_size(bs), current_idx(0), rng(42) {
            shuffle();
        }
        
        void shuffle() {
            std::shuffle(indices.begin(), indices.end(), rng);
            current_idx = 0;
        }

        bool hasNext() const {
            return current_idx < indices.size();
        }

        std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
        next() {
            size_t end_idx = std::min(current_idx + batch_size, indices.size());

            std::vector<std::vector<float>> batch_x;
            std::vector<std::vector<float>> batch_y;

            // Load samples on-demand (lazy loading)
            for(size_t i = current_idx; i < end_idx; ++i) {
                ArielSample sample = loader->loadSample(indices[i]);
                batch_x.push_back(sample.airs_spectrum);
                batch_y.push_back(sample.targets);
            }

            current_idx = end_idx;
            return {batch_x, batch_y};
        }
        
        void reset() {
            current_idx = 0;
            shuffle();
        }
    };
    
    BatchIterator getTrainIterator(size_t batch_size) {
        return BatchIterator(this, train_indices, batch_size);
    }

    BatchIterator getValIterator(size_t batch_size) {
        return BatchIterator(this, val_indices, batch_size);
    }
    
    const std::vector<ArielSample>& getTestData() const {
        return test_data;
    }
};

#endif // ARIEL_DATA_LOADER_HPP
