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

struct ArielSample {
    std::vector<float> airs_spectrum;  // 282 wavelengths
    std::vector<float> fgs_photometry; // FGS1 data
    std::vector<float> targets;        // CO2, H2O, CH4, NH3, T, R
    int planet_id;
};

class ArielDataLoader {
private:
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
        // Load AIRS spectra
        auto airs = cnpy::npy_load(path + "/data_train.npy");
        auto fgs = cnpy::npy_load(path + "/data_train_FGS.npy");
        
        // Load targets
        auto targets = loadTargets(path + "/train_labels.csv");
        
        int n_samples = airs.shape[0];
        int n_time = airs.shape[1];
        int n_wavelengths = airs.shape[2];
        int n_pixels = airs.shape[3];
        
        std::cout << "  Training samples: " << n_samples << std::endl;
        std::cout << "  Time bins: " << n_time << std::endl;
        std::cout << "  Wavelengths: " << n_wavelengths << std::endl;
        
        float* airs_ptr = airs.data<float>();
        float* fgs_ptr = fgs.data<float>();
        
        // Process each sample
        for(int s = 0; s < n_samples; ++s) {
            ArielSample sample;
            sample.planet_id = s;
            
            // Average spectrum over time and spatial pixels
            sample.airs_spectrum.resize(n_wavelengths);
            for(int w = 0; w < n_wavelengths; ++w) {
                float sum = 0.0;
                int count = 0;
                
                for(int t = 0; t < n_time; ++t) {
                    for(int p = 0; p < n_pixels; ++p) {
                        int idx = s * n_time * n_wavelengths * n_pixels +
                                 t * n_wavelengths * n_pixels +
                                 w * n_pixels + p;
                        sum += airs_ptr[idx];
                        count++;
                    }
                }
                sample.airs_spectrum[w] = sum / count;
            }
            
            // Process FGS photometry
            sample.fgs_photometry.resize(32*32);
            // ... similar averaging for FGS ...
            
            // Add targets if available
            if(s < targets.size()) {
                sample.targets = targets[s];
            }
            
            train_data.push_back(sample);
        }
        
        // Split into train/val
        int n_val = static_cast<int>(train_data.size() * val_split);
        
        // Shuffle and split
        std::shuffle(train_data.begin(), train_data.end(), rng);
        
        val_data.insert(val_data.end(), 
                       train_data.end() - n_val, 
                       train_data.end());
        train_data.erase(train_data.end() - n_val, train_data.end());
        
        std::cout << "  Train: " << train_data.size() 
                  << " | Val: " << val_data.size() << std::endl;
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
    
    // Batch generator for training
    class BatchIterator {
    private:
        std::vector<ArielSample>& data;
        size_t batch_size;
        size_t current_idx;
        std::mt19937 rng;
        
    public:
        BatchIterator(std::vector<ArielSample>& d, size_t bs) 
            : data(d), batch_size(bs), current_idx(0), rng(42) {
            shuffle();
        }
        
        void shuffle() {
            std::shuffle(data.begin(), data.end(), rng);
            current_idx = 0;
        }
        
        bool hasNext() const {
            return current_idx < data.size();
        }
        
        std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> 
        next() {
            size_t end_idx = std::min(current_idx + batch_size, data.size());
            
            std::vector<std::vector<float>> batch_x;
            std::vector<std::vector<float>> batch_y;
            
            for(size_t i = current_idx; i < end_idx; ++i) {
                batch_x.push_back(data[i].airs_spectrum);
                batch_y.push_back(data[i].targets);
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
        return BatchIterator(train_data, batch_size);
    }
    
    BatchIterator getValIterator(size_t batch_size) {
        return BatchIterator(val_data, batch_size);
    }
    
    const std::vector<ArielSample>& getTestData() const {
        return test_data;
    }
};

#endif // ARIEL_DATA_LOADER_HPP
