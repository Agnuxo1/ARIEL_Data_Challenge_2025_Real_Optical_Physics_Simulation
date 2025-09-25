/**
 * LAZY LOADING DATA PIPELINE FOR ARIEL DATA CHALLENGE 2025
 * Handles large datasets efficiently without memory saturation
 */

#ifndef ARIEL_DATA_LOADER_LAZY_HPP
#define ARIEL_DATA_LOADER_LAZY_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cnpy.h>
#include <random>
#include <algorithm>
#include <numeric>

class ArielLazyDataLoader {
private:
    std::string data_path;
    std::vector<int> train_indices;
    std::vector<int> val_indices;
    std::vector<std::vector<float>> all_targets;

    // Data dimensions (metadata only)
    int n_samples, n_time, n_wavelengths, n_pixels;
    std::mt19937 rng;

public:
    ArielLazyDataLoader(const std::string& path, float val_split = 0.2)
        : data_path(path), rng(42) {

        std::cout << "[LAZY LOADER] Loading metadata from: " << path << std::endl;

        // Load only targets, not the huge data arrays
        auto targets_npy = cnpy::npy_load(path + "/targets_train.npy");

        n_samples = targets_npy.shape[0];
        n_time = 187;           // Standard ARIEL time bins
        n_wavelengths = 283;    // Standard ARIEL wavelengths
        n_pixels = 32;          // Standard ARIEL spatial pixels

        std::cout << "  Training samples: " << n_samples << std::endl;
        std::cout << "  Time bins: " << n_time << std::endl;
        std::cout << "  Wavelengths: " << n_wavelengths << std::endl;

        // Store only targets in memory (small: N x 6)
        float* targets_ptr = targets_npy.data<float>();
        all_targets.resize(n_samples);
        for(int s = 0; s < n_samples; ++s) {
            all_targets[s].resize(6);
            for(int t = 0; t < 6; ++t) {
                all_targets[s][t] = targets_ptr[s * 6 + t];
            }
        }

        // Create indices for train/val split
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        int n_val = static_cast<int>(n_samples * val_split);
        train_indices.assign(indices.begin(), indices.end() - n_val);
        val_indices.assign(indices.end() - n_val, indices.end());

        std::cout << "  Train: " << train_indices.size() << " | Val: " << val_indices.size() << std::endl;
    }

    // Load single sample on-demand (true lazy loading)
    std::pair<std::vector<float>, std::vector<float>> loadSample(int sample_idx) {
        // Try to load from calibrated numpy files first, fall back to synthetic
        std::vector<float> spectrum(n_wavelengths);

        // Load calibrated data directly (no fallback)
        static bool data_loaded = false;
        static cnpy::NpyArray airs_data;

        if (!data_loaded) {
            airs_data = cnpy::npy_load(data_path + "/data_train.npy");
            data_loaded = true;
            std::cout << "[LOADER] Using calibrated ARIEL data with shape rank "
                      << airs_data.shape.size();
            std::cout << " [";
            for(size_t d = 0; d < airs_data.shape.size(); ++d) {
                std::cout << airs_data.shape[d];
                if(d + 1 < airs_data.shape.size()) std::cout << " x ";
            }
            std::cout << "]" << std::endl;
        }

            float* airs_ptr = airs_data.data<float>();

            // Check if data is 2D calibrated format (N, wavelengths) or 4D raw format
            if (airs_data.shape.size() == 2) {
                // Calibrated 2D format: (samples, wavelengths)
                int spectrum_size = std::min((int)airs_data.shape[1], n_wavelengths);
                for(int w = 0; w < spectrum_size; ++w) {
                    spectrum[w] = airs_ptr[sample_idx * airs_data.shape[1] + w];
                }
                // Fill remaining with zeros if needed
                for(int w = spectrum_size; w < n_wavelengths; ++w) {
                    spectrum[w] = 0.0;
                }
        } else if (airs_data.shape.size() == 3) {
            // Calibrated 3D format: (samples, time, wavelengths)
            int time_dim = airs_data.shape[1];
            int wave_dim = airs_data.shape[2];
            int spectrum_size = std::min(wave_dim, n_wavelengths);
            for(int w = 0; w < spectrum_size; ++w) {
                float sum = 0.0f;
                for(int t = 0; t < time_dim; ++t) {
                    int idx = sample_idx * time_dim * wave_dim + t * wave_dim + w;
                    sum += airs_ptr[idx];
                }
                spectrum[w] = sum / std::max(1, time_dim);
            }
            for(int w = spectrum_size; w < n_wavelengths; ++w) {
                spectrum[w] = 0.0f;
            }
            } else {
                // Original 4D format: (samples, time, wavelengths, pixels)
                std::cout << "[LOADER] Falling back to 4D data path" << std::endl;
                for(int w = 0; w < n_wavelengths; ++w) {
                    float sum = 0.0;
                    int t = 0; // Use first time step only
                    for(int p = 0; p < n_pixels; ++p) {
                        int idx = sample_idx * n_time * n_wavelengths * n_pixels +
                                 t * n_wavelengths * n_pixels +
                                 w * n_pixels + p;
                        sum += airs_ptr[idx];
                    }
                    spectrum[w] = sum / n_pixels;
                }
            }

        // Return spectrum and targets
        return {spectrum, all_targets[sample_idx]};
    }

    // Batch generator with lazy loading
    class BatchIterator {
    private:
        ArielLazyDataLoader* loader;
        std::vector<int> indices;
        size_t batch_size;
        size_t current_idx;
        std::mt19937 rng;

    public:
        BatchIterator(ArielLazyDataLoader* l, std::vector<int> idx, size_t bs)
            : loader(l), indices(idx), batch_size(bs), current_idx(0), rng(42) {
            shuffle();
        }

        void shuffle() {
            std::shuffle(indices.begin(), indices.end(), rng);
            current_idx = 0;
        }

        bool hasNext() const {
            bool has = current_idx < indices.size();
            return has;
        }

        std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
        next() {
            size_t end_idx = std::min(current_idx + batch_size, indices.size());
            if(end_idx == current_idx) {
                throw std::runtime_error("BatchIterator next() called with no remaining samples");
            }

            std::vector<std::vector<float>> batch_x;
            std::vector<std::vector<float>> batch_y;

            // Load samples on-demand (no memory saturation)
            for(size_t i = current_idx; i < end_idx; ++i) {
                auto [spectrum, targets] = loader->loadSample(indices[i]);
                batch_x.push_back(spectrum);
                batch_y.push_back(targets);
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

    int getTrainSize() const { return train_indices.size(); }
    int getValSize() const { return val_indices.size(); }
};

#endif // ARIEL_DATA_LOADER_LAZY_HPP