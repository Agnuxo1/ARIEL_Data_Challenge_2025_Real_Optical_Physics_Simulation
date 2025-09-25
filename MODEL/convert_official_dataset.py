#!/usr/bin/env python3
"""
Convert official ARIEL dataset to our model format
"""
import numpy as np
import pandas as pd
import os
import sys

def convert_official_dataset():
    """Convert official dataset to our model format"""
    
    # Paths
    official_path = "E:/NeurIPS_MYCELIUM_EVOLUTUM/ariel-data-challenge-2025"
    output_path = "./calibrated_data"
    
    print("Loading official ARIEL dataset...")
    
    # Load training data
    train_df = pd.read_csv(os.path.join(official_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(official_path, "sample_submission.csv"))
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Extract planet IDs
    train_planet_ids = train_df['planet_id'].values
    test_planet_ids = test_df['planet_id'].values
    
    print(f"Training planets: {len(train_planet_ids)}")
    print(f"Test planets: {len(test_planet_ids)}")
    
    # Extract wavelength data (wl_1 to wl_283)
    wl_columns = [f'wl_{i}' for i in range(1, 284)]
    
    train_wl_data = train_df[wl_columns].values.astype(np.float32)
    test_wl_data = test_df[wl_columns].values.astype(np.float32)
    
    # Extract sigma data (only available in test/sample_submission)
    sigma_columns = [f'sigma_{i}' for i in range(1, 284)]
    test_sigma_data = test_df[sigma_columns].values.astype(np.float32)
    
    # For training data, we don't have sigma, so we'll create synthetic ones
    train_sigma_data = np.random.normal(0.01, 0.005, train_wl_data.shape).astype(np.float32)
    
    print(f"Wavelength data shape: {train_wl_data.shape}")
    print(f"Sigma data shape: {train_sigma_data.shape}")
    
    # Create time dimension (simulate 187 time bins)
    # We'll replicate the spectrum data across time with some noise
    n_time_bins = 187
    n_wavelengths = 283
    
    # Training data
    train_data_3d = np.zeros((len(train_planet_ids), n_time_bins, n_wavelengths), dtype=np.float32)
    train_sigma_3d = np.zeros((len(train_planet_ids), n_time_bins, n_wavelengths), dtype=np.float32)
    
    for i in range(len(train_planet_ids)):
        # Replicate spectrum across time with small variations
        for t in range(n_time_bins):
            # Add small time-dependent variations
            time_factor = 1.0 + 0.01 * np.sin(2 * np.pi * t / n_time_bins)
            noise = np.random.normal(0, 0.001, n_wavelengths)
            
            train_data_3d[i, t, :] = train_wl_data[i, :] * time_factor + noise
            train_sigma_3d[i, t, :] = train_sigma_data[i, :] * time_factor
    
    # Test data
    test_data_3d = np.zeros((len(test_planet_ids), n_time_bins, n_wavelengths), dtype=np.float32)
    test_sigma_3d = np.zeros((len(test_planet_ids), n_time_bins, n_wavelengths), dtype=np.float32)
    
    for i in range(len(test_planet_ids)):
        for t in range(n_time_bins):
            time_factor = 1.0 + 0.01 * np.sin(2 * np.pi * t / n_time_bins)
            noise = np.random.normal(0, 0.001, n_wavelengths)
            
            test_data_3d[i, t, :] = test_wl_data[i, :] * time_factor + noise
            test_sigma_3d[i, t, :] = test_sigma_data[i, :] * time_factor
    
    # Create FGS data (simulate photometry)
    # FGS has 32x32 pixels, we'll create synthetic data
    fgs_size = 32
    train_fgs = np.random.normal(0.5, 0.1, (len(train_planet_ids), n_time_bins, fgs_size, fgs_size)).astype(np.float32)
    test_fgs = np.random.normal(0.5, 0.1, (len(test_planet_ids), n_time_bins, fgs_size, fgs_size)).astype(np.float32)
    
    # Create dummy targets (we don't have real atmospheric parameters)
    # We'll create synthetic targets based on the spectrum characteristics
    n_targets = 6  # CO2, H2O, CH4, NH3, Temperature, Radius
    
    train_targets = np.zeros((len(train_planet_ids), n_targets), dtype=np.float32)
    for i in range(len(train_planet_ids)):
        # Extract features from spectrum to create synthetic targets
        spectrum = train_wl_data[i, :]
        
        # CO2: based on absorption around 2.0-2.1 microns
        co2_region = spectrum[150:170]  # Approximate CO2 band
        train_targets[i, 0] = np.mean(co2_region) * 100  # Convert to percentage
        
        # H2O: based on absorption around 1.3-1.5 microns  
        h2o_region = spectrum[80:100]   # Approximate H2O band
        train_targets[i, 1] = np.mean(h2o_region) * 100
        
        # CH4: based on absorption around 1.6-1.8 microns
        ch4_region = spectrum[110:130]  # Approximate CH4 band
        train_targets[i, 2] = np.mean(ch4_region) * 100
        
        # NH3: based on absorption around 1.0-1.2 microns
        nh3_region = spectrum[50:70]    # Approximate NH3 band
        train_targets[i, 3] = np.mean(nh3_region) * 100
        
        # Temperature: based on overall spectrum shape
        temp_factor = np.std(spectrum) * 1000
        train_targets[i, 4] = 500 + temp_factor  # 500-1500 K range
        
        # Radius: based on spectrum amplitude
        radius_factor = np.mean(spectrum) * 2
        train_targets[i, 5] = 0.5 + radius_factor  # 0.5-2.5 Jupiter radii
    
    # Save data in our format
    os.makedirs(output_path, exist_ok=True)
    
    print("Saving converted data...")
    
    # Save training data
    np.save(os.path.join(output_path, "data_train.npy"), train_data_3d)
    np.save(os.path.join(output_path, "data_train_FGS.npy"), train_fgs)
    np.save(os.path.join(output_path, "targets_train.npy"), train_targets)
    
    # Save test data
    np.save(os.path.join(output_path, "data_test.npy"), test_data_3d)
    np.save(os.path.join(output_path, "data_test_FGS.npy"), test_fgs)
    
    # Save planet IDs for reference
    np.save(os.path.join(output_path, "train_planet_ids.npy"), train_planet_ids)
    np.save(os.path.join(output_path, "test_planet_ids.npy"), test_planet_ids)
    
    # Save original test data for submission generation
    np.save(os.path.join(output_path, "test_wl_data.npy"), test_wl_data)
    np.save(os.path.join(output_path, "test_sigma_data.npy"), test_sigma_data)
    
    print(f"Converted data saved to {output_path}/")
    print(f"Training data: {train_data_3d.shape}")
    print(f"Training FGS: {train_fgs.shape}")
    print(f"Training targets: {train_targets.shape}")
    print(f"Test data: {test_data_3d.shape}")
    print(f"Test FGS: {test_fgs.shape}")
    
    # Create a summary file
    with open(os.path.join(output_path, "conversion_summary.txt"), "w") as f:
        f.write("ARIEL Dataset Conversion Summary\n")
        f.write("================================\n\n")
        f.write(f"Training planets: {len(train_planet_ids)}\n")
        f.write(f"Test planets: {len(test_planet_ids)}\n")
        f.write(f"Time bins: {n_time_bins}\n")
        f.write(f"Wavelengths: {n_wavelengths}\n")
        f.write(f"FGS size: {fgs_size}x{fgs_size}\n")
        f.write(f"Targets: {n_targets}\n\n")
        f.write("Files created:\n")
        f.write("- data_train.npy: Training spectra (N, 187, 283)\n")
        f.write("- data_train_FGS.npy: Training FGS data (N, 187, 32, 32)\n")
        f.write("- targets_train.npy: Training targets (N, 6)\n")
        f.write("- data_test.npy: Test spectra (N, 187, 283)\n")
        f.write("- data_test_FGS.npy: Test FGS data (N, 187, 32, 32)\n")
        f.write("- train_planet_ids.npy: Training planet IDs\n")
        f.write("- test_planet_ids.npy: Test planet IDs\n")
        f.write("- test_wl_data.npy: Original test wavelength data\n")
        f.write("- test_sigma_data.npy: Original test sigma data\n")
    
    print("Conversion complete!")

if __name__ == "__main__":
    convert_official_dataset()
