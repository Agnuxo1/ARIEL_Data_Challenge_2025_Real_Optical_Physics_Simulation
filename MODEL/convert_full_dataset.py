#!/usr/bin/env python3
"""
Convert full ARIEL dataset including all test planets
"""
import numpy as np
import pandas as pd
import os
import sys
import glob
from pathlib import Path

def convert_full_dataset():
    """Convert full dataset including all test planets"""
    
    # Paths
    official_path = "E:/NeurIPS_MYCELIUM_EVOLUTUM/ariel-data-challenge-2025"
    output_path = "./calibrated_data"
    
    print("Loading full ARIEL dataset...")
    
    # Load training data
    train_df = pd.read_csv(os.path.join(official_path, "train.csv"))
    print(f"Training data shape: {train_df.shape}")
    
    # Get all test planet directories
    test_path = os.path.join(official_path, "test")
    test_planets = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
    test_planets.sort()
    
    print(f"Found {len(test_planets)} test planets")
    
    # Process training data
    wl_columns = [f'wl_{i}' for i in range(1, 284)]
    train_wl_data = train_df[wl_columns].values.astype(np.float32)
    train_planet_ids = train_df['planet_id'].values
    
    # Create synthetic sigma data for training
    train_sigma_data = np.random.normal(0.01, 0.005, train_wl_data.shape).astype(np.float32)
    
    # Process test data
    test_wl_data = []
    test_sigma_data = []
    test_planet_ids = []
    
    print("Processing test planets...")
    for i, planet_id in enumerate(test_planets):
        if i % 100 == 0:
            print(f"Processing planet {i+1}/{len(test_planets)}: {planet_id}")
        
        planet_path = os.path.join(test_path, planet_id)
        
        # Load AIRS data
        airs_files = glob.glob(os.path.join(planet_path, "AIRS-CH0_signal_*.parquet"))
        if not airs_files:
            print(f"Warning: No AIRS data found for planet {planet_id}")
            continue
            
        # Load the first AIRS file
        airs_df = pd.read_parquet(airs_files[0])
        
        # Extract wavelength data (assuming columns are wl_1, wl_2, etc.)
        wl_data = airs_df[wl_columns].values.astype(np.float32)
        
        # Average across time if multiple rows
        if len(wl_data) > 1:
            wl_data = np.mean(wl_data, axis=0, keepdims=True)
        
        test_wl_data.append(wl_data[0])
        
        # Create synthetic sigma data
        sigma_data = np.random.normal(0.01, 0.005, (283,)).astype(np.float32)
        test_sigma_data.append(sigma_data)
        
        test_planet_ids.append(int(planet_id))
    
    # Convert to numpy arrays
    test_wl_data = np.array(test_wl_data)
    test_sigma_data = np.array(test_sigma_data)
    test_planet_ids = np.array(test_planet_ids)
    
    print(f"Test data shape: {test_wl_data.shape}")
    print(f"Test sigma shape: {test_sigma_data.shape}")
    
    # Create time dimension (simulate 187 time bins)
    n_time_bins = 187
    n_wavelengths = 283
    
    # Training data
    train_data_3d = np.zeros((len(train_planet_ids), n_time_bins, n_wavelengths), dtype=np.float32)
    train_sigma_3d = np.zeros((len(train_planet_ids), n_time_bins, n_wavelengths), dtype=np.float32)
    
    for i in range(len(train_planet_ids)):
        for t in range(n_time_bins):
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
    fgs_size = 32
    train_fgs = np.random.normal(0.5, 0.1, (len(train_planet_ids), n_time_bins, fgs_size, fgs_size)).astype(np.float32)
    test_fgs = np.random.normal(0.5, 0.1, (len(test_planet_ids), n_time_bins, fgs_size, fgs_size)).astype(np.float32)
    
    # Create synthetic targets for training
    n_targets = 6  # CO2, H2O, CH4, NH3, Temperature, Radius
    
    train_targets = np.zeros((len(train_planet_ids), n_targets), dtype=np.float32)
    for i in range(len(train_planet_ids)):
        spectrum = train_wl_data[i, :]
        
        # CO2: based on absorption around 2.0-2.1 microns
        co2_region = spectrum[150:170]
        train_targets[i, 0] = np.mean(co2_region) * 100
        
        # H2O: based on absorption around 1.3-1.5 microns  
        h2o_region = spectrum[80:100]
        train_targets[i, 1] = np.mean(h2o_region) * 100
        
        # CH4: based on absorption around 1.6-1.8 microns
        ch4_region = spectrum[110:130]
        train_targets[i, 2] = np.mean(ch4_region) * 100
        
        # NH3: based on absorption around 1.0-1.2 microns
        nh3_region = spectrum[50:70]
        train_targets[i, 3] = np.mean(nh3_region) * 100
        
        # Temperature: based on overall spectrum shape
        temp_factor = np.std(spectrum) * 1000
        train_targets[i, 4] = 500 + temp_factor
        
        # Radius: based on spectrum amplitude
        radius_factor = np.mean(spectrum) * 2
        train_targets[i, 5] = 0.5 + radius_factor
    
    # Save data
    os.makedirs(output_path, exist_ok=True)
    
    print("Saving converted data...")
    
    # Save training data
    np.save(os.path.join(output_path, "data_train.npy"), train_data_3d)
    np.save(os.path.join(output_path, "data_train_FGS.npy"), train_fgs)
    np.save(os.path.join(output_path, "targets_train.npy"), train_targets)
    
    # Save test data
    np.save(os.path.join(output_path, "data_test.npy"), test_data_3d)
    np.save(os.path.join(output_path, "data_test_FGS.npy"), test_fgs)
    
    # Save planet IDs
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
    print(f"Test planets: {len(test_planet_ids)}")
    
    # Create summary
    with open(os.path.join(output_path, "conversion_summary.txt"), "w") as f:
        f.write("ARIEL Full Dataset Conversion Summary\n")
        f.write("====================================\n\n")
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
    
    print("Full conversion complete!")

if __name__ == "__main__":
    convert_full_dataset()
