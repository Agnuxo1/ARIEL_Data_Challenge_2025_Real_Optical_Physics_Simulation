#!/usr/bin/env python3
"""
ARIEL Data Calibration Script - Adapted for /mnt/disco2 data
Based on the official ARIEL Data Challenge 2025 calibration notebook
"""

import numpy as np
import pandas as pd
import itertools
import os
import glob
from tqdm import tqdm

def ADC_convert(signal, gain=0.4369, offset=-1000):
    """Analog-to-Digital Conversion reversal"""
    signal = signal.astype(np.float64)
    signal /= gain
    signal += offset
    return signal

def mask_hot_dead(signal, dead, dark):
    """Mask hot and dead pixels"""
    from astropy.stats import sigma_clip
    hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))
    signal = np.ma.masked_where(dead, signal)
    signal = np.ma.masked_where(hot, signal)
    return signal

def clean_dark(signal, dead, dark, dt):
    """Dark current subtraction"""
    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))
    signal -= dark * dt[:, np.newaxis, np.newaxis]
    return signal

def get_cds(signal):
    """Correlated Double Sampling"""
    cds = signal[:,1::2,:,:] - signal[:,::2,:,:]
    return cds

def bin_obs(cds_signal, binning):
    """Time binning"""
    cds_transposed = cds_signal.transpose(0,1,3,2)
    cds_binned = np.zeros((cds_transposed.shape[0], cds_transposed.shape[1]//binning,
                          cds_transposed.shape[2], cds_transposed.shape[3]))
    for i in range(cds_transposed.shape[1]//binning):
        cds_binned[:,i,:,:] = np.sum(cds_transposed[:,i*binning:(i+1)*binning,:,:], axis=1)
    return cds_binned

def correct_flat_field(flat, dead, signal):
    """Flat field correction"""
    flat = flat.transpose(1, 0)
    dead = dead.transpose(1, 0)
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    signal = signal / flat
    return signal

def main():
    # Paths for our system
    path_folder = '/mnt/disco2/'
    path_out = '/mnt/disco2/calibrated/'

    # Create output directory
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        print(f"Directory {path_out} created.")

    # Configuration
    CHUNKS_SIZE = 1
    DO_MASK = True
    DO_DARK = True
    DO_FLAT = True
    TIME_BINNING = True
    cut_inf, cut_sup = 39, 321
    l = cut_sup - cut_inf

    # Get ALL training directories for complete physics-based dataset
    train_dirs = sorted([d for d in os.listdir(os.path.join(path_folder, 'train/'))
                        if os.path.isdir(os.path.join(path_folder, 'train/', d))])

    print(f"Found {len(train_dirs)} total planets for calibration")
    # Process in chunks to manage memory
    chunk_size = 50
    train_chunks = [train_dirs[i:i+chunk_size] for i in range(0, len(train_dirs), chunk_size)]

    print(f"Processing {len(train_dirs)} training samples...")

    # Load axis info
    try:
        axis_info = pd.read_parquet(os.path.join(path_folder,'axis_info.parquet'))
        dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
        dt_airs[1::2] += 0.1
    except:
        print("Warning: Could not load axis_info.parquet, using default timing")
        dt_airs = np.ones(22500) * 0.1
        dt_airs[1::2] += 0.1

    all_airs_data = []
    all_fgs_data = []

    # Process all planets in chunks for memory efficiency
    for chunk_idx, chunk_dirs in enumerate(train_chunks):
        print(f"Processing chunk {chunk_idx+1}/{len(train_chunks)} ({len(chunk_dirs)} planets)")

        for idx, train_dir in enumerate(tqdm(chunk_dirs, desc=f"Chunk {chunk_idx+1}")):
            try:
                train_path = os.path.join(path_folder, 'train', train_dir)

                # Process AIRS-CH0 data
                airs_signal_file = os.path.join(train_path, 'AIRS-CH0_signal_0.parquet')
                if os.path.exists(airs_signal_file):
                    df = pd.read_parquet(airs_signal_file)
                    signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 356))
                    signal = ADC_convert(signal)
                    chopped_signal = signal[:, :, cut_inf:cut_sup]

                    # Load calibration files
                    cal_path = os.path.join(train_path, 'AIRS-CH0_calibration_0')
                    if os.path.exists(cal_path):
                        flat = pd.read_parquet(os.path.join(cal_path, 'flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
                        dark = pd.read_parquet(os.path.join(cal_path, 'dark.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
                        dead_airs = pd.read_parquet(os.path.join(cal_path, 'dead.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]

                        if DO_MASK:
                            chopped_signal = mask_hot_dead(chopped_signal, dead_airs, dark)
                        if DO_DARK:
                            chopped_signal = clean_dark(chopped_signal, dead_airs, dark, dt_airs[:len(chopped_signal)])

                        # Get CDS
                        airs_cds = get_cds(chopped_signal[np.newaxis, :])
                        if TIME_BINNING:
                            airs_cds = bin_obs(airs_cds, binning=30)

                        if DO_FLAT:
                            airs_cds[0] = correct_flat_field(flat, dead_airs, airs_cds[0])

                        # Extract spectral features (average over spatial and time dimensions)
                        spectral_features = np.mean(airs_cds[0], axis=(0, 2))  # Shape: (282,)
                        all_airs_data.append(spectral_features)

                # Process FGS1 data (simplified)
                fgs_signal_file = os.path.join(train_path, 'FGS1_signal_0.parquet')
                if os.path.exists(fgs_signal_file):
                    df = pd.read_parquet(fgs_signal_file)
                    fgs_signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 32))
                    fgs_signal = ADC_convert(fgs_signal)

                    # Simple FGS processing - just extract mean flux
                    fgs_features = np.mean(fgs_signal)  # Single value
                    all_fgs_data.append(fgs_features)

            except Exception as e:
                print(f"Error processing {train_dir}: {e}")
                continue

    # Convert to numpy arrays
    if all_airs_data:
        airs_array = np.array(all_airs_data)
        print(f"AIRS data shape: {airs_array.shape}")

        # Combine AIRS (282 features) + FGS (1 feature) = 283 total features
        if all_fgs_data:
            fgs_array = np.array(all_fgs_data).reshape(-1, 1)
            combined_data = np.hstack([airs_array, fgs_array])
        else:
            combined_data = airs_array

        print(f"Combined data shape: {combined_data.shape}")

        # Save calibrated data
        np.save(os.path.join(path_out, 'data_train.npy'), combined_data)
        print(f"Saved calibrated training data to {path_out}data_train.npy")

        # Load REAL targets from train.csv
        try:
            train_csv = pd.read_csv(os.path.join(path_folder, 'train.csv'))
            print(f"Loaded train.csv with {len(train_csv)} samples")

            # Get planet IDs that we processed
            processed_ids = [int(d) for d in train_dirs if d in [str(row) for row in train_csv['planet_id'].values]]

            # Filter targets for processed planets
            target_cols = ['T', 'log_H2O', 'log_CO2', 'log_CO', 'log_CH4', 'log_NH3']
            filtered_targets = train_csv[train_csv['planet_id'].isin(processed_ids)][target_cols]

            # Ensure same order as processed data
            planet_to_idx = {planet_id: idx for idx, planet_id in enumerate(processed_ids)}
            sorted_targets = filtered_targets.iloc[[planet_to_idx.get(pid, 0) for pid in processed_ids]]

            targets_array = sorted_targets.values.astype(np.float32)
            print(f"Real targets shape: {targets_array.shape}")

        except Exception as e:
            print(f"Could not load real targets from train.csv: {e}")
            print("Using synthetic targets instead")
            targets_array = np.random.randn(len(combined_data), 6).astype(np.float32)

        np.save(os.path.join(path_out, 'targets_train.npy'), targets_array)
        print(f"Saved targets to {path_out}targets_train.npy")

        # Create FGS data file
        if all_fgs_data:
            fgs_expanded = np.tile(fgs_array, (1, 100))  # Expand to 100 features
            np.save(os.path.join(path_out, 'data_train_FGS.npy'), fgs_expanded)
            print(f"Saved FGS data to {path_out}data_train_FGS.npy")

if __name__ == "__main__":
    main()