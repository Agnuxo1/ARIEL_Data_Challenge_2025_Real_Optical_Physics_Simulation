#!/usr/bin/env python3
"""
ARIEL Data Challenge 2025 - Official Calibration Pipeline
Adapted from the official notebook for the hybrid quantum-NEBULA model
"""

import numpy as np
import pandas as pd
import itertools
import os
import glob
from astropy.stats import sigma_clip
from tqdm import tqdm
import sys

def ADC_convert(signal, gain=0.4369, offset=-1000):
    """Analog-to-Digital Conversion reversal"""
    signal = signal.astype(np.float64)
    signal /= gain
    signal += offset
    return signal

def mask_hot_dead(signal, dead, dark):
    """Mask hot and dead pixels"""
    hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))
    signal = np.ma.masked_where(dead, signal)
    signal = np.ma.masked_where(hot, signal)
    return signal

def apply_linear_corr(linear_corr, clean_signal):
    """Apply linearity correction"""
    linear_corr = np.flip(linear_corr, axis=0)
    for x, y in itertools.product(
                range(clean_signal.shape[1]), range(clean_signal.shape[2])
            ):
        poli = np.poly1d(linear_corr[:, x, y])
        clean_signal[:, x, y] = poli(clean_signal[:, x, y])
    return clean_signal

def clean_dark(signal, dead, dark, dt):
    """Dark current subtraction"""
    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))
    signal -= dark * dt[:, np.newaxis, np.newaxis]
    return signal

def get_cds(signal):
    """Get Correlated Double Sampling"""
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

def get_index(files, CHUNKS_SIZE):
    """Get indices for processing chunks"""
    index = []
    for file in files:
        file_name = os.path.basename(file)
        if (file_name.split('_')[0] == 'AIRS-CH0' and
            file_name.split('_')[1] == 'signal' and
            file_name.split('_')[2] == '0.parquet'):
            file_index = os.path.basename(os.path.dirname(file))
            index.append(int(file_index))

    if len(index) == 0:
        return []

    index = np.array(index)
    index = np.sort(index)
    # Each chunk contains CHUNKS_SIZE planets
    num_chunks = max(1, len(index) // CHUNKS_SIZE)
    index = np.array_split(index, num_chunks)
    return index

def process_ariel_data(path_folder, output_dir, max_planets=100):
    """
    Process ARIEL data using the official calibration pipeline
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory {output_dir} created.")

    # Configuration
    CHUNKS_SIZE = 1
    DO_MASK = True
    DO_THE_NL_CORR = False
    DO_DARK = True
    DO_FLAT = True
    TIME_BINNING = True

    cut_inf, cut_sup = 39, 321
    l = cut_sup - cut_inf  # 282 wavelengths

    # Get file indices
    train_path = os.path.join(path_folder, 'train', '*', '*')
    files = glob.glob(train_path)
    files = files[:max_planets*4]  # Limit number of files for faster processing

    index = get_index(files, CHUNKS_SIZE)

    # Load axis info
    axis_info = pd.read_parquet(os.path.join(path_folder, 'axis_info.parquet'))

    print(f"Processing {len(index)} planets...")

    all_spectra = []
    all_fgs = []
    planet_ids = []

    for n, index_chunk in enumerate(tqdm(index)):
        if n >= max_planets:  # Limit for testing
            break

        try:
            planet_id = index_chunk[0]
            planet_ids.append(planet_id)

            # Process AIRS data
            df = pd.read_parquet(os.path.join(path_folder, f'train/{planet_id}/AIRS-CH0_signal_0.parquet'))
            signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 356))

            signal = ADC_convert(signal)
            dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
            dt_airs[1::2] += 0.1
            chopped_signal = signal[:, :, cut_inf:cut_sup]
            del signal, df

            # Load calibration files
            flat = pd.read_parquet(os.path.join(path_folder, f'train/{planet_id}/AIRS-CH0_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
            dark = pd.read_parquet(os.path.join(path_folder, f'train/{planet_id}/AIRS-CH0_calibration_0/dark.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
            dead_airs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_id}/AIRS-CH0_calibration_0/dead.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]

            # Apply calibrations
            if DO_MASK:
                chopped_signal = mask_hot_dead(chopped_signal, dead_airs, dark)

            if DO_DARK:
                chopped_signal = clean_dark(chopped_signal, dead_airs, dark, dt_airs)

            # Process FGS data
            df_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_id}/FGS1_signal_0.parquet'))
            fgs_signal = df_fgs.values.astype(np.float64).reshape((df_fgs.shape[0], 32, 32))

            fgs_signal = ADC_convert(fgs_signal)
            dt_fgs1 = np.ones(len(fgs_signal)) * 0.1
            dt_fgs1[1::2] += 0.1

            # Load FGS calibration
            flat_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_id}/FGS1_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 32))
            dark_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_id}/FGS1_calibration_0/dark.parquet')).values.astype(np.float64).reshape((32, 32))
            dead_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_id}/FGS1_calibration_0/dead.parquet')).values.astype(np.float64).reshape((32, 32))

            # Apply FGS calibrations
            if DO_MASK:
                fgs_signal = mask_hot_dead(fgs_signal, dead_fgs, dark_fgs)

            if DO_DARK:
                fgs_signal = clean_dark(fgs_signal, dead_fgs, dark_fgs, dt_fgs1)

            # Get CDS
            airs_cds = get_cds(chopped_signal.reshape(1, *chopped_signal.shape))
            fgs_cds = get_cds(fgs_signal.reshape(1, *fgs_signal.shape))

            # Time binning
            if TIME_BINNING:
                airs_binned = bin_obs(airs_cds, binning=30)
                fgs_binned = bin_obs(fgs_cds, binning=30*12)
            else:
                airs_binned = airs_cds.transpose(0,1,3,2)
                fgs_binned = fgs_cds.transpose(0,1,3,2)

            # Flat field correction
            if DO_FLAT:
                airs_binned[0] = correct_flat_field(flat, dead_airs, airs_binned[0])
                fgs_binned[0] = correct_flat_field(flat_fgs, dead_fgs, fgs_binned[0])

            # Extract features for the model
            # Average over time and spatial dimensions to get spectrum
            airs_spectrum = np.mean(airs_binned[0], axis=(0, 2))  # Shape: (282,)
            fgs_mean = np.mean(fgs_binned[0])  # Single value

            all_spectra.append(airs_spectrum)
            all_fgs.append(fgs_mean)

        except Exception as e:
            print(f"Error processing planet {index_chunk[0]}: {e}")
            continue

    # Convert to numpy arrays
    spectra_array = np.array(all_spectra, dtype=np.float32)  # Shape: (N, 282)
    fgs_array = np.array(all_fgs, dtype=np.float32).reshape(-1, 1)  # Shape: (N, 1)

    # Combine AIRS spectrum + FGS
    combined_data = np.concatenate([spectra_array, fgs_array], axis=1)  # Shape: (N, 283)

    print(f"Processed {len(planet_ids)} planets")
    print(f"Final data shape: {combined_data.shape}")

    # Load targets
    train_csv = pd.read_csv(os.path.join(path_folder, 'train.csv'))
    targets = []

    for planet_id in planet_ids:
        planet_targets = train_csv[train_csv['planet_id'] == planet_id]
        if len(planet_targets) > 0:
            # Extract first 6 atmospheric parameters
            target_values = planet_targets.iloc[0, 1:7].values
            targets.append(target_values)
        else:
            targets.append(np.zeros(6))

    targets_array = np.array(targets, dtype=np.float32)

    # Save processed data
    np.save(os.path.join(output_dir, 'data_train.npy'), combined_data)
    np.save(os.path.join(output_dir, 'targets_train.npy'), targets_array)

    print(f"Saved calibrated data to: {output_dir}")
    print(f"Spectra shape: {combined_data.shape}")
    print(f"Targets shape: {targets_array.shape}")

    return combined_data, targets_array

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ariel_calibration.py <data_path> [output_dir] [max_planets]")
        sys.exit(1)

    data_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else data_path
    max_planets = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    process_ariel_data(data_path, output_dir, max_planets)