#!/usr/bin/env python3
"""
ARIEL Data Challenge 2025 - CPU-Optimized Calibration Pipeline
Fast CPU processing for all planets without GPU dependencies
"""

import numpy as np
import pandas as pd
import itertools
import os
import glob
from astropy.stats import sigma_clip
from tqdm import tqdm
import sys
import time
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

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
    """Apply linearity correction - vectorized for speed"""
    linear_corr = np.flip(linear_corr, axis=0)

    # Vectorized approach for speed
    original_shape = clean_signal.shape
    signal_2d = clean_signal.reshape(original_shape[0], -1)
    linear_2d = linear_corr.reshape(linear_corr.shape[0], -1)

    for i in range(signal_2d.shape[1]):
        coeffs = linear_2d[:, i]
        signal_2d[:, i] = np.polyval(coeffs, signal_2d[:, i])

    return signal_2d.reshape(original_shape)

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
    """Time binning - optimized"""
    cds_transposed = cds_signal.transpose(0,1,3,2)
    n_bins = cds_transposed.shape[1] // binning
    cds_binned = np.zeros((cds_transposed.shape[0], n_bins, cds_transposed.shape[2], cds_transposed.shape[3]))

    for i in range(n_bins):
        start_idx = i * binning
        end_idx = (i + 1) * binning
        cds_binned[:,i,:,:] = np.sum(cds_transposed[:,start_idx:end_idx,:,:], axis=1)

    return cds_binned

def correct_flat_field(flat, dead, signal):
    """Flat field correction"""
    flat = flat.transpose(1, 0)
    dead = dead.transpose(1, 0)
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    signal = signal / flat
    return signal

def process_single_planet(args):
    """Process a single planet - for parallel processing"""
    planet_idx, path_folder, cut_inf, cut_sup = args

    try:
        # Configuration
        DO_MASK = True
        DO_THE_NL_CORR = True
        DO_DARK = True
        DO_FLAT = True
        TIME_BINNING = True

        # Process AIRS data
        df = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/AIRS-CH0_signal_0.parquet'))
        signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 356))

        signal = ADC_convert(signal)
        # Load dt info for AIRS
        axis_info = pd.read_parquet(os.path.join(path_folder, 'axis_info.parquet'))
        dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
        dt_airs[1::2] += 0.1

        chopped_signal = signal[:, :, cut_inf:cut_sup]
        del signal, df

        # Load calibration files
        flat = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/AIRS-CH0_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        dark = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/AIRS-CH0_calibration_0/dark.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        dead_airs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/AIRS-CH0_calibration_0/dead.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        linear_corr = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/AIRS-CH0_calibration_0/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 356))[:, :, cut_inf:cut_sup]

        # Apply calibrations
        if DO_MASK:
            chopped_signal = mask_hot_dead(chopped_signal, dead_airs, dark)

        if DO_THE_NL_CORR:
            chopped_signal = apply_linear_corr(linear_corr, chopped_signal)

        if DO_DARK:
            chopped_signal = clean_dark(chopped_signal, dead_airs, dark, dt_airs)

        # Process FGS data
        df_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/FGS1_signal_0.parquet'))
        fgs_signal = df_fgs.values.astype(np.float64).reshape((df_fgs.shape[0], 32, 32))

        fgs_signal = ADC_convert(fgs_signal)
        dt_fgs1 = np.ones(len(fgs_signal)) * 0.1
        dt_fgs1[1::2] += 0.1

        # Load FGS calibration
        flat_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/FGS1_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 32))
        dark_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/FGS1_calibration_0/dark.parquet')).values.astype(np.float64).reshape((32, 32))
        dead_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/FGS1_calibration_0/dead.parquet')).values.astype(np.float64).reshape((32, 32))
        linear_corr_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/FGS1_calibration_0/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 32))

        # Apply FGS calibrations
        if DO_MASK:
            fgs_signal = mask_hot_dead(fgs_signal, dead_fgs, dark_fgs)

        if DO_THE_NL_CORR:
            fgs_signal = apply_linear_corr(linear_corr_fgs, fgs_signal)

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
        airs_spectrum = np.nanmean(airs_binned[0], axis=(0, 2))  # Shape: (282,)
        fgs_mean = np.nanmean(fgs_binned[0])  # Single value

        return planet_idx, airs_spectrum, fgs_mean

    except Exception as e:
        print(f"Error processing planet {planet_idx}: {e}")
        return planet_idx, None, None

def get_all_planet_indices(path_folder):
    """Get all available planet indices"""
    train_path = os.path.join(path_folder, 'train', '*', '*')
    files = glob.glob(train_path)

    planet_indices = set()
    for file in files:
        file_name = os.path.basename(file)
        if (file_name.split('_')[0] == 'AIRS-CH0' and
            file_name.split('_')[1] == 'signal' and
            file_name.split('_')[2] == '0.parquet'):
            file_index = os.path.basename(os.path.dirname(file))
            planet_indices.add(int(file_index))

    return sorted(list(planet_indices))

def process_ariel_data_cpu_fast(path_folder, output_dir, max_planets=None):
    """
    Fast CPU-based ARIEL data processing using multiprocessing
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory {output_dir} created.")

    cut_inf, cut_sup = 39, 321

    # Get all planet indices
    all_planet_indices = get_all_planet_indices(path_folder)

    if max_planets:
        all_planet_indices = all_planet_indices[:max_planets]

    print(f"CPU-Optimized Calibration Pipeline")
    print(f"Total planets to process: {len(all_planet_indices)}")
    print(f"Using {cpu_count()} CPU cores")

    # Prepare arguments for parallel processing
    args_list = [(planet_idx, path_folder, cut_inf, cut_sup) for planet_idx in all_planet_indices]

    start_time = time.time()

    # Process planets in parallel
    all_spectra = []
    all_fgs = []
    planet_ids = []

    # Use only 2 processes to avoid overwhelming the system
    n_processes = 2

    print(f"Starting parallel processing with {n_processes} processes...")

    with Pool(n_processes) as pool:
        results = list(tqdm(pool.imap(process_single_planet, args_list),
                           total=len(args_list),
                           desc="Processing planets"))

    # Collect results
    for planet_id, spectrum, fgs_val in results:
        if spectrum is not None and fgs_val is not None:
            planet_ids.append(planet_id)
            all_spectra.append(spectrum)
            all_fgs.append(fgs_val)

    print(f"Successfully processed {len(planet_ids)} out of {len(all_planet_indices)} planets")

    if len(all_spectra) == 0:
        print("No valid spectra processed!")
        return None, None

    # Convert to numpy arrays
    spectra_array = np.array(all_spectra, dtype=np.float32)
    fgs_array = np.array(all_fgs, dtype=np.float32).reshape(-1, 1)

    print(f"Spectra array shape: {spectra_array.shape}")
    print(f"FGS array shape: {fgs_array.shape}")

    # Combine AIRS spectrum + FGS
    combined_data = np.concatenate([spectra_array, fgs_array], axis=1)

    print(f"Final data shape: {combined_data.shape}")

    # Load targets
    train_csv = pd.read_csv(os.path.join(path_folder, 'train.csv'))
    targets = []

    for planet_id in planet_ids:
        planet_targets = train_csv[train_csv['planet_id'] == planet_id]
        if len(planet_targets) > 0:
            target_values = planet_targets.iloc[0, 1:7].values
            targets.append(target_values)
        else:
            targets.append(np.zeros(6))

    targets_array = np.array(targets, dtype=np.float32)

    # Save processed data
    np.save(os.path.join(output_dir, 'data_train.npy'), combined_data)
    np.save(os.path.join(output_dir, 'targets_train.npy'), targets_array)

    # Save planet IDs for reference
    np.save(os.path.join(output_dir, 'planet_ids.npy'), np.array(planet_ids))

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time/60:.1f} minutes")
    print(f"Average time per planet: {total_time/len(planet_ids):.2f} seconds")
    print(f"Saved calibrated data to: {output_dir}")
    print(f"Spectra shape: {combined_data.shape}")
    print(f"Targets shape: {targets_array.shape}")

    return combined_data, targets_array

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ariel_calibration_cpu_fast.py <data_path> [output_dir] [max_planets]")
        sys.exit(1)

    data_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./calibrated_data"
    max_planets = int(sys.argv[3]) if len(sys.argv) > 3 else None

    print("="*60)
    print("ARIEL Data Challenge 2025 - CPU Fast Calibration Pipeline")
    print("="*60)

    process_ariel_data_cpu_fast(data_path, output_dir, max_planets)