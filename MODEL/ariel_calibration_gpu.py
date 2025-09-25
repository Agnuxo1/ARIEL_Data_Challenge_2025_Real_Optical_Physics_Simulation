#!/usr/bin/env python3
"""
ARIEL Data Challenge 2025 - GPU-Accelerated Calibration Pipeline
Optimized for CUDA acceleration to process all planets efficiently
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

# GPU acceleration imports
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage
    GPU_AVAILABLE = True
    print("CUDA GPU detected and enabled")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("Warning: CuPy not available, falling back to CPU")

def ADC_convert_gpu(signal, gain=0.4369, offset=-1000):
    """GPU-accelerated Analog-to-Digital Conversion reversal"""
    if GPU_AVAILABLE:
        signal = cp.asarray(signal, dtype=cp.float64)
        signal /= gain
        signal += offset
        return signal
    else:
        signal = signal.astype(np.float64)
        signal /= gain
        signal += offset
        return signal

def mask_hot_dead_gpu(signal, dead, dark):
    """GPU-accelerated hot and dead pixel masking"""
    if GPU_AVAILABLE:
        signal = cp.asarray(signal)
        dead = cp.asarray(dead)
        dark = cp.asarray(dark)

        # Compute hot pixels using GPU
        dark_mean = cp.mean(dark)
        dark_std = cp.std(dark)
        hot = cp.abs(dark - dark_mean) > 5 * dark_std

        hot = cp.tile(hot, (signal.shape[0], 1, 1))
        dead = cp.tile(dead, (signal.shape[0], 1, 1))

        signal = cp.where(dead, cp.nan, signal)
        signal = cp.where(hot, cp.nan, signal)
        return signal
    else:
        hot = sigma_clip(dark, sigma=5, maxiters=5).mask
        hot = np.tile(hot, (signal.shape[0], 1, 1))
        dead = np.tile(dead, (signal.shape[0], 1, 1))
        signal = np.ma.masked_where(dead, signal)
        signal = np.ma.masked_where(hot, signal)
        return signal

def apply_linear_corr_gpu(linear_corr, clean_signal):
    """GPU-accelerated linearity correction"""
    if GPU_AVAILABLE:
        linear_corr = cp.asarray(linear_corr)
        clean_signal = cp.asarray(clean_signal)
        linear_corr = cp.flip(linear_corr, axis=0)

        # Vectorized polynomial correction on GPU
        for x in range(clean_signal.shape[1]):
            for y in range(clean_signal.shape[2]):
                coeffs = linear_corr[:, x, y]
                clean_signal[:, x, y] = cp.polyval(coeffs, clean_signal[:, x, y])
        return clean_signal
    else:
        linear_corr = np.flip(linear_corr, axis=0)
        for x, y in itertools.product(
                    range(clean_signal.shape[1]), range(clean_signal.shape[2])
                ):
            poli = np.poly1d(linear_corr[:, x, y])
            clean_signal[:, x, y] = poli(clean_signal[:, x, y])
        return clean_signal

def clean_dark_gpu(signal, dead, dark, dt):
    """GPU-accelerated dark current subtraction"""
    if GPU_AVAILABLE:
        signal = cp.asarray(signal)
        dead = cp.asarray(dead)
        dark = cp.asarray(dark)
        dt = cp.asarray(dt)

        dark = cp.where(dead, cp.nan, dark)
        dark = cp.tile(dark, (signal.shape[0], 1, 1))
        signal -= dark * dt[:, cp.newaxis, cp.newaxis]
        return signal
    else:
        dark = np.ma.masked_where(dead, dark)
        dark = np.tile(dark, (signal.shape[0], 1, 1))
        signal -= dark * dt[:, np.newaxis, np.newaxis]
        return signal

def get_cds_gpu(signal):
    """GPU-accelerated Correlated Double Sampling"""
    if GPU_AVAILABLE:
        signal = cp.asarray(signal)
        cds = signal[:,1::2,:,:] - signal[:,::2,:,:]
        return cds
    else:
        cds = signal[:,1::2,:,:] - signal[:,::2,:,:]
        return cds

def bin_obs_gpu(cds_signal, binning):
    """GPU-accelerated time binning"""
    if GPU_AVAILABLE:
        cds_signal = cp.asarray(cds_signal)
        cds_transposed = cds_signal.transpose(0,1,3,2)
        cds_binned = cp.zeros((cds_transposed.shape[0], cds_transposed.shape[1]//binning,
                              cds_transposed.shape[2], cds_transposed.shape[3]))

        for i in range(cds_transposed.shape[1]//binning):
            cds_binned[:,i,:,:] = cp.sum(cds_transposed[:,i*binning:(i+1)*binning,:,:], axis=1)
        return cds_binned
    else:
        cds_transposed = cds_signal.transpose(0,1,3,2)
        cds_binned = np.zeros((cds_transposed.shape[0], cds_transposed.shape[1]//binning,
                              cds_transposed.shape[2], cds_transposed.shape[3]))
        for i in range(cds_transposed.shape[1]//binning):
            cds_binned[:,i,:,:] = np.sum(cds_transposed[:,i*binning:(i+1)*binning,:,:], axis=1)
        return cds_binned

def correct_flat_field_gpu(flat, dead, signal):
    """GPU-accelerated flat field correction"""
    if GPU_AVAILABLE:
        flat = cp.asarray(flat).transpose(1, 0)
        dead = cp.asarray(dead).transpose(1, 0)
        signal = cp.asarray(signal)

        flat = cp.where(dead, cp.nan, flat)
        flat = cp.tile(flat, (signal.shape[0], 1, 1))
        signal = signal / flat
        return signal
    else:
        flat = flat.transpose(1, 0)
        dead = dead.transpose(1, 0)
        flat = np.ma.masked_where(dead, flat)
        flat = np.tile(flat, (signal.shape[0], 1, 1))
        signal = signal / flat
        return signal

def to_numpy(array):
    """Convert GPU array back to numpy if needed"""
    if GPU_AVAILABLE and hasattr(array, 'get'):
        return array.get()
    return array

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
    # Process in chunks but don't limit the dataset
    num_chunks = max(1, len(index) // CHUNKS_SIZE)
    index = np.array_split(index, num_chunks)
    return index

def process_ariel_data_gpu(path_folder, output_dir, max_planets=None):
    """
    GPU-accelerated ARIEL data processing using the official calibration pipeline
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory {output_dir} created.")

    # Configuration for full processing
    CHUNKS_SIZE = 4 if GPU_AVAILABLE else 1  # Larger chunks with GPU
    DO_MASK = True
    DO_THE_NL_CORR = True  # Enable non-linearity correction for better accuracy
    DO_DARK = True
    DO_FLAT = True
    TIME_BINNING = True

    cut_inf, cut_sup = 39, 321
    l = cut_sup - cut_inf  # 282 wavelengths

    # Get ALL files - no limiting
    train_path = os.path.join(path_folder, 'train', '*', '*')
    files = glob.glob(train_path)

    if max_planets:
        files = files[:max_planets*4]

    index = get_index(files, CHUNKS_SIZE)

    # Load axis info
    axis_info = pd.read_parquet(os.path.join(path_folder, 'axis_info.parquet'))

    print(f"GPU Acceleration: {'ENABLED' if GPU_AVAILABLE else 'DISABLED'}")
    print(f"Processing {len(index)} planet chunks with chunk size {CHUNKS_SIZE}")
    print(f"Total planets to process: {sum(len(chunk) for chunk in index)}")

    all_spectra = []
    all_fgs = []
    planet_ids = []

    start_time = time.time()

    for n, index_chunk in enumerate(tqdm(index, desc="Processing planet chunks")):
        try:
            chunk_spectra = []
            chunk_fgs = []

            for planet_idx in index_chunk:
                planet_ids.append(planet_idx)

                # Process AIRS data
                df = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/AIRS-CH0_signal_0.parquet'))
                signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 356))

                signal = ADC_convert_gpu(signal)
                dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
                dt_airs[1::2] += 0.1
                chopped_signal = signal[:, :, cut_inf:cut_sup]
                del signal, df

                # Load calibration files
                flat = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/AIRS-CH0_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
                dark = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/AIRS-CH0_calibration_0/dark.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
                dead_airs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/AIRS-CH0_calibration_0/dead.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
                linear_corr = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/AIRS-CH0_calibration_0/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 356))[:, :, cut_inf:cut_sup]

                # Apply calibrations with GPU acceleration
                if DO_MASK:
                    chopped_signal = mask_hot_dead_gpu(chopped_signal, dead_airs, dark)

                if DO_THE_NL_CORR:
                    chopped_signal = apply_linear_corr_gpu(linear_corr, chopped_signal)

                if DO_DARK:
                    chopped_signal = clean_dark_gpu(chopped_signal, dead_airs, dark, dt_airs)

                # Process FGS data
                df_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/FGS1_signal_0.parquet'))
                fgs_signal = df_fgs.values.astype(np.float64).reshape((df_fgs.shape[0], 32, 32))

                fgs_signal = ADC_convert_gpu(fgs_signal)
                dt_fgs1 = np.ones(len(fgs_signal)) * 0.1
                dt_fgs1[1::2] += 0.1

                # Load FGS calibration
                flat_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/FGS1_calibration_0/flat.parquet')).values.astype(np.float64).reshape((32, 32))
                dark_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/FGS1_calibration_0/dark.parquet')).values.astype(np.float64).reshape((32, 32))
                dead_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/FGS1_calibration_0/dead.parquet')).values.astype(np.float64).reshape((32, 32))
                linear_corr_fgs = pd.read_parquet(os.path.join(path_folder, f'train/{planet_idx}/FGS1_calibration_0/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 32))

                # Apply FGS calibrations
                if DO_MASK:
                    fgs_signal = mask_hot_dead_gpu(fgs_signal, dead_fgs, dark_fgs)

                if DO_THE_NL_CORR:
                    fgs_signal = apply_linear_corr_gpu(linear_corr_fgs, fgs_signal)

                if DO_DARK:
                    fgs_signal = clean_dark_gpu(fgs_signal, dead_fgs, dark_fgs, dt_fgs1)

                # Get CDS
                airs_cds = get_cds_gpu(chopped_signal.reshape(1, *chopped_signal.shape))
                fgs_cds = get_cds_gpu(fgs_signal.reshape(1, *fgs_signal.shape))

                # Time binning
                if TIME_BINNING:
                    airs_binned = bin_obs_gpu(airs_cds, binning=30)
                    fgs_binned = bin_obs_gpu(fgs_cds, binning=30*12)
                else:
                    airs_binned = airs_cds.transpose(0,1,3,2) if GPU_AVAILABLE else airs_cds.transpose(0,1,3,2)
                    fgs_binned = fgs_cds.transpose(0,1,3,2) if GPU_AVAILABLE else fgs_cds.transpose(0,1,3,2)

                # Flat field correction
                if DO_FLAT:
                    airs_binned[0] = correct_flat_field_gpu(flat, dead_airs, airs_binned[0])
                    fgs_binned[0] = correct_flat_field_gpu(flat_fgs, dead_fgs, fgs_binned[0])

                # Convert back to numpy and extract features
                airs_binned = to_numpy(airs_binned)
                fgs_binned = to_numpy(fgs_binned)

                # Extract features for the model
                airs_spectrum = np.nanmean(airs_binned[0], axis=(0, 2))  # Shape: (282,)
                fgs_mean = np.nanmean(fgs_binned[0])  # Single value

                chunk_spectra.append(airs_spectrum)
                chunk_fgs.append(fgs_mean)

            all_spectra.extend(chunk_spectra)
            all_fgs.extend(chunk_fgs)

        except Exception as e:
            print(f"Error processing chunk {n}: {e}")
            continue

        # Progress update
        if (n + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (n + 1)
            eta = avg_time * (len(index) - n - 1)
            print(f"Processed {n+1}/{len(index)} chunks. ETA: {eta/60:.1f} minutes")

    # Convert to numpy arrays
    print(f"Raw data: {len(all_spectra)} spectra, {len(all_fgs)} FGS values")

    if len(all_spectra) == 0:
        print("No valid spectra processed!")
        return None, None

    spectra_array = np.array(all_spectra, dtype=np.float32)
    fgs_array = np.array(all_fgs, dtype=np.float32).reshape(-1, 1)

    print(f"Spectra array shape: {spectra_array.shape}")
    print(f"FGS array shape: {fgs_array.shape}")

    # Ensure arrays have compatible shapes
    if len(spectra_array.shape) == 1:
        spectra_array = spectra_array.reshape(1, -1)
    if len(fgs_array.shape) == 1:
        fgs_array = fgs_array.reshape(1, -1)

    # Combine AIRS spectrum + FGS
    combined_data = np.concatenate([spectra_array, fgs_array], axis=1)

    print(f"Processed {len(planet_ids)} planets")
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

    print(f"Saved calibrated data to: {output_dir}")
    print(f"Spectra shape: {combined_data.shape}")
    print(f"Targets shape: {targets_array.shape}")

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time/60:.1f} minutes")
    print(f"Average time per planet: {total_time/len(planet_ids):.2f} seconds")

    return combined_data, targets_array

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ariel_calibration_gpu.py <data_path> [output_dir] [max_planets]")
        sys.exit(1)

    data_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./calibrated_data"
    max_planets = int(sys.argv[3]) if len(sys.argv) > 3 else None  # No limit by default

    print("="*60)
    print("ARIEL Data Challenge 2025 - GPU Calibration Pipeline")
    print("="*60)

    process_ariel_data_gpu(data_path, output_dir, max_planets)