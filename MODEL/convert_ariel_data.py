#!/usr/bin/env python3
"""
Convert ARIEL Data Challenge parquet format to numpy arrays for the hybrid model
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

def convert_ariel_data(data_path):
    """Convert ARIEL parquet data to numpy format expected by the C++ model"""

    data_path = Path(data_path)
    train_dir = data_path / "train"

    # Read the labels
    train_csv = pd.read_csv(data_path / "train.csv")
    print(f"Found {len(train_csv)} training examples")

    # Get list of planet directories
    planet_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    planet_dirs = sorted(planet_dirs, key=lambda x: int(x.name))

    print(f"Processing {len(planet_dirs)} planets...")

    # Initialize arrays
    all_spectra = []
    all_targets = []

    for i, planet_dir in enumerate(planet_dirs[:100]):  # Start with first 100 for testing
        planet_id = int(planet_dir.name)

        try:
            # Read AIRS spectral data
            airs_path = planet_dir / "AIRS-CH0_signal_0.parquet"
            if airs_path.exists():
                airs_data = pd.read_parquet(airs_path)

                # Average over time to get mean spectrum (282 wavelengths)
                spectrum = airs_data.mean().values[:282]  # Take first 282 wavelengths

                # Read FGS photometry data
                fgs_path = planet_dir / "FGS1_signal_0.parquet"
                if fgs_path.exists():
                    fgs_data = pd.read_parquet(fgs_path)
                    # Average over time and pixels
                    fgs_mean = fgs_data.mean().mean()  # Single value

                    # Combine spectrum + FGS
                    combined = np.append(spectrum, fgs_mean)
                    all_spectra.append(combined)

                    # Get targets for this planet
                    planet_targets = train_csv[train_csv['planet_id'] == planet_id]
                    if len(planet_targets) > 0:
                        # Extract the 6 atmospheric parameters (columns 1-282 are spectrum)
                        targets = planet_targets.iloc[0, 1:7].values  # First 6 target columns
                        all_targets.append(targets)
                    else:
                        print(f"Warning: No targets found for planet {planet_id}")
                        all_targets.append(np.zeros(6))

        except Exception as e:
            print(f"Error processing planet {planet_id}: {e}")
            continue

        if i % 10 == 0:
            print(f"Processed {i+1}/{len(planet_dirs[:100])} planets")

    # Convert to numpy arrays
    spectra_array = np.array(all_spectra, dtype=np.float32)
    targets_array = np.array(all_targets, dtype=np.float32)

    print(f"Final data shape: spectra {spectra_array.shape}, targets {targets_array.shape}")

    # Save as numpy files
    np.save(data_path / "data_train.npy", spectra_array)
    np.save(data_path / "targets_train.npy", targets_array)

    print("Data conversion complete!")
    print(f"Saved to: {data_path / 'data_train.npy'}")
    print(f"Saved to: {data_path / 'targets_train.npy'}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_ariel_data.py <data_path>")
        sys.exit(1)

    convert_ariel_data(sys.argv[1])