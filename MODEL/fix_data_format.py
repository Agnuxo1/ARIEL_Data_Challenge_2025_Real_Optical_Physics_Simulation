#!/usr/bin/env python3
"""
Fix data format to match expected C++ code structure
"""
import numpy as np
import sys
import os

def fix_data_format(data_path):
    """Split combined data into expected format"""

    # Load our combined data
    combined_data = np.load(os.path.join(data_path, "data_train.npy"))
    targets = np.load(os.path.join(data_path, "targets_train.npy"))

    print(f"Loaded combined data shape: {combined_data.shape}")
    print(f"Loaded targets shape: {targets.shape}")

    # Split combined data
    # combined_data is (N, 283) where 283 = 282 AIRS + 1 FGS
    airs_data = combined_data[:, :282]  # (N, 282)
    fgs_data = combined_data[:, 282:283]  # (N, 1)

    # The C++ code expects different shapes based on the original notebook:
    # AIRS: (N, time_bins, wavelengths, spatial)
    # FGS: (N, time_bins, spatial, spatial)

    # For now, let's create compatible shapes by expanding
    N = airs_data.shape[0]

    # AIRS: (N, 187, 282, 32) - 187 time bins, 282 wavelengths, 32 spatial
    airs_expanded = np.zeros((N, 187, 282, 32), dtype=np.float32)
    for i in range(N):
        # Repeat the spectrum across time and spatial dimensions
        for t in range(187):
            for s in range(32):
                airs_expanded[i, t, :, s] = airs_data[i, :]

    # FGS: (N, 187, 32, 32) - 187 time bins, 32x32 spatial
    fgs_expanded = np.zeros((N, 187, 32, 32), dtype=np.float32)
    for i in range(N):
        # Repeat the FGS value across time and spatial dimensions
        fgs_value = fgs_data[i, 0]
        fgs_expanded[i, :, :, :] = fgs_value

    # Save in expected format
    np.save(os.path.join(data_path, "data_train.npy"), airs_expanded)
    np.save(os.path.join(data_path, "data_train_FGS.npy"), fgs_expanded)

    print(f"Saved AIRS data shape: {airs_expanded.shape}")
    print(f"Saved FGS data shape: {fgs_expanded.shape}")
    print(f"Targets shape: {targets.shape}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_data_format.py <data_path>")
        sys.exit(1)

    fix_data_format(sys.argv[1])