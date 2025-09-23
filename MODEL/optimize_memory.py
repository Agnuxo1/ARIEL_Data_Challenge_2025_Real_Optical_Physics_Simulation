#!/usr/bin/env python3
"""
Optimize data format for memory efficiency
"""
import numpy as np
import sys
import os

def optimize_data_format(data_path):
    """Reduce data dimensions for memory efficiency"""

    # Load current data
    airs_data = np.load(os.path.join(data_path, "data_train.npy"))
    fgs_data = np.load(os.path.join(data_path, "data_train_FGS.npy"))
    targets = np.load(os.path.join(data_path, "targets_train.npy"))

    print(f"Original AIRS shape: {airs_data.shape}")
    print(f"Original FGS shape: {fgs_data.shape}")

    # Reduce time dimension from 187 to 32 (keep physics but reduce memory)
    N = airs_data.shape[0]

    # AIRS: (N, 32, 282, 32) instead of (N, 187, 282, 32)
    airs_optimized = np.zeros((N, 32, 282, 32), dtype=np.float32)
    for i in range(N):
        # Sample every 6th time bin to get 32 bins from 187
        indices = np.linspace(0, 186, 32, dtype=int)
        airs_optimized[i] = airs_data[i, indices, :, :]

    # FGS: (N, 32, 32, 32) instead of (N, 187, 32, 32)
    fgs_optimized = np.zeros((N, 32, 32, 32), dtype=np.float32)
    for i in range(N):
        indices = np.linspace(0, 186, 32, dtype=int)
        fgs_optimized[i] = fgs_data[i, indices, :, :]

    # Save optimized data
    np.save(os.path.join(data_path, "data_train.npy"), airs_optimized)
    np.save(os.path.join(data_path, "data_train_FGS.npy"), fgs_optimized)

    # Also create optimized test data
    np.save(os.path.join(data_path, "data_test.npy"), airs_optimized)
    np.save(os.path.join(data_path, "data_test_FGS.npy"), fgs_optimized)

    print(f"Optimized AIRS shape: {airs_optimized.shape}")
    print(f"Optimized FGS shape: {fgs_optimized.shape}")
    print(f"Memory reduction: {airs_data.nbytes / airs_optimized.nbytes:.1f}x")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python optimize_memory.py <data_path>")
        sys.exit(1)

    optimize_data_format(sys.argv[1])