#!/usr/bin/env python3
import numpy as np
import os

print("Starting debug test...")

# Test basic imports
print("OK - Imports successful")

# Test data loading
print("Loading training data...")
train_data = np.load("calibrated_data/data_train.npy")
print(f"OK - Training data loaded: {train_data.shape}")

targets_data = np.load("calibrated_data/targets_train.npy")
print(f"OK - Targets loaded: {targets_data.shape}")

# Test data processing
print("Processing data...")
n_train_planets = train_data.shape[0]
spectra = []
targets = []

print(f"Processing {n_train_planets} planets...")
for i in range(min(10, n_train_planets)):  # Only test first 10
    spectrum = np.mean(train_data[i], axis=0)
    spectra.append(spectrum)
    targets.append(targets_data[i])
    if i % 2 == 0:
        print(f"  Processed planet {i}")

print(f"OK - Data processing successful")
print("Debug test completed!")