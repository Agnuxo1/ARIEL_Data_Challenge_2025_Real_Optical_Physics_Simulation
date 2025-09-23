#!/usr/bin/env python3
import numpy as np
import os

# Create test data directory
data_dir = "test_data"
os.makedirs(data_dir, exist_ok=True)

print("Creating synthetic ARIEL test data...")

# Create synthetic targets (atmospheric parameters)
# Based on ARIEL challenge: 6 parameters (T, log_H2O, log_CO2, log_CO, log_CH4, log_NH3)
num_samples = 1000
num_targets = 6

targets = np.random.randn(num_samples, num_targets).astype(np.float32)
np.save(os.path.join(data_dir, "targets_train.npy"), targets)
print(f"Created targets_train.npy: {targets.shape}")

# Create synthetic spectral data (AIRS)
# Typical ARIEL spectral data: wavelength x flux
num_wavelengths = 283  # Common ARIEL wavelength grid size
spectral_data = np.random.randn(num_samples, num_wavelengths).astype(np.float32)
np.save(os.path.join(data_dir, "data_train.npy"), spectral_data)
print(f"Created data_train.npy: {spectral_data.shape}")

# Create synthetic FGS data (Fine Guidance Sensor)
fgs_data = np.random.randn(num_samples, 100).astype(np.float32)  # Smaller FGS dataset
np.save(os.path.join(data_dir, "data_train_FGS.npy"), fgs_data)
print(f"Created data_train_FGS.npy: {fgs_data.shape}")

# Create test data (for prediction)
test_samples = 200
test_spectral = np.random.randn(test_samples, num_wavelengths).astype(np.float32)
np.save(os.path.join(data_dir, "data_test.npy"), test_spectral)
print(f"Created data_test.npy: {test_spectral.shape}")

print(f"\nSynthetic ARIEL data created in: {os.path.abspath(data_dir)}")
print("Files created:")
print(f"  - targets_train.npy: {targets.shape}")
print(f"  - data_train.npy: {spectral_data.shape}")
print(f"  - data_train_FGS.npy: {fgs_data.shape}")
print(f"  - data_test.npy: {test_spectral.shape}")