#!/usr/bin/env python3
"""
TEST CHECKPOINT BEFORE KAGGLE UPLOAD
Quick validation that our best_model.pkl works correctly
"""

import numpy as np
import pandas as pd
import pickle
import os
from hybrid_ariel_python import HybridArielModel

print("=" * 60)
print("TESTING CHECKPOINT BEFORE KAGGLE UPLOAD")
print("=" * 60)

# Test checkpoint loading
MODEL_PATH = "./hybrid_training_outputs/best_model.pkl"
print(f"Testing checkpoint: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print("ERROR: Checkpoint not found!")
    exit(1)

print(f"File size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB")

# Load checkpoint
print("\n[1] Loading checkpoint...")
try:
    with open(MODEL_PATH, 'rb') as f:
        checkpoint = pickle.load(f)
    print("OK Checkpoint loaded successfully")

    # Show checkpoint contents
    print(f"  Keys: {list(checkpoint.keys())}")
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            print(f"  {key}: {list(value.keys())}")
        else:
            print(f"  {key}: {np.array(value).shape if hasattr(value, 'shape') else type(value)}")

except Exception as e:
    print(f"ERROR: {e}")
    exit(1)

# Test model initialization
print("\n[2] Initializing model...")
model = HybridArielModel()
model.load_checkpoint(MODEL_PATH)
print("OK Model initialized with checkpoint parameters")

# Test with dummy spectrum
print("\n[3] Testing forward pass...")
try:
    dummy_spectrum = np.random.normal(0.015, 0.011, 283)
    predictions = model.forward(dummy_spectrum)
    print(f"OK Forward pass successful")
    print(f"  Input shape: {dummy_spectrum.shape}")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Output range: [{predictions.min():.6f}, {predictions.max():.6f}]")

    # Check for NaN or infinite values
    if np.any(np.isnan(predictions)):
        print("WARNING: NaN values in predictions")
        nan_count = np.sum(np.isnan(predictions))
        print(f"  NaN count: {nan_count}/{len(predictions)}")
    elif np.any(np.isinf(predictions)):
        print("WARNING: Infinite values in predictions")
    else:
        print("OK All predictions are finite")

except Exception as e:
    print(f"ERROR during forward pass: {e}")
    exit(1)

# Test submission format
print("\n[4] Testing submission format...")
n_test_dummy = 5
dummy_ids = np.arange(1100001, 1100001 + n_test_dummy)
dummy_predictions = []

for i in range(n_test_dummy):
    spectrum = np.random.normal(0.015, 0.011, 283)
    preds = model.forward(spectrum)
    dummy_predictions.append(preds)

predictions_array = np.array(dummy_predictions)

# Create submission format
columns = ['planet_id']
for i in range(1, 284): columns.append(f'wl_{i}')
for i in range(1, 284): columns.append(f'sigma_{i}')

submission_data = np.column_stack([dummy_ids, predictions_array])
submission_df = pd.DataFrame(submission_data, columns=columns)
submission_df['planet_id'] = submission_df['planet_id'].astype(int)

print(f"OK Submission DataFrame created")
print(f"  Shape: {submission_df.shape}")
print(f"  Columns: {len(submission_df.columns)} (expected: 567)")
print(f"  Sample IDs: {submission_df['planet_id'].iloc[:3].tolist()}")

# Validate format
try:
    assert len(submission_df.columns) == 567
    assert not submission_df.isnull().any().any()
    assert submission_df.columns[0] == 'planet_id'
    print("OK Format validation passed")
except AssertionError as e:
    print(f"ERROR: Format validation failed - {e}")
    exit(1)

print("\n" + "=" * 60)
print("SUCCESS: CHECKPOINT VALIDATION SUCCESSFUL!")
print("Ready for Kaggle upload:")
print(f"   File: {MODEL_PATH}")
print(f"   Size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB")
print("   Status: Stable epoch 5 parameters")
print("=" * 60)