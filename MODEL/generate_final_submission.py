#!/usr/bin/env python3
"""
FINAL SUBMISSION GENERATOR FOR ARIEL DATA CHALLENGE 2025
Uses trained hybrid quantum-NEBULA model for physics-based predictions
"""

import numpy as np
import pandas as pd
import os
import sys
import time
from hybrid_ariel_python import HybridArielModel

# Configuration
MODEL_PATH = "./hybrid_training_outputs/best_model.pkl"
TEST_DATA_PATH = "E:/ARIEL_COMPLETE_BACKUP_2025-09-22_19-40-31/ARIEL_REAL_PHYSIC_SIMULATE_NeurIPS/calibrated_data"
OUTPUT_PATH = "./final_submission.csv"

def load_test_data():
    """Load official test data"""
    print("[DATA] Loading official test data...")

    try:
        test_data_path = os.path.join(TEST_DATA_PATH, "data_test.npy")
        test_data = np.load(test_data_path)
        print(f"  Test data shape: {test_data.shape}")

        # Load planet IDs from calibrated data
        test_ids_path = os.path.join(TEST_DATA_PATH, "test_planet_ids.npy")
        if os.path.exists(test_ids_path):
            test_ids = np.load(test_ids_path)
        else:
            # Fallback: generate IDs starting from 1100001
            n_test = len(test_data)
            test_ids = np.arange(1100001, 1100001 + n_test)
        n_test = len(test_data)
        print(f"  Test samples: {n_test}")
        print(f"  Planet ID range: {test_ids[0]} to {test_ids[-1]}")

        return test_data, test_ids
    except Exception as e:
        print(f"ERROR loading test data: {e}")
        return None, None

def generate_predictions(model, test_data):
    """Generate predictions using trained hybrid model"""
    print("[PREDICT] Generating predictions using hybrid quantum-optical model...")

    n_test = len(test_data)
    predictions_list = []
    start_time = time.time()

    for i in range(n_test):
        if (i + 1) % 50 == 0 or i == n_test - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (n_test - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i + 1:4d}/{n_test} ({100*(i+1)/n_test:5.1f}%) | Rate: {rate:.1f}/s | ETA: {eta:.0f}s")

        # Process spectrum (average over time if needed)
        if len(test_data.shape) == 3:
            spectrum = np.mean(test_data[i], axis=0)
        else:
            spectrum = test_data[i]

        # Forward pass through hybrid model
        predictions = model.forward(spectrum)
        predictions_list.append(predictions)

    total_time = time.time() - start_time
    print(f"  Completed {n_test} predictions in {total_time:.1f}s")
    return np.array(predictions_list)

def create_submission(test_ids, predictions):
    """Create submission DataFrame"""
    print("[FORMAT] Creating submission DataFrame...")

    # Create columns
    columns = ['planet_id']
    for i in range(1, 284): columns.append(f'wl_{i}')
    for i in range(1, 284): columns.append(f'sigma_{i}')

    # Create DataFrame
    submission_data = np.column_stack([test_ids, predictions])
    submission_df = pd.DataFrame(submission_data, columns=columns)
    submission_df['planet_id'] = submission_df['planet_id'].astype(int)

    print(f"  Shape: {submission_df.shape}")
    return submission_df

def main():
    """Main submission pipeline"""
    print("=" * 70)
    print("ARIEL DATA CHALLENGE 2025 - FINAL SUBMISSION")
    print("Hybrid Quantum-NEBULA Model - Physics-Based Spectroscopy")
    print("=" * 70)

    # Check for trained model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Trained model not found at: {MODEL_PATH}")
        return 1

    # Load model
    print(f"[MODEL] Loading trained model: {MODEL_PATH}")
    model = HybridArielModel()
    model.load_checkpoint(MODEL_PATH)

    # Load test data
    test_data, test_ids = load_test_data()
    if test_data is None:
        return 1

    # Generate predictions
    predictions = generate_predictions(model, test_data)

    # Create submission
    submission_df = create_submission(test_ids, predictions)

    # Save
    submission_df.to_csv(OUTPUT_PATH, index=False, float_format='%.6f')
    print(f"[SAVE] Submission saved: {OUTPUT_PATH}")
    print(f"  File size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.2f} MB")

    # Validation
    print(f"[VALIDATE] Checking format...")
    assert len(submission_df) > 0, "Empty submission"
    assert len(submission_df.columns) == 567, f"Expected 567 columns, got {len(submission_df.columns)}"
    print(f"  ✅ Format valid!")

    print("\n" + "=" * 70)
    print("✅ FINAL SUBMISSION GENERATED!")
    print(f"📁 File: {OUTPUT_PATH}")
    print(f"🚀 Ready for Kaggle upload!")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
