#!/usr/bin/env python3
"""
FULL ARIEL HYBRID TRAINER - PHYSICS-BASED MODEL
Training script using quantum-optical physics simulation
This prepares the model weights for the final C++/CUDA implementation
"""

import numpy as np
import os
import sys
import time
from pathlib import Path
from hybrid_ariel_python import HybridArielModel, export_to_cpp_format

# Data paths - using calibrated data with 1100 planets
CALIBRATED_DATA_PATH = "E:/ARIEL_COMPLETE_BACKUP_2025-09-22_19-40-31/ARIEL_REAL_PHYSIC_SIMULATE_NeurIPS/calibrated_data"
TEST_DATA_PATH = "E:/NeurIPS_MYCELIUM_EVOLUTUM/ariel-data-challenge-2025"
OUTPUT_PATH = "./hybrid_training_outputs"

# Training configuration
EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
LR_DECAY = 0.98
SAVE_EVERY = 5
VALIDATION_SPLIT = 0.2

def load_ariel_data():
    """Load calibrated ARIEL data - 1100 planets with proper physics"""
    print("=" * 60)
    print("ARIEL DATA CHALLENGE 2025 - HYBRID QUANTUM-NEBULA MODEL")
    print("Physics-Based Approach with Quantum-Optical Processing")
    print("=" * 60)
    print()

    print("[DATA] Loading calibrated ARIEL data...")
    print(f"  Source: {CALIBRATED_DATA_PATH}")

    # Load training data
    data_train = np.load(os.path.join(CALIBRATED_DATA_PATH, "data_train.npy"))
    targets_train = np.load(os.path.join(CALIBRATED_DATA_PATH, "targets_train.npy"))

    print(f"  Training data shape: {data_train.shape}")  # (1100, 187, 283)
    print(f"  Training targets shape: {targets_train.shape}")  # (1100, 6)
    print(f"  Spectral channels: {data_train.shape[2]} wavelengths")
    print(f"  Time bins: {data_train.shape[1]}")
    print(f"  Total planets: {data_train.shape[0]}")

    return data_train, targets_train

def prepare_training_data(data_train, targets_train):
    """Prepare data for physics-based training"""
    print("\\n[PREP] Preparing data for quantum-optical processing...")

    # Average over time to get mean spectrum per planet
    # This simulates the time-averaged observations that would be used
    spectra = np.mean(data_train, axis=1)  # (1100, 283)

    print(f"  Time-averaged spectra shape: {spectra.shape}")

    # Train/validation split with fixed seed for reproducibility
    np.random.seed(42)
    n_samples = len(spectra)
    n_val = int(n_samples * VALIDATION_SPLIT)

    indices = np.random.permutation(n_samples)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    train_X = spectra[train_indices]
    train_Y = targets_train[train_indices]
    val_X = spectra[val_indices]
    val_Y = targets_train[val_indices]

    print(f"  Training set: {train_X.shape[0]} planets")
    print(f"  Validation set: {val_X.shape[0]} planets")
    print(f"  Target parameters: {train_Y.shape[1]} (CO2, H2O, CH4, NH3, T, R)")

    return train_X, train_Y, val_X, val_Y

def train_epoch(model, train_X, train_Y, epoch, learning_rate):
    """Train one epoch with physics-based processing"""
    n_batches = len(train_X) // BATCH_SIZE
    total_loss = 0.0
    batch_losses = []

    print(f"\\r[Epoch {epoch:3d}] Processing batches: ", end="", flush=True)

    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        batch_spectra = [train_X[i] for i in range(start_idx, end_idx)]
        batch_targets = [train_Y[i] for i in range(start_idx, end_idx)]

        # Physics-based forward pass and training
        batch_loss = model.train_batch(batch_spectra, batch_targets, learning_rate)
        batch_losses.append(batch_loss)
        total_loss += batch_loss

        # Progress indicator
        if (batch_idx + 1) % 100 == 0 or batch_idx == n_batches - 1:
            print(f"{batch_idx + 1}/{n_batches}", end=" ", flush=True)

    avg_loss = total_loss / n_batches
    print(f"| Loss: {avg_loss:.6f}")

    return avg_loss

def validate(model, val_X, val_Y):
    """Validate model on validation set"""
    val_losses = []
    mae_per_target = np.zeros(6)
    target_names = ["CO2", "H2O", "CH4", "NH3", "Temp", "Radius"]

    print("  Validating...", end=" ", flush=True)

    for i in range(len(val_X)):
        spectrum = val_X[i]
        target = val_Y[i]

        # Forward pass only (no training)
        predictions = model.forward(spectrum)
        atm_pred = predictions[:6]  # First 6 are atmospheric parameters

        loss = np.mean((atm_pred - target)**2)
        val_losses.append(loss)

        # MAE per target
        mae_per_target += np.abs(atm_pred - target)

    mae_per_target /= len(val_X)
    avg_loss = np.mean(val_losses)

    print("\\n    Validation MAE by parameter:")
    for i, name in enumerate(target_names):
        print(f"      {name}: {mae_per_target[i]:.6f}")

    return avg_loss

def main():
    """Main training loop for hybrid quantum-NEBULA model"""
    start_time = time.time()

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Load data
    try:
        data_train, targets_train = load_ariel_data()
        train_X, train_Y, val_X, val_Y = prepare_training_data(data_train, targets_train)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return 1

    # Initialize physics-based hybrid model
    print("\\n[MODEL] Initializing Hybrid Quantum-NEBULA Model...")
    model = HybridArielModel()

    # Compute normalization from training data
    model.compute_normalization(train_X)

    # Training loop
    print("\\n[TRAINING] Starting physics-based training...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  LR decay: {LR_DECAY} every 20 epochs")
    print()

    best_val_loss = float('inf')
    best_epoch = -1
    train_losses = []
    val_losses = []

    current_lr = LEARNING_RATE

    for epoch in range(1, EPOCHS + 1):
        # Training
        train_loss = train_epoch(model, train_X, train_Y, epoch, current_lr)
        train_losses.append(train_loss)

        # Validation
        val_loss = validate(model, val_X, val_Y)
        val_losses.append(val_loss)

        # Learning rate decay
        if epoch % 20 == 0:
            current_lr *= LR_DECAY
            print(f"    Learning rate decayed to: {current_lr:.2e}")

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            model.save_checkpoint(f"{OUTPUT_PATH}/best_model.pkl")
            print(f"    * NEW BEST MODEL * (epoch {epoch})")

        # Regular checkpoints
        if epoch % SAVE_EVERY == 0:
            model.save_checkpoint(f"{OUTPUT_PATH}/model_epoch_{epoch}.pkl")

        # Progress summary
        elapsed = time.time() - start_time
        print(f"    Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {elapsed:.0f}s")
        print()

    # Training complete
    elapsed_total = time.time() - start_time
    print("=" * 60)
    print("TRAINING COMPLETE!")
    print(f"  Total time: {elapsed_total:.0f} seconds ({elapsed_total/60:.1f} minutes)")
    print(f"  Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"  Final train loss: {train_losses[-1]:.6f}")
    print("=" * 60)

    # Load best model for final operations
    print("\\n[FINAL] Loading best model for submission generation...")
    model.load_checkpoint(f"{OUTPUT_PATH}/best_model.pkl")

    # Export parameters for C++/CUDA implementation
    print("\\n[EXPORT] Exporting parameters for C++/CUDA model...")
    export_files = export_to_cpp_format(model, f"{OUTPUT_PATH}/cpp_model_params")

    print("\\nC++/CUDA Export Summary:")
    for name, path in export_files.items():
        print(f"  {name}: {path}")

    # Generate test submission
    print("\\n[SUBMISSION] Generating Kaggle submission...")
    try:
        model.generate_submission(TEST_DATA_PATH, f"{OUTPUT_PATH}/kaggle_submission.csv")
    except Exception as e:
        print(f"Warning: Could not generate submission: {e}")

    # Save training metrics
    np.save(f"{OUTPUT_PATH}/train_losses.npy", train_losses)
    np.save(f"{OUTPUT_PATH}/val_losses.npy", val_losses)

    print("\\n" + "=" * 60)
    print("HYBRID QUANTUM-NEBULA MODEL TRAINING COMPLETE")
    print("Physics-based weights exported for C++/CUDA implementation")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    sys.exit(main())