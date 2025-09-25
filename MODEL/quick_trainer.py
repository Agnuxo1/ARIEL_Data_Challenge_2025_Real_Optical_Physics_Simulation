#!/usr/bin/env python3
import numpy as np
import os
import time

def main():
    print("=" * 50)
    print("QUICK ARIEL TRAINER - 60 EPOCHS")
    print("=" * 50)

    # Configuration
    data_path = "./calibrated_data"
    output_path = "./python_output"
    epochs = 60

    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    print(f"Epochs: {epochs}")
    print("=" * 50)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load training data
    print("Loading training data...")
    train_data = np.load(os.path.join(data_path, "data_train.npy"))
    targets_data = np.load(os.path.join(data_path, "targets_train.npy"))

    print(f"Training data shape: {train_data.shape}")
    print(f"Targets shape: {targets_data.shape}")

    # Prepare data (simplified)
    print("Preparing data...")
    n_train = train_data.shape[0]
    n_wavelengths = train_data.shape[2]  # 283
    n_targets = targets_data.shape[1]    # 6

    # Average spectra over time
    spectra = np.mean(train_data, axis=1)  # (1100, 283)
    targets = targets_data  # (1100, 6)

    print(f"Prepared spectra shape: {spectra.shape}")
    print(f"Prepared targets shape: {targets.shape}")

    # Simple model parameters
    print("Initializing model...")
    np.random.seed(42)
    weights = np.random.normal(0, 0.01, (n_targets, n_wavelengths))
    bias = np.zeros(n_targets)

    # Training loop
    print("Starting training...")
    learning_rate = 0.001

    for epoch in range(epochs):
        start_time = time.time()

        total_loss = 0.0

        # Simple batch training
        for i in range(n_train):
            # Forward pass
            prediction = np.dot(weights, spectra[i]) + bias

            # Loss (MSE)
            loss = np.mean((prediction - targets[i])**2)
            total_loss += loss

            # Backward pass (simple gradient descent)
            error = prediction - targets[i]

            # Update weights
            for j in range(n_targets):
                weights[j] -= learning_rate * error[j] * spectra[i] / n_train
                bias[j] -= learning_rate * error[j] / n_train

        avg_loss = total_loss / n_train

        end_time = time.time()
        duration = end_time - start_time

        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | Time: {duration:.1f}s")

        # Decay learning rate
        learning_rate *= 0.99

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_file = os.path.join(output_path, f"checkpoint_epoch_{epoch+1}.npy")
            np.save(checkpoint_file, {'weights': weights, 'bias': bias})
            print(f"  Checkpoint saved: {checkpoint_file}")

    # Save final model
    final_model_file = os.path.join(output_path, "final_model.npy")
    np.save(final_model_file, {'weights': weights, 'bias': bias})
    print(f"Final model saved: {final_model_file}")

    print("=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()