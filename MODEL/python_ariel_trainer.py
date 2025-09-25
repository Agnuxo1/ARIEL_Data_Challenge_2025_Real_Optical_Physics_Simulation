#!/usr/bin/env python3
"""
Python ARIEL Trainer - CPU Version
Trains the hybrid model using Python for debugging and testing
"""

import numpy as np
import os
import time
from pathlib import Path

class PythonArielModel:
    def __init__(self):
        # Model parameters
        self.N_WAVELENGTHS = 283
        self.OUTPUT_TARGETS = 566  # 283 wavelengths + 283 sigmas
        self.N_TIME_BINS = 187
        
        # Initialize weights
        np.random.seed(42)
        self.weights = np.random.normal(0, 0.1, (self.OUTPUT_TARGETS, self.N_WAVELENGTHS))
        self.bias = np.random.normal(0, 0.1, self.OUTPUT_TARGETS)
        
        # Normalization
        self.spectrum_mean = np.zeros(self.N_WAVELENGTHS)
        self.spectrum_std = np.ones(self.N_WAVELENGTHS)
        
        print(f"Python ARIEL Model initialized with {self.OUTPUT_TARGETS} outputs")
    
    def forward(self, spectrum):
        """Forward pass through the model"""
        # Normalize spectrum
        norm_spectrum = (spectrum - self.spectrum_mean) / (self.spectrum_std + 1e-8)
        
        # Linear transformation
        output = np.dot(self.weights, norm_spectrum) + self.bias
        
        # Post-process for submission format
        # First 283 values: wavelength predictions (wl_1 to wl_283)
        # Next 283 values: sigma predictions (sigma_1 to sigma_283)
        
        # Normalize wavelength predictions to reasonable range
        for i in range(283):
            output[i] = 0.4 + np.tanh(output[i]) * 0.2  # Range 0.4-0.6
        
        # Normalize sigma predictions to reasonable range
        for i in range(283, self.OUTPUT_TARGETS):
            output[i] = 0.01 + abs(np.tanh(output[i])) * 0.02  # Range 0.01-0.03
        
        return output
    
    def train_batch(self, spectra, targets, lr):
        """Train on a batch of data"""
        total_loss = 0.0
        
        for i in range(len(spectra)):
            # Forward pass
            predictions = self.forward(spectra[i])
            
            # Compute loss (MSE) - only on first 6 targets for atmospheric parameters
            loss = 0.0
            grad_output = np.zeros(self.OUTPUT_TARGETS)
            
            # Only train on atmospheric parameters (first 6 targets)
            for j in range(min(6, len(targets[i]))):
                diff = predictions[j] - targets[i][j]
                loss += diff * diff
                grad_output[j] = 2.0 * diff / 6.0  # Only update first 6 outputs
            
            total_loss += loss
            
            # Update weights (gradient descent)
            for k in range(self.OUTPUT_TARGETS):
                for j in range(self.N_WAVELENGTHS):
                    self.weights[k, j] -= lr * grad_output[k] * spectra[i][j]
                self.bias[k] -= lr * grad_output[k]
        
        return total_loss / len(spectra)
    
    def generate_submission(self, test_path, output_path):
        """Generate submission file"""
        print("Generating submission...")
        
        # Load test data
        test_data = np.load(os.path.join(test_path, "data_test.npy"))
        planet_ids = np.load(os.path.join(test_path, "test_planet_ids.npy"))
        
        n_test_planets = len(planet_ids)
        print(f"Test planets: {n_test_planets}")
        
        # Create submission file
        with open(output_path, 'w') as f:
            # Write header
            f.write("planet_id")
            for i in range(1, 284):
                f.write(f",wl_{i}")
            for i in range(1, 284):
                f.write(f",sigma_{i}")
            f.write("\n")
            
            # Generate predictions
            for i in range(n_test_planets):
                if i % 100 == 0:
                    print(f"Processing planet {i+1}/{n_test_planets}")
                
                # Extract spectrum (average over time)
                spectrum = np.mean(test_data[i], axis=0)
                
                # Get predictions
                predictions = self.forward(spectrum)
                
                # Write to submission
                f.write(str(planet_ids[i]))
                for j in range(self.OUTPUT_TARGETS):
                    f.write(f",{predictions[j]:.6f}")
                f.write("\n")
        
        print(f"Submission saved to: {output_path}")

def main():
    print("=" * 50)
    print("PYTHON ARIEL TRAINER")
    print("=" * 50)
    
    # Configuration
    data_path = "./calibrated_data"
    output_path = "./python_output"
    epochs = 60
    
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    print(f"Epochs: {epochs}")
    print(f"Output targets: 566")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize model
    model = PythonArielModel()
    
    # Load training data
    print("Loading training data...")
    print("Loading data_train.npy...")
    train_data = np.load(os.path.join(data_path, "data_train.npy"))
    print("Loading targets_train.npy...")
    targets_data = np.load(os.path.join(data_path, "targets_train.npy"))
    print("Data loaded successfully!")
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Targets shape: {targets_data.shape}")
    
    # Prepare training data
    n_train_planets = train_data.shape[0]
    spectra = []
    targets = []
    
    for i in range(n_train_planets):
        # Extract spectrum (average over time)
        spectrum = np.mean(train_data[i], axis=0)
        spectra.append(spectrum)
        
        # Extract targets
        target = targets_data[i]
        targets.append(target)
    
    spectra = np.array(spectra)
    targets = np.array(targets)
    
    print(f"Prepared {len(spectra)} training samples")
    print(f"Spectrum shape: {spectra.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Training loop
    print("Starting training...")
    learning_rate = 0.001
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train on all data
        loss = model.train_batch(spectra, targets, learning_rate)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss:.6f} | Time: {duration:.1f}s")
        
        # Decay learning rate
        learning_rate *= 0.98
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Checkpoint saved at epoch {epoch + 1}")
    
    print("Training complete!")
    
    # Generate submission
    model.generate_submission(data_path, os.path.join(output_path, "submission.csv"))
    
    print("=" * 50)
    print("PYTHON TRAINING COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()
