#!/usr/bin/env python3
"""
ARIEL Data Challenge 2025 - Submission Generator
Converts Hybrid Quantum-NEBULA predictions to spectral format
"""

import numpy as np
import pandas as pd
import sys
import os

def load_model_predictions():
    """Load atmospheric parameters from trained Hybrid Quantum-NEBULA model"""
    import json

    # Try to load real model predictions
    if os.path.exists("real_model_predictions.json"):
        print("Loading REAL model predictions from trained checkpoint...")
        with open("real_model_predictions.json", "r") as f:
            predictions = json.load(f)
        print(f"Loaded real predictions: {predictions}")
        return predictions
    else:
        print("Warning: Using fallback synthetic values - real predictions not found")
        predictions = {
            'CO2': 100.0,      # ppm
            'H2O': 50.0,       # ppm
            'CH4': 5.0,        # ppm
            'NH3': 0.5,        # ppm
            'temperature': 1500.0,  # K
            'radius': 1.0      # Earth radii
        }
        return predictions

def atmospheric_to_spectrum(co2, h2o, ch4, nh3, temp, radius):
    """
    Convert atmospheric parameters to transit spectrum using physics
    Based on real ARIEL wavelength grid and atmospheric modeling
    """

    # Load official wavelength grid
    wavelengths_file = "/mnt/disco2/wavelengths.csv"
    if os.path.exists(wavelengths_file):
        wl_data = pd.read_csv(wavelengths_file)
        # Take the second row which contains the actual wavelength values
        wavelengths = wl_data.iloc[0].values  # First row of data after header
    else:
        # Fallback: ARIEL wavelength range 0.5-7.8 microns
        wavelengths = np.logspace(np.log10(0.5), np.log10(7.8), 283)

    spectrum = []
    sigma = []

    for wl in wavelengths:
        # Physics-based spectral model
        # Baseline continuum adjusted by planet size
        baseline = 0.400 + (radius - 1.0) * 0.1  # Size-dependent baseline

        # Molecular absorption features
        absorption = 0.0

        # H2O absorption (strong at 1.4, 1.9, 2.7, 6.3 microns) - scaled by real concentration
        h2o_scale = h2o / 50.0  # Normalize relative to expected range
        if 1.3 <= wl <= 1.5:
            absorption += h2o_scale * 2e-5 * np.exp(-((wl-1.4)/0.1)**2)
        if 1.8 <= wl <= 2.0:
            absorption += h2o_scale * 3e-5 * np.exp(-((wl-1.9)/0.1)**2)
        if 2.6 <= wl <= 2.8:
            absorption += h2o_scale * 4e-5 * np.exp(-((wl-2.7)/0.1)**2)
        if 6.0 <= wl <= 6.6:
            absorption += h2o_scale * 6e-5 * np.exp(-((wl-6.3)/0.3)**2)

        # CO2 absorption (strong at 2.0, 2.7, 4.3, 15 microns) - scaled by real concentration
        co2_scale = co2 / 100.0  # Normalize relative to expected range
        if 1.9 <= wl <= 2.1:
            absorption += co2_scale * 1.5e-5 * np.exp(-((wl-2.0)/0.1)**2)
        if 2.6 <= wl <= 2.8:
            absorption += co2_scale * 2e-5 * np.exp(-((wl-2.7)/0.1)**2)
        if 4.2 <= wl <= 4.4:
            absorption += co2_scale * 3e-5 * np.exp(-((wl-4.3)/0.1)**2)

        # CH4 absorption (strong at 1.7, 2.3, 3.3 microns) - scaled by real concentration
        ch4_scale = ch4 / 5.0  # Normalize relative to expected range
        if 1.6 <= wl <= 1.8:
            absorption += ch4_scale * 1e-5 * np.exp(-((wl-1.7)/0.1)**2)
        if 2.2 <= wl <= 2.4:
            absorption += ch4_scale * 1.5e-5 * np.exp(-((wl-2.3)/0.1)**2)
        if 3.2 <= wl <= 3.4:
            absorption += ch4_scale * 2e-5 * np.exp(-((wl-3.3)/0.1)**2)

        # NH3 absorption (strong at 1.5, 2.0, 10.5 microns) - scaled by real concentration
        nh3_scale = nh3 / 0.5  # Normalize relative to expected range
        if 1.4 <= wl <= 1.6:
            absorption += nh3_scale * 0.5e-5 * np.exp(-((wl-1.5)/0.1)**2)
        if 1.9 <= wl <= 2.1:
            absorption += nh3_scale * 0.8e-5 * np.exp(-((wl-2.0)/0.1)**2)

        # Temperature effect (Rayleigh scattering slope)
        rayleigh = 0.0001 * (temp/1500.0) * (0.55/wl)**4

        # Size effect
        size_factor = radius**2

        # Final spectrum value
        transit_depth = (baseline + absorption + rayleigh) * size_factor

        # Add realistic noise level
        noise_level = 0.001 * np.sqrt(baseline)  # Photon noise scaling

        spectrum.append(transit_depth)
        sigma.append(noise_level)

    return np.array(spectrum), np.array(sigma)

def generate_submission():
    """Generate submission file for ARIEL challenge"""

    print("Generating ARIEL submission using Hybrid Quantum-NEBULA model predictions...")

    # Load sample submission format
    sample_file = "/mnt/disco2/sample_submission.csv"
    sample_df = pd.read_csv(sample_file)

    # Get test planet ID
    planet_id = sample_df['planet_id'].iloc[0]
    print(f"Generating prediction for planet: {planet_id}")

    # Load our model predictions
    predictions = load_model_predictions()
    print(f"Model predictions: {predictions}")

    # Convert to spectrum using physics
    spectrum, sigma = atmospheric_to_spectrum(
        predictions['CO2'],
        predictions['H2O'],
        predictions['CH4'],
        predictions['NH3'],
        predictions['temperature'],
        predictions['radius']
    )

    # Create submission dataframe
    submission_data = {'planet_id': [planet_id]}

    # Add wavelength columns
    for i in range(283):
        submission_data[f'wl_{i+1}'] = [spectrum[i]]

    # Add sigma columns
    for i in range(283):
        submission_data[f'sigma_{i+1}'] = [sigma[i]]

    submission_df = pd.DataFrame(submission_data)

    # Save submission
    output_file = "ariel_quantum_nebula_submission.csv"
    submission_df.to_csv(output_file, index=False)

    print(f"\n✅ SUBMISSION GENERATED: {output_file}")
    print(f"Shape: {submission_df.shape}")
    print(f"Columns: {len(submission_df.columns)} (expected: 567 = planet_id + 283*2)")

    # Verify format
    print(f"\nSpectrum range: {spectrum.min():.6f} - {spectrum.max():.6f}")
    print(f"Sigma range: {sigma.min():.6f} - {sigma.max():.6f}")

    return output_file

if __name__ == "__main__":
    submission_file = generate_submission()
    print(f"\n🚀 Ready for Kaggle upload: {submission_file}")