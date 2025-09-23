# 🚀 ARIEL DATA CHALLENGE 2025 - 3-DAY ROADMAP

## HYBRID QUANTUM-NEBULA MODEL - COMPLETE PIPELINE

### 📅 TIME CRITICAL: 3 DAYS REMAINING

---

## DAY 1: Setup & Data Preparation (TODAY)

### Morning (4 hours)
```bash
# 1. Install Dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake git
sudo apt-get install -y libblas-dev liblapack-dev libz-dev
sudo apt-get install -y nvidia-cuda-toolkit

# 2. Install ITensor
cd ~
git clone https://github.com/ITensor/ITensor
cd ITensor
cp options.mk.sample options.mk
# Edit options.mk: set BLAS/LAPACK paths
make -j8

# 3. Install cnpy
cd ~
git clone https://github.com/rogersce/cnpy
cd cnpy
cmake .
make

# 4. Setup project
cd smb://nebula-system-product-name.local/disco2/
mkdir hybrid_model
cd hybrid_model
```

### Afternoon (4 hours)
```python
# 5. Preprocess Ariel data using their notebook
python calibrate_ariel_data.py --input ../train --output ./preprocessed

# 6. Verify data
python verify_data.py

# Expected output:
# Train samples: 600+
# Wavelengths: 282
# Time bins: 187
```

### Evening (2 hours)
```bash
# 7. Compile hybrid model
make -f Makefile_ariel nebula_kernels.cu
make -f Makefile_ariel

# 8. Test compilation
./ariel_trainer --epochs 1 --batch 4
```

---

## DAY 2: Training & Optimization

### Morning (6 hours)
```bash
# 1. Initial training run
./ariel_trainer \
  --data ../preprocessed \
  --epochs 50 \
  --batch 32 \
  --lr 0.001 \
  --output ./run1

# Monitor loss curves
tail -f run1/metrics.csv
```

### Afternoon (4 hours)
```bash
# 2. Hyperparameter optimization
for lr in 0.001 0.0005 0.0001; do
  for batch in 16 32 64; do
    ./ariel_trainer \
      --data ../preprocessed \
      --epochs 100 \
      --batch $batch \
      --lr $lr \
      --output ./runs/lr${lr}_b${batch}
  done
done

# 3. Select best model
python select_best_model.py --runs ./runs
```

### Evening (2 hours)
```bash
# 4. Extended training with best params
./ariel_trainer \
  --data ../preprocessed \
  --epochs 200 \
  --batch 32 \
  --lr 0.0005 \
  --output ./final_run
```

---

## DAY 3: Final Training & Submission

### Morning (4 hours)
```bash
# 1. Final training with ensemble
for seed in 42 1337 2024; do
  ./ariel_trainer \
    --data ../preprocessed \
    --epochs 150 \
    --batch 32 \
    --lr 0.0005 \
    --seed $seed \
    --output ./ensemble/model_$seed
done

# 2. Create ensemble predictions
python ensemble_predictions.py \
  --models ./ensemble/model_* \
  --output ./ensemble_submission.csv
```

### Afternoon (3 hours)
```python
# 3. Generate final submission
import pandas as pd
import numpy as np

# Load test data
test = np.load('preprocessed/data_test.npy')

# Run inference
./ariel_trainer --inference --model ./final_run/checkpoint_best

# Format submission
submission = pd.read_csv('outputs/submission.csv')
submission.columns = ['planet_ID', 'CO2', 'H2O', 'CH4', 'NH3', 'temperature', 'radius']

# Validate ranges
submission['CO2'] = submission['CO2'].clip(0, 100)
submission['H2O'] = submission['H2O'].clip(0, 100)
submission['CH4'] = submission['CH4'].clip(0, 100)
submission['NH3'] = submission['NH3'].clip(0, 100)
submission['temperature'] = submission['temperature'].clip(100, 5000)
submission['radius'] = submission['radius'].clip(0.1, 10)

submission.to_csv('final_submission.csv', index=False)
```

### Final Hour
```bash
# 4. Upload to Kaggle
kaggle competitions submit \
  -c ariel-data-challenge-2025 \
  -f final_submission.csv \
  -m "Hybrid Quantum-NEBULA Model v1.0"
```

---

## 🔧 KEY IMPLEMENTATION DETAILS

### 1. Quantum Stage (ITensor)
- Encodes spectra as quantum states
- Extracts 128 quantum features via entanglement/coherence
- Captures molecular transition signatures

### 2. NEBULA Stage (CUDA)
- Processes quantum features through optical Fourier system
- Learnable amplitude/phase masks
- Outputs 6 atmospheric parameters

### 3. Training Strategy
- MSE loss for regression
- Adam optimizer with decay
- Early stopping on validation loss

### 4. Data Augmentation
```python
# Add noise to simulate measurement uncertainty
augmented = spectrum + np.random.normal(0, 0.01, spectrum.shape)

# Time averaging variations
for window in [150, 187, 200]:
    averaged = moving_average(spectrum, window)
```

---

## 📊 EXPECTED PERFORMANCE

Based on the hybrid approach:
- **Baseline CNN/Transformer**: MAE ~5-10%
- **Our Hybrid Model Target**: MAE ~3-7%

Why we might win:
1. Physical modeling of light (spectra ARE light)
2. Quantum features capture molecular physics
3. No overfitting (physics-constrained)

---

## 🚨 CRITICAL CHECKPOINTS

### After Day 1:
- [ ] Data preprocessed successfully
- [ ] Model compiles without errors  
- [ ] Can run 1 training epoch

### After Day 2:
- [ ] Training loss decreasing
- [ ] Validation loss stable
- [ ] Best hyperparameters identified

### After Day 3:
- [ ] Final model trained
- [ ] Submission validates locally
- [ ] Uploaded to Kaggle

---

## 🔥 EMERGENCY FALLBACKS

If quantum stage too slow:
```cpp
// Disable quantum, use NEBULA only
#define SKIP_QUANTUM 1
```

If CUDA unavailable:
```cpp
// CPU-only version
#define USE_CPU_ONLY 1
```

If time runs out:
```bash
# Use best checkpoint so far
cp run1/checkpoint_best final_model
./ariel_trainer --inference --model final_model
```

---

## 📝 SUBMISSION CHECKLIST

- [ ] submission.csv has correct format
- [ ] All planet_IDs present (0-N)
- [ ] Values in physical ranges
- [ ] File < 100MB
- [ ] No NaN or inf values

```python
# Final validation
def validate_submission(file):
    df = pd.read_csv(file)
    assert len(df) == len(test_data)
    assert df.columns.tolist() == ['planet_ID', 'CO2', 'H2O', 'CH4', 'NH3', 'temperature', 'radius']
    assert df.isnull().sum().sum() == 0
    assert (df['temperature'] > 0).all()
    print("✓ Submission valid!")
```

---

## 💪 LET'S WIN THIS!

The hybrid Quantum-NEBULA model is unique:
- **No one else** is using quantum physics for this
- **Physical accuracy** beats black-box models
- **Novel approach** = potential breakthrough

### Remember:
1. Focus on getting a valid submission first
2. Optimize performance second
3. Document everything for reproducibility

---

## FINAL COMMANDS TO RUN NOW:

```bash
# Start immediately:
cd smb://nebula-system-product-name.local/disco2/
mkdir hybrid_model && cd hybrid_model

# Copy all files from Claude
# Save these files: hybrid_ariel_model.hpp, ariel_data_loader.hpp, 
#                   ariel_trainer.cpp, Makefile_ariel

# Compile
make -f Makefile_ariel

# Train
./ariel_trainer --data ../ --epochs 100 --output ./output

# Generate submission
./ariel_trainer --inference --model ./output/checkpoint_best

# Upload
kaggle competitions submit -c ariel-data-challenge-2025 -f submission.csv -m "Hybrid"
```

**GO GO GO! 3 DAYS TO GLORY! 🚀**
