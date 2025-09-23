#!/usr/bin/env python3
"""
Compare Model Performance: Base vs Fine-tuned
Evaluate both models to determine which gives better results
"""

import pandas as pd
import numpy as np
import os
import json

def analyze_metrics(metrics_file, model_name):
    """Analyze training metrics from CSV file"""
    if not os.path.exists(metrics_file):
        return None

    df = pd.read_csv(metrics_file, header=None, names=['epoch', 'train_loss', 'val_loss'])

    stats = {
        'model': model_name,
        'total_epochs': len(df),
        'final_train_loss': df['train_loss'].iloc[-1],
        'final_val_loss': df['val_loss'].iloc[-1],
        'best_val_loss': df['val_loss'].min(),
        'best_val_epoch': df.loc[df['val_loss'].idxmin(), 'epoch'],
        'convergence_stability': df['val_loss'].tail(10).std(),  # Stability in last 10 epochs
        'improvement': df['val_loss'].iloc[0] - df['val_loss'].iloc[-1]
    }

    return stats

def compare_models():
    """Compare base model vs fine-tuned model"""

    print("=" * 60)
    print("ARIEL HYBRID QUANTUM-NEBULA MODEL COMPARISON")
    print("=" * 60)

    # Model paths
    base_model_metrics = "./outputs_FINAL_CONVERGENCE_FIXED/metrics.csv"
    finetuned_model_metrics = "./outputs_FINAL_FINETUNED/metrics.csv"

    # Analyze both models
    base_stats = analyze_metrics(base_model_metrics, "Base Model (FINAL_CONVERGENCE_FIXED)")
    finetuned_stats = analyze_metrics(finetuned_model_metrics, "Fine-tuned Model (FINAL_FINETUNED)")

    if base_stats is None:
        print("❌ Base model metrics not found!")
        return

    if finetuned_stats is None:
        print("⚠️  Fine-tuned model still training or metrics not available")
        print("Base model analysis only:")

    # Display comparison
    print(f"\n📊 BASE MODEL ANALYSIS:")
    print(f"   Total epochs: {base_stats['total_epochs']}")
    print(f"   Final train loss: {base_stats['final_train_loss']:.1f}")
    print(f"   Final val loss: {base_stats['final_val_loss']:.1f}")
    print(f"   Best val loss: {base_stats['best_val_loss']:.1f} (epoch {base_stats['best_val_epoch']})")
    print(f"   Convergence stability: {base_stats['convergence_stability']:.2f}")
    print(f"   Total improvement: {base_stats['improvement']:.1f}")

    if finetuned_stats:
        print(f"\n🎯 FINE-TUNED MODEL ANALYSIS:")
        print(f"   Total epochs: {finetuned_stats['total_epochs']}")
        print(f"   Final train loss: {finetuned_stats['final_train_loss']:.1f}")
        print(f"   Final val loss: {finetuned_stats['final_val_loss']:.1f}")
        print(f"   Best val loss: {finetuned_stats['best_val_loss']:.1f} (epoch {finetuned_stats['best_val_epoch']})")
        print(f"   Convergence stability: {finetuned_stats['convergence_stability']:.2f}")
        print(f"   Total improvement: {finetuned_stats['improvement']:.1f}")

        # Determine winner
        print(f"\n🏆 COMPARISON RESULTS:")

        if finetuned_stats['best_val_loss'] < base_stats['best_val_loss']:
            winner = "Fine-tuned Model"
            improvement = base_stats['best_val_loss'] - finetuned_stats['best_val_loss']
            print(f"   🥇 WINNER: {winner}")
            print(f"   📈 Improvement: {improvement:.1f} validation loss reduction")
        else:
            winner = "Base Model"
            print(f"   🥇 WINNER: {winner}")
            print(f"   📉 Fine-tuning did not improve validation loss")

        # Stability comparison
        if finetuned_stats['convergence_stability'] < base_stats['convergence_stability']:
            print(f"   ✅ Fine-tuned model is more stable")
        else:
            print(f"   ⚠️  Base model is more stable")

        # Save comparison results
        comparison = {
            'base_model': base_stats,
            'finetuned_model': finetuned_stats,
            'winner': winner,
            'recommendation': f"Use {winner} for best results"
        }

        with open("model_comparison_results.json", "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"\n💾 Results saved to: model_comparison_results.json")

        # Checkpoint recommendations
        if winner == "Fine-tuned Model":
            best_checkpoint = f"./outputs_FINAL_FINETUNED/checkpoint_best"
            print(f"\n🎯 RECOMMENDED CHECKPOINT: {best_checkpoint}")
        else:
            best_checkpoint = f"./outputs_FINAL_CONVERGENCE_FIXED/checkpoint_best"
            print(f"\n🎯 RECOMMENDED CHECKPOINT: {best_checkpoint}")

        return best_checkpoint

    else:
        print(f"\n💡 Recommendation: Continue using base model until fine-tuning completes")
        return "./outputs_FINAL_CONVERGENCE_FIXED/checkpoint_best"

def check_training_status():
    """Check if fine-tuning is still running"""
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'ariel_trainer.*FINAL_FINETUNED'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("🔄 Fine-tuning still in progress...")
            return True
        else:
            print("✅ Fine-tuning completed!")
            return False
    except:
        return False

if __name__ == "__main__":
    print("Checking training status...")
    still_training = check_training_status()

    if still_training:
        print("Performing partial comparison (base model + progress check)...")

    best_checkpoint = compare_models()

    if still_training:
        print(f"\n⏳ Run this script again when fine-tuning completes for full comparison")
    else:
        print(f"\n🚀 Use checkpoint: {best_checkpoint}")
        print(f"🎯 Ready for Kaggle submission with optimal model!")