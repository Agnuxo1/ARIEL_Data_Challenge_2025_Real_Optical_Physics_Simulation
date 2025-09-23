/**
 * FIXED ARIEL TRAINER WITH MULTI-GPU SUPPORT
 * Resolves convergence issues and utiliza las 3 GPUs
 */

#include "fixed_hybrid_model.hpp"
#include "ariel_data_loader_lazy.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>

using namespace std;

struct FixedTrainingConfig {
    string data_path = "/mnt/disco2/calibrated";
    string output_path = "./outputs_fixed";
    int epochs = 1000;
    int batch_size = 8;  // Increased for multi-GPU
    float learning_rate = 0.01f;  // Increased LR
    float lr_decay = 0.95f;
    int save_every = 50;
    int seed = 42;
};

class FixedArielTrainer {
private:
    FixedHybridArielModel model;
    ArielLazyDataLoader data_loader;
    FixedTrainingConfig config;

    vector<float> train_losses;
    vector<float> val_losses;
    float best_val_loss = 1e9;
    int best_epoch = -1;

    chrono::high_resolution_clock::time_point start_time;

public:
    FixedArielTrainer(const FixedTrainingConfig& cfg)
        : config(cfg), data_loader(cfg.data_path) {

        srand(cfg.seed);
        system(("mkdir -p " + cfg.output_path).c_str());

        // CRITICAL: Load real data into model
        cout << "[TRAINER] Loading real calibrated data..." << endl;
        model.loadData(cfg.data_path);

        logConfig();
        start_time = chrono::high_resolution_clock::now();
    }

    void logConfig() {
        cout << "\n================================\n";
        cout << "FIXED ARIEL DATA CHALLENGE 2025\n";
        cout << "Multi-GPU Quantum-NEBULA Model\n";
        cout << "================================\n\n";
        cout << "Configuration:\n";
        cout << "  Data path: " << config.data_path << "\n";
        cout << "  Epochs: " << config.epochs << "\n";
        cout << "  Batch size: " << config.batch_size << "\n";
        cout << "  Learning rate: " << config.learning_rate << "\n";
        cout << "  LR decay: " << config.lr_decay << "\n";
        cout << "  Multi-GPU: 3x RTX 3080\n";
        cout << "================================\n\n";

        ofstream cfg_file(config.output_path + "/config.txt");
        cfg_file << "Fixed Model Configuration\n";
        cfg_file << "Data: " << config.data_path << "\n";
        cfg_file << "Epochs: " << config.epochs << "\n";
        cfg_file << "Batch: " << config.batch_size << "\n";
        cfg_file << "LR: " << config.learning_rate << "\n";
        cfg_file << "Multi-GPU: Enabled\n";
        cfg_file.close();
    }

    void train() {
        cout << "Starting FIXED training with real data...\n\n";

        for(int epoch = 1; epoch <= config.epochs; ++epoch) {
            float train_loss = trainEpoch(epoch);
            train_losses.push_back(train_loss);

            float val_loss = validate();
            val_losses.push_back(val_loss);

            // Learning rate decay
            if(epoch % 50 == 0) {
                config.learning_rate *= config.lr_decay;
                cout << "[LR-DECAY] New learning rate: " << config.learning_rate << endl;
            }

            // Save best model
            if(val_loss < best_val_loss) {
                best_val_loss = val_loss;
                best_epoch = epoch;
                saveCheckpoint("best");
            }

            // Regular checkpoint
            if(epoch % config.save_every == 0) {
                saveCheckpoint("epoch_" + to_string(epoch));
            }

            logProgress(epoch, train_loss, val_loss);

            // Early convergence check
            if(train_loss < 1000.0 && val_loss < 1000.0) {
                cout << "\n[CONVERGENCE] Model converged early at epoch " << epoch << endl;
                break;
            }
        }

        cout << "\nTraining complete!\n";
        cout << "Best validation loss: " << best_val_loss
             << " at epoch " << best_epoch << "\n";

        generateSubmission();
    }

    float trainEpoch(int epoch) {
        auto train_iter = data_loader.getTrainIterator(config.batch_size);

        float total_loss = 0.0;
        int batch_count = 0;

        while(train_iter.hasNext()) {
            auto [batch_x, batch_y] = train_iter.next();

            // Multi-GPU training with real data
            float batch_loss = model.trainBatch(batch_x, batch_y);

            total_loss += batch_loss;
            batch_count++;

            if(batch_count % 5 == 0) {
                cout << "\r[Epoch " << epoch << "] Batch " << batch_count
                     << " | Loss: " << scientific << setprecision(3) << batch_loss
                     << "      " << flush;
            }
        }

        cout << "\r";
        return total_loss / max(batch_count, 1);
    }

    float validate() {
        auto val_iter = data_loader.getValIterator(config.batch_size);

        float total_loss = 0.0;
        int batch_count = 0;
        vector<float> mae_per_target(6, 0.0);
        int total_samples = 0;

        while(val_iter.hasNext()) {
            auto [batch_x, batch_y] = val_iter.next();

            float batch_loss = 0.0;

            for(size_t i = 0; i < batch_x.size(); ++i) {
                auto pred = model.forward(batch_x[i]);

                for(int j = 0; j < 6; ++j) {
                    float diff = pred[j] - batch_y[i][j];
                    batch_loss += diff * diff;
                    mae_per_target[j] += abs(diff);
                }
                total_samples++;
            }

            total_loss += batch_loss / batch_x.size();
            batch_count++;
        }

        // Print per-target validation metrics
        cout << "  Multi-GPU Val MAE: ";
        vector<string> target_names = {"CO2", "H2O", "CH4", "NH3", "Temp", "Radius"};
        for(int i = 0; i < 6; ++i) {
            cout << target_names[i] << "="
                 << fixed << setprecision(3)
                 << mae_per_target[i] / max(total_samples, 1) << " ";
        }
        cout << "\n";

        return total_loss / max(batch_count, 1);
    }

    void logProgress(int epoch, float train_loss, float val_loss) {
        auto now = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(now - start_time).count();

        cout << "[Epoch " << setw(3) << epoch << "/" << config.epochs << "] "
             << "Train: " << scientific << setprecision(4) << train_loss << " | "
             << "Val: " << scientific << setprecision(4) << val_loss << " | "
             << "Time: " << duration << "s | "
             << "LR: " << scientific << config.learning_rate;

        if(val_loss == best_val_loss) {
            cout << " *BEST*";
        }
        cout << "\n";

        // Save metrics with proper formatting
        ofstream metrics(config.output_path + "/metrics.csv", ios::app);
        if(epoch == 1) {
            metrics << "epoch,train_loss,val_loss,time,lr\n";
        }
        metrics << epoch << "," << train_loss << "," << val_loss
                << "," << duration << "," << config.learning_rate << "\n";
        metrics.close();
    }

    void saveCheckpoint(const string& name) {
        string path = config.output_path + "/checkpoint_" + name;
        model.saveCheckpoint(path);  // CRITICAL: Now implemented for Kaggle!
        cout << "[CHECKPOINT] Saved: " << name << endl;
    }

    void generateSubmission() {
        cout << "\nGenerating submission with multi-GPU model...\n";

        string submission_path = config.output_path + "/submission.csv";
        model.generateSubmission(config.data_path, submission_path);

        cout << "Submission saved to: " << submission_path << "\n";

        // Copy for Kaggle upload
        string kaggle_path = config.output_path + "/kaggle_submission.csv";
        system(("cp " + submission_path + " " + kaggle_path).c_str());

        cout << "Kaggle submission ready: " << kaggle_path << "\n";
    }
};

// ==================== MAIN ====================
int main(int argc, char** argv) {
    try {
        FixedTrainingConfig config;

        // Parse command line arguments
        for(int i = 1; i < argc; i += 2) {
            string arg = argv[i];
            if(arg == "--data" && i+1 < argc) {
                config.data_path = argv[i+1];
            }
            else if(arg == "--epochs" && i+1 < argc) {
                config.epochs = stoi(argv[i+1]);
            }
            else if(arg == "--batch" && i+1 < argc) {
                config.batch_size = stoi(argv[i+1]);
            }
            else if(arg == "--lr" && i+1 < argc) {
                config.learning_rate = stof(argv[i+1]);
            }
            else if(arg == "--output" && i+1 < argc) {
                config.output_path = argv[i+1];
            }
        }

        cout << "[SYSTEM] Starting FIXED ARIEL trainer with multi-GPU support\n";

        // Initialize trainer with fixed model
        FixedArielTrainer trainer(config);

        // Run corrected training
        trainer.train();

        cout << "\n=== FIXED TRAINING COMPLETE ===\n";
        cout << "Multi-GPU submission ready for Kaggle!\n";
        cout << "Files in: " << config.output_path << "/\n";

    } catch(const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}