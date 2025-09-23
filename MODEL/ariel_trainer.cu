/**
 * MAIN TRAINING SCRIPT FOR ARIEL DATA CHALLENGE 2025
 * Complete end-to-end pipeline
 */

#include "hybrid_ariel_model.hpp"
#include "ariel_data_loader.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>

using namespace std;

// Training configuration
struct TrainingConfig {
    string data_path = "E:/NeurIPS_MYCELIUM_EVOLUTUM/ariel-data-challenge-2025";
    string output_path = "./outputs";
    int epochs = 100;
    int batch_size = 32;
    float learning_rate = 1e-3;
    float lr_decay = 0.95;
    int save_every = 10;
    bool use_cuda = true;
    int seed = 42;
};

class ArielTrainer {
private:
    HybridArielModel model;
    ArielDataLoader data_loader;
    TrainingConfig config;
    
    // Metrics tracking
    vector<float> train_losses;
    vector<float> val_losses;
    float best_val_loss = 1e9;
    int best_epoch = -1;
    
    // Timing
    chrono::high_resolution_clock::time_point start_time;
    
public:
    ArielTrainer(const TrainingConfig& cfg) 
        : config(cfg), data_loader(cfg.data_path) {
        
        // Set random seed
        srand(cfg.seed);
        
        // Create output directory
        system(("mkdir -p " + cfg.output_path).c_str());
        
        // Log configuration
        logConfig();
        
        start_time = chrono::high_resolution_clock::now();
    }
    
    void logConfig() {
        cout << "\n================================\n";
        cout << "ARIEL DATA CHALLENGE 2025\n";
        cout << "Hybrid Quantum-NEBULA Model\n";
        cout << "================================\n\n";
        cout << "Configuration:\n";
        cout << "  Data path: " << config.data_path << "\n";
        cout << "  Epochs: " << config.epochs << "\n";
        cout << "  Batch size: " << config.batch_size << "\n";
        cout << "  Learning rate: " << config.learning_rate << "\n";
        cout << "  LR decay: " << config.lr_decay << "\n";
        cout << "================================\n\n";
        
        // Save config to file
        ofstream cfg_file(config.output_path + "/config.txt");
        cfg_file << "Data: " << config.data_path << "\n";
        cfg_file << "Epochs: " << config.epochs << "\n";
        cfg_file << "Batch: " << config.batch_size << "\n";
        cfg_file << "LR: " << config.learning_rate << "\n";
        cfg_file.close();
    }
    
    void train() {
        cout << "Starting training...\n\n";
        
        for(int epoch = 1; epoch <= config.epochs; ++epoch) {
            // Train epoch
            float train_loss = trainEpoch(epoch);
            train_losses.push_back(train_loss);
            
            // Validation
            float val_loss = validate();
            val_losses.push_back(val_loss);
            
            // Learning rate decay
            if(epoch % 20 == 0) {
                config.learning_rate *= config.lr_decay;
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
            
            // Log progress
            logProgress(epoch, train_loss, val_loss);
        }
        
        cout << "\nTraining complete!\n";
        cout << "Best validation loss: " << best_val_loss 
             << " at epoch " << best_epoch << "\n";
        
        // Generate final submission
        generateSubmission();
    }
    
    float trainEpoch(int epoch) {
        auto train_iter = data_loader.getTrainIterator(config.batch_size);
        
        float total_loss = 0.0;
        int batch_count = 0;
        
        while(train_iter.hasNext()) {
            auto [batch_x, batch_y] = train_iter.next();
            
            // Forward + backward + update
            float batch_loss = model.trainBatch(batch_x, batch_y);
            
            total_loss += batch_loss;
            batch_count++;
            
            // Progress bar
            if(batch_count % 10 == 0) {
                cout << "\r[Epoch " << epoch << "] Batch " << batch_count 
                     << " | Loss: " << batch_loss << "      " << flush;
            }
        }
        
        cout << "\r";  // Clear line
        return total_loss / batch_count;
    }
    
    float validate() {
        auto val_iter = data_loader.getValIterator(config.batch_size);
        
        float total_loss = 0.0;
        int batch_count = 0;
        
        // Metrics for each target
        vector<float> mae_per_target(6, 0.0);
        
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
            }
            
            total_loss += batch_loss / batch_x.size();
            batch_count++;
        }
        
        // Print per-target metrics
        cout << "  Validation MAE: ";
        vector<string> target_names = {"CO2", "H2O", "CH4", "NH3", "Temp", "Radius"};
        for(int i = 0; i < 6; ++i) {
            cout << target_names[i] << "=" 
                 << fixed << setprecision(3) 
                 << mae_per_target[i] / (batch_count * config.batch_size) << " ";
        }
        cout << "\n";
        
        return total_loss / batch_count;
    }
    
    void logProgress(int epoch, float train_loss, float val_loss) {
        auto now = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(now - start_time).count();
        
        cout << "[Epoch " << setw(3) << epoch << "/" << config.epochs << "] "
             << "Train: " << fixed << setprecision(4) << train_loss << " | "
             << "Val: " << fixed << setprecision(4) << val_loss << " | "
             << "Time: " << duration << "s | "
             << "LR: " << scientific << config.learning_rate;
        
        if(val_loss == best_val_loss) {
            cout << " *BEST*";
        }
        cout << "\n";
        
        // Save metrics
        ofstream metrics(config.output_path + "/metrics.csv", ios::app);
        metrics << epoch << "," << train_loss << "," << val_loss << "\n";
        metrics.close();
    }
    
    void saveCheckpoint(const string& name) {
        string path = config.output_path + "/checkpoint_" + name;
        model.saveCheckpoint(path);
    }
    
    void generateSubmission() {
        cout << "\nGenerating submission...\n";
        
        // Load best checkpoint
        // model.loadCheckpoint(config.output_path + "/checkpoint_best");
        
        // Generate predictions for test set
        string submission_path = config.output_path + "/submission.csv";
        model.generateSubmission(config.data_path, submission_path);
        
        cout << "Submission saved to: " << submission_path << "\n";
        
        // Also create a Kaggle-ready version
        createKaggleSubmission(submission_path);
    }
    
    void createKaggleSubmission(const string& submission_path) {
        // Format for Kaggle upload
        ifstream in(submission_path);
        ofstream out(config.output_path + "/kaggle_submission.csv");
        
        string line;
        while(getline(in, line)) {
            out << line << "\n";
        }
        
        in.close();
        out.close();
        
        cout << "Kaggle submission ready: " 
             << config.output_path << "/kaggle_submission.csv\n";
    }
    
    // Plot training curves (optional, requires gnuplot)
    void plotMetrics() {
        ofstream plot_script(config.output_path + "/plot.gnuplot");
        plot_script << "set terminal png size 1200,600\n";
        plot_script << "set output '" << config.output_path << "/training_curves.png'\n";
        plot_script << "set multiplot layout 1,2\n";
        plot_script << "set title 'Training Loss'\n";
        plot_script << "plot '" << config.output_path << "/metrics.csv' using 1:2 with lines title 'Train'\n";
        plot_script << "set title 'Validation Loss'\n";
        plot_script << "plot '" << config.output_path << "/metrics.csv' using 1:3 with lines title 'Val'\n";
        plot_script << "unset multiplot\n";
        plot_script.close();
        
        system(("gnuplot " + config.output_path + "/plot.gnuplot").c_str());
    }
};

// ==================== MAIN ====================
int main(int argc, char** argv) {
    try {
        TrainingConfig config;
        
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
        
        // Initialize trainer
        ArielTrainer trainer(config);
        
        // Run training
        trainer.train();
        
        // Plot results
        trainer.plotMetrics();
        
        cout << "\n=== COMPLETE ===\n";
        cout << "Submission ready for Kaggle upload!\n";
        cout << "Files in: " << config.output_path << "/\n";
        
    } catch(const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
