/**
 * ARIEL KAGGLE STANDALONE SUBMISSION
 * Loads checkpoint and generates CSV immediately
 * Optimized for Kaggle environment - no training dependencies
 */

#include "hybrid_ariel_model.hpp"
#ifndef _WIN32
// Linux/Unix headers
#include <dirent.h>
#include <sys/stat.h>
#include <cstring>
#else
// Windows headers
#include <filesystem>
#endif
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>

using namespace std;

// Simplified config for Kaggle
struct KaggleConfig {
    string data_path = "./";  // Current directory in Kaggle
    string output_path = "./";
    string checkpoint_name = "checkpoint_best";  // Look for best checkpoint
    bool force_csv_generation = true;  // Always generate CSV
};

class KaggleArielRunner {
private:
    HybridArielModel model;
    KaggleConfig config;

public:
    KaggleArielRunner(const KaggleConfig& cfg) : config(cfg) {
        cout << "\n====================================\n";
        cout << "ARIEL KAGGLE SUBMISSION RUNNER\n";
        cout << "====================================\n";
        cout << "Data path: " << config.data_path << "\n";
        cout << "Looking for checkpoint: " << config.checkpoint_name << "\n";
        cout << "====================================\n\n";
    }

    string findBestCheckpoint() {
        vector<string> candidates = {
            config.data_path + "/checkpoint_best",
            "/kaggle/input/ariel-model/checkpoint_best",  // Kaggle dataset path
            config.data_path + "/checkpoint_epoch_1000",
            config.data_path + "/checkpoint_epoch_900",
            config.data_path + "/checkpoint_epoch_800",
            config.data_path + "/checkpoint_epoch_700",
            config.data_path + "/checkpoint_epoch_600",
            config.data_path + "/checkpoint_epoch_500"
        };

        for (const auto& candidate : candidates) {
            ifstream test(candidate);
            if (test.good()) {
                test.close();
                cout << "[CHECKPOINT] Found: " << candidate << endl;
                return candidate;
            }
        }

        cout << "[CHECKPOINT] No checkpoints found - using default model" << endl;
        return "";
    }

    void generateSubmissionCSV() {
        cout << "\n=== KAGGLE CSV GENERATION ===\n";

        // Try to load checkpoint
        string checkpoint_path = findBestCheckpoint();
        if (!checkpoint_path.empty()) {
            bool loaded = model.loadCheckpoint(checkpoint_path);
            if (loaded) {
                cout << "[KAGGLE] Checkpoint loaded successfully!" << endl;
            } else {
                cout << "[KAGGLE] Failed to load checkpoint, using default model" << endl;
            }
        }

        // Generate CSV
        string csv_path = config.output_path + "/submission.csv";
        cout << "[KAGGLE] Generating submission: " << csv_path << endl;

        auto start_time = chrono::high_resolution_clock::now();
        model.generateSubmission(config.data_path, csv_path);
        auto end_time = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time).count();
        cout << "[KAGGLE] CSV generation completed in " << duration << " seconds" << endl;
        cout << "[KAGGLE] Ready for submission: " << csv_path << endl;

        // Verify file was created
        ifstream verify(csv_path);
        if (verify.good()) {
            // Count lines
            string line;
            int line_count = 0;
            while (getline(verify, line)) {
                line_count++;
            }
            verify.close();
            cout << "[KAGGLE] Verification: " << line_count << " lines in CSV" << endl;
        } else {
            cout << "[KAGGLE] ERROR: CSV file was not created!" << endl;
        }

        cout << "\n=== KAGGLE SUBMISSION COMPLETE ===\n";
    }
};

int main(int argc, char** argv) {
    try {
        cout << "ARIEL Kaggle Standalone Runner v1.0\n";
        cout << "Automated checkpoint loading and CSV generation\n\n";

        KaggleConfig config;

        // Parse command line arguments
        for(int i = 1; i < argc; i += 2) {
            string arg = argv[i];
            if(arg == "--data" && i+1 < argc) {
                config.data_path = argv[i+1];
            }
            else if(arg == "--output" && i+1 < argc) {
                config.output_path = argv[i+1];
            }
            else if(arg == "--checkpoint" && i+1 < argc) {
                config.checkpoint_name = argv[i+1];
            }
        }

        // Create runner and generate CSV
        KaggleArielRunner runner(config);
        runner.generateSubmissionCSV();

        cout << "\nSuccess! Submission file ready for Kaggle upload.\n";
        return 0;

    } catch(const exception& e) {
        cerr << "KAGGLE ERROR: " << e.what() << "\n";
        return 1;
    }
}