#!/bin/bash
# Build script for Kaggle Linux environment

echo "========================================"
echo "ARIEL MODEL - KAGGLE LINUX BUILD"
echo "========================================"

# Install dependencies
echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y g++ libeigen3-dev libopenblas-dev zlib1g-dev

# Create cnpy stub (simplified numpy loader)
echo "Creating cnpy stub for Linux..."
cat > cnpy.h << 'EOF'
#ifndef CNPY_H
#define CNPY_H
#include <vector>
#include <string>
#include <fstream>
#include <complex>

namespace cnpy {
    struct NpyArray {
        std::vector<size_t> shape;
        size_t word_size;
        bool fortran_order;
        char* data_ptr;

        template<typename T> T* data() { return reinterpret_cast<T*>(data_ptr); }

        ~NpyArray() { if(data_ptr) delete[] data_ptr; }
    };

    NpyArray npy_load(const std::string& fname);
}
#endif
EOF

cat > cnpy.cpp << 'EOF'
#include "cnpy.h"
#include <iostream>

cnpy::NpyArray cnpy::npy_load(const std::string& fname) {
    NpyArray arr;
    std::ifstream file(fname, std::ios::binary);

    if (!file) {
        throw std::runtime_error("npy_load: Unable to open file " + fname);
    }

    // Simplified NPY loader - read header
    char magic[6];
    file.read(magic, 6);

    if (magic[0] != '\x93' || memcmp(magic+1, "NUMPY", 5) != 0) {
        throw std::runtime_error("Invalid NPY file format");
    }

    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    std::string header(header_len, ' ');
    file.read(&header[0], header_len);

    // Parse basic shape info (simplified)
    size_t pos = header.find("'shape': (");
    if (pos != std::string::npos) {
        pos += 10; // "'shape': ("
        while (pos < header.length() && header[pos] != ')') {
            if (std::isdigit(header[pos])) {
                size_t num = 0;
                while (pos < header.length() && std::isdigit(header[pos])) {
                    num = num * 10 + (header[pos] - '0');
                    pos++;
                }
                arr.shape.push_back(num);
            }
            pos++;
        }
    }

    // Calculate data size
    size_t total_elements = 1;
    for (size_t dim : arr.shape) total_elements *= dim;

    arr.word_size = 4; // float32
    size_t data_size = total_elements * arr.word_size;

    // Read data
    arr.data_ptr = new char[data_size];
    file.read(arr.data_ptr, data_size);

    return arr;
}
EOF

# Compile
echo "Compiling ARIEL model for Linux..."
g++ -std=c++17 -O3 -DNDEBUG -I. -I/usr/include/eigen3 \
    ariel_kaggle_standalone.cpp cnpy.cpp \
    -o ariel_kaggle_standalone \
    -lopenblas -lz

if [ $? -eq 0 ]; then
    echo "========================================"
    echo "BUILD SUCCESSFUL!"
    echo "Executable: ./ariel_kaggle_standalone"
    echo "========================================"

    # Test execution
    echo "Testing executable..."
    ./ariel_kaggle_standalone --help || echo "Ready for execution"
else
    echo "BUILD FAILED!"
    exit 1
fi