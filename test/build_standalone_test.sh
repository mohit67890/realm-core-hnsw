#!/bin/bash
# Build standalone HNSW test without full CMake configuration

echo "Building standalone HNSW vector search test..."

# Find the realm-core build directory
BUILD_DIR="../build"
SRC_DIR=".."

# Compiler settings
CXX="g++"
CXXFLAGS="-std=c++17 -I${SRC_DIR}/src -I${BUILD_DIR}/src"
LDFLAGS="-L${BUILD_DIR}/src -lrealm -lpthread"

# Try to compile
$CXX $CXXFLAGS test_hnsw_vector_search.cpp $LDFLAGS -o test_hnsw_vector_search 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo "Run with: ./test_hnsw_vector_search"
else
    echo "✗ Build failed. Trying alternative approach..."
    
    # Try with clang++ (common on macOS)
    CXX="clang++"
    echo "Trying with clang++..."
    $CXX $CXXFLAGS test_hnsw_vector_search.cpp $LDFLAGS -o test_hnsw_vector_search 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ Build successful with clang++!"
        echo "Run with: ./test_hnsw_vector_search"
    else
        echo "✗ Build failed. You may need to:"
        echo "  1. Build realm-core library first"
        echo "  2. Check compiler flags"
        echo "  3. Install development dependencies"
    fi
fi
