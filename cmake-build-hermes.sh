#!/usr/bin/env bash
#
# Requirements:
# - git
# - cmake
# - fftw
# - NetCDF, NetCDF-C++

# Fetch Hermes-3
git submodule update --init
cd hermes-3

# Configure. NetCDF is needed for grid files; ADIOS2 for restarts and data output
# Note: CMake minimum version 3.5 needed for NetCDF
cmake . -B build \
        -DBOUT_DOWNLOAD_SUNDIALS=ON \
        -DBOUT_DOWNLOAD_ADIOS2=ON -DBOUT_USE_ADIOS2=ON  \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DBOUT_ENABLE_PYTHON=OFF \
        -DBOUT_ENABLE_BACKTRACE=OFF \
        -DBOUT_USE_COLOR=OFF -DBOUT_ENABLE_TRACK=OFF

cd build

make -j
