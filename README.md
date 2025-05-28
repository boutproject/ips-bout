# IPS-BOUT

Interface to [BOUT++](https://github.com/boutproject/BOUT-dev/) and
the [Hermes-3](https://github.com/boutproject/hermes-3/) code, pre-
and post-processing tools.

** Installation

Install Hermes-3 using one of the following methods:

1. Run the `cmake-build-hermes.sh` script. This will checkout the
   `hermes-3` submodule, configure and build it using cmake.  This
   assumes that CMake, Git, a C++ MPI compiler, FFTW and NetCDF C++
   bindings are installed and can be found by CMake.  If successful,
   the built executable will be `hermes-3/build/hermes-3`.

2. Follow instructions in the [Hermes-3
   manual](https://hermes3.readthedocs.io/en/latest/installation.html). To
   work with IPS you must configure hermes-3 to use ADIOS2 by adding
   `-DBOUT_DOWNLOAD_ADIOS2=ON -DBOUT_USE_ADIOS2=ON` configuration
   options.

