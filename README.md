# lina
Linear Algebra library in C++ and OpenCL for machine learning algorithms.
GPU-accelerated routines for multidimensional optimization, linear regression,
later: logistic regression and neural networks. Even later: SVM and recommender systems. 

## Dependencies:

### Boost uBLAS

    sudo apt-get install libboost-dev
    
### ViennaCl

    sudo apt-get install libviennacl-dev
    
### OpenCl headers and drivers

YMMV, packages to install depend on present GPU:

    sudo apt-get install ocl-icd-libopencl1 ocl-icd-opencl-dev opencl-headers
    
### Google Test

    sudo apt-get install libgtest-dev
    cd /usr/src/gtest
    sudo cmake CMakeLists.txt
    sudo make
    sudo cp *.a /usr/lib

## To do

* ensure that only one matrix is stored in GPU memory at each time when using LinearRegressionSolver
* factor out matrix and vector types so they can be used as template parameters
  * easier conversion between ublas and viennacl data types and algorithms?
* logistic regression
  * multi-class classification
* regularization
* neural networks
* compile conditionally on presence of gtest

