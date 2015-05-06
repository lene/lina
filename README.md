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
* find out why running gradient descent on the feature normalized matrix still yields the unscaled values
* find out why result differs (only slightly) from the results obtained with octave
* test suite for gradient descent
* factor out the bookkeeping in main.cpp into an easily usable class
* factor out setting up matrices in test/
* overhaul matrix and vector printer
* factor out matrix and vector types so they can be used as template parameters
  * easier conversion between ublas and viennacl data types and algorithms?
* instantiate template classes for reasonable template parameters so the classes can be linked instead of #included
* logistic regression
  * multi-class classification
* regularization
* neural networks
