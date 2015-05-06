# lina
Linear Algebra library in C++ and OpenCL for machine learning algorithms

## Dependencies:

### Boost uBLAS
<installation instructions here>
### ViennaCl
    sudo apt-get install libviennacl-dev
### OpenCl headers and drivers
<installation instructions here>
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
** easier conversion between ublas and viennacl data types and algorithms?
* instantiate template classes for reasonable template parameters so the classes can be linked instead of #included
* logistic regression
** multi-class classification
* regularization
* neural networks
