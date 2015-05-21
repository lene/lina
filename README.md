# lina
Linear Algebra library in C++ and OpenCL for machine learning algorithms.
GPU-accelerated routines for multidimensional optimization, linear regression,
logistic regression. Later: neural networks. Even later: SVM and recommender systems. 

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

* improve RegressionSolver
  * make the kind of regression a class template parameter
  * add predict() function
  * add function to determine training accuracy
* use smart pointers again (in gradient descent)
* logistic regression
  * minimization - is there a better way than gradient descent? does GD always give so bad results in nontrivial systems?
  * training accuracy
  * regularization
    * coursera example
  * multi-class classification
* ensure that only one matrix is stored in GPU memory at each time when using LinearRegressionSolver
* gradient descent seems to run on one CPU core only, at least with logistic regression
* factor out matrix and vector types so they can be used as template parameters
  * easier conversion between ublas and viennacl data types and algorithms?
* neural networks
* compile conditionally on presence of gtest so that it can be distributed without it

