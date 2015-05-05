//
// Created by lene on 04.05.15.
//

#include "CostFunction.h"

#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/prod.hpp"
#include "MatrixPrinter.h"
#include "VectorPrinter.h"

template <typename ScalarType>
CostFunction<ScalarType>::CostFunction(
        const viennacl::matrix<ScalarType> &X,
        const viennacl::vector<ScalarType> &y): X_(X), y_(y) {
    assert(y_.size() == X_.size2());
}


/**
function J = computeCostMulti(X, y, theta)
  m = length(y); % number of training examples
  h_theta = X*theta;
  summand_linear = h_theta-y;
  summand_squared = summand_linear .^ 2;
  J = sum(summand_squared)/2/m;
 */
template <typename ScalarType>
viennacl::scalar<ScalarType> CostFunction<ScalarType>::operator()(const viennacl::vector<ScalarType> &theta) {
    assert(theta.size() == X_.size1());

    viennacl::vector<ScalarType> h_theta = viennacl::linalg::prod(trans(X_), theta);
    viennacl::vector<ScalarType> deviation = h_theta - y_;
    return viennacl::linalg::inner_prod(deviation, deviation);
}
