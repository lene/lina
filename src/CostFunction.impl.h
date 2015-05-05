//
// Created by lene on 04.05.15.
//

#include "CostFunction.h"

#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/prod.hpp"
#include "main.cpp"

template <typename ScalarType>
CostFunction<ScalarType>::CostFunction(
        const viennacl::matrix<ScalarType> &X,
        const viennacl::vector<ScalarType> &y): X_(X), y_(y) { }

template <typename ScalarType>
viennacl::vector<ScalarType>
CostFunction<ScalarType>::h_theta(const viennacl::vector<ScalarType> &theta) const {
    assert(theta.size() == X_.size1());
    return viennacl::linalg::prod(trans(X_), theta);
}

template <typename ScalarType>
viennacl::vector<ScalarType>
CostFunction<ScalarType>::deviation(const viennacl::vector<ScalarType> &theta) const {
    return h_theta(theta) - y_;
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
viennacl::scalar<ScalarType>
CostFunction<ScalarType>::operator()(const viennacl::vector<ScalarType> &theta) const {
    viennacl::vector<ScalarType> d = deviation(theta);
    return viennacl::linalg::inner_prod(d, d) / ScalarType(2*y_.size());
}

template <typename ScalarType>
viennacl::vector<ScalarType>
CostFunction<ScalarType>::gradient(const viennacl::vector<ScalarType> &theta) const {
    return viennacl::linalg::prod(X_, deviation(theta)) / ScalarType(y_.size());
}