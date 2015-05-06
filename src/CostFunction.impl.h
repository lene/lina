//
// Created by lene on 04.05.15.
//

#include "CostFunction.h"

#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/prod.hpp"
#include "MatrixPrinter.h"
#include "VectorPrinter.h"

/**
 *  \param X features
 *  \param y training examples (y_i = f(X_i))
 */
template <typename ScalarType>
CostFunction<ScalarType>::CostFunction(
        const viennacl::matrix<ScalarType> &X,
        const viennacl::vector<ScalarType> &y): X_(X), y_(y) {
    assert(y_.size() == X_.size2());
}

/**
 *  hypothesis \f$h_\theta(X)\f$
 *  \param theta
 */
template <typename ScalarType>
viennacl::vector<ScalarType>
CostFunction<ScalarType>::h_theta(const viennacl::vector<ScalarType> &theta) const {
    assert(theta.size() == X_.size1());
    return viennacl::linalg::prod(trans(X_), theta);
}

/**
 *  How far hypothesis \f$h_\theta(X)\f$ misses training examples \f$y\f$
 *  \param theta
 */
template <typename ScalarType>
viennacl::vector<ScalarType>
CostFunction<ScalarType>::deviation(const viennacl::vector<ScalarType> &theta) const {
    return h_theta(theta) - y_;
}

/**
 *  cost function for given \f$\theta\f$
 *  \param theta
 */
template <typename ScalarType>
viennacl::scalar<ScalarType>
CostFunction<ScalarType>::operator()(const viennacl::vector<ScalarType> &theta) const {
    viennacl::vector<ScalarType> d = deviation(theta);
    return viennacl::linalg::inner_prod(d, d) / ScalarType(2*y_.size());
}

/**
 *  gradient of cost function for given \f$\theta\f$
 *  \param theta
 *  \todo the arguments to prod() may be wrong; in Matlab it is actually (X*theta-y)' * X
 */
template <typename ScalarType>
viennacl::vector<ScalarType>
CostFunction<ScalarType>::gradient(const viennacl::vector<ScalarType> &theta) const {
    return viennacl::linalg::prod(X_, deviation(theta)) / ScalarType(y_.size());
}