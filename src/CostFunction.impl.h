//
// Created by lene on 04.05.15.
//

#ifndef LINA_COSTFUNCTION_IMPL_H
#define LINA_COSTFUNCTION_IMPL_H

#include "CostFunction.h"
#include "MatrixPrinter.h"
#include "VectorPrinter.h"

#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/matrix_operations.hpp"

/**
 *  \param X features
 *  \param y training examples (y_i = f(X_i))
 */
template <typename Scalar>
CostFunction<Scalar>::CostFunction(
        const viennacl::matrix<Scalar> &X,
        const viennacl::vector<Scalar> &y): X_(X), y_(y) {
    assert(y_.size() == X_.size1());
}

/**
 *  hypothesis \f$h_\theta(X)\f$
 *  \param theta

 */
template <typename Scalar>
viennacl::vector<Scalar>
CostFunction<Scalar>::h_theta(const viennacl::vector<Scalar> &theta) const {
#   if 0
        std::cout << X_ << theta << std::endl;
#   endif
    assert(theta.size() == X_.size2());
    return viennacl::linalg::prod(X_, theta);
}

/**
 *  How far hypothesis \f$h_\theta(X)\f$ misses training examples \f$y\f$
 *  \param theta
 */
template <typename Scalar>
viennacl::vector<Scalar>
CostFunction<Scalar>::deviation(const viennacl::vector<Scalar> &theta) const {
    return h_theta(theta) - y_;
}

/**
 *  cost function for given \f$\theta\f$
 *  \param theta
 */
template <typename Scalar>
viennacl::scalar<Scalar>
CostFunction<Scalar>::operator()(const viennacl::vector<Scalar> &theta) const {
    viennacl::vector<Scalar> d = deviation(theta);
    return viennacl::linalg::inner_prod(d, d) / Scalar(2*y_.size());
}

/**
 *  gradient of cost function for given \f$\theta\f$
 *  \param theta
 *  \todo the arguments to prod() may be wrong; in Matlab it is (X*theta-y)' * X, here X' * (X*theta-y)
 */
template <typename Scalar>
viennacl::vector<Scalar>
CostFunction<Scalar>::gradient(const viennacl::vector<Scalar> &theta) const {
    return viennacl::linalg::prod(trans(X_), deviation(theta)) / Scalar(y_.size());
}

#endif