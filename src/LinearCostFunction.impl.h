//
// Created by lene on 04.05.15.
//

#ifndef LINA_COSTFUNCTION_IMPL_H
#define LINA_COSTFUNCTION_IMPL_H

#include "LinearCostFunction.h"
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
LinearCostFunction<Scalar>::LinearCostFunction(
        const viennacl::matrix<Scalar> &X,
        const viennacl::vector<Scalar> &y): CostFunction<Scalar>(X, y) { }

/**
 *  hypothesis \f$h_\theta(X)\f$
 *  \param theta

 */
template <typename Scalar>
viennacl::vector<Scalar>
LinearCostFunction<Scalar>::h_theta(const viennacl::vector<Scalar> &theta) const {
#   if 0
        std::cout << X_ << theta << std::endl;
#   endif
    assert(theta.size() == CostFunction<Scalar>::X_.size2());
    return viennacl::linalg::prod(CostFunction<Scalar>::X_, theta);
}

/**
 *  How far hypothesis \f$h_\theta(X)\f$ misses training examples \f$y\f$
 *  \param theta
 */
template <typename Scalar>
viennacl::vector<Scalar>
LinearCostFunction<Scalar>::deviation(const viennacl::vector<Scalar> &theta) const {
    return h_theta(theta) - CostFunction<Scalar>::y_;
}

/**
 *  cost function for given \f$\theta\f$
 *  \param theta
 */
template <typename Scalar>
viennacl::scalar<Scalar>
LinearCostFunction<Scalar>::operator()(const viennacl::vector<Scalar> &theta) const {
    viennacl::vector<Scalar> d = deviation(theta);
#   ifdef DEBUG_LOGISTIC_REGRESSION
    std::cout << "COST(" << theta << ")" << *this << std::endl;
#   endif
    return viennacl::linalg::inner_prod(d, d) / Scalar(2*CostFunction<Scalar>::y_.size());
}

/**
 *  gradient of cost function for given \f$\theta\f$
 *  \param theta
 *  \todo the arguments to prod() may be wrong; in Matlab it is (X*theta-y)' * X, here X' * (X*theta-y)
 */
template <typename Scalar>
viennacl::vector<Scalar>
LinearCostFunction<Scalar>::gradient(const viennacl::vector<Scalar> &theta) const {
    return viennacl::linalg::prod(trans(CostFunction<Scalar>::X_), deviation(theta)) / Scalar(CostFunction<Scalar>::y_.size());
}

#endif