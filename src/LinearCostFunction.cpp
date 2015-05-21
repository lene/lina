//
// Created by lene on 07.05.15.
//

#include "LinearCostFunction.h"

template class LinearCostFunction<float>;
template class LinearCostFunction<double>;

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
    assert(theta.size() == X().size2());
    return viennacl::linalg::prod(X(), theta);
}

/**
 *  cost function for given \f$\theta\f$
 *  \param theta
 */
template <typename Scalar>
viennacl::scalar<Scalar>
LinearCostFunction<Scalar>::operator()(const viennacl::vector<Scalar> &theta) const {
    viennacl::vector<Scalar> d = deviation(theta);
    return viennacl::linalg::inner_prod(d, d) / Scalar(2*y().size());
}

/**
 *  gradient of cost function for given \f$\theta\f$
 *  \param theta
 *  \todo the arguments to prod() may be wrong; in Matlab it is (X*theta-y)' * X, here X' * (X*theta-y)
 */
template <typename Scalar>
viennacl::vector<Scalar>
LinearCostFunction<Scalar>::gradient(const viennacl::vector<Scalar> &theta) const {
    return viennacl::linalg::prod(trans(X()), deviation(theta)) / Scalar(y().size());
}