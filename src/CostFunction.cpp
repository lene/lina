//
// Created by lene on 21.05.15.
//

#include "CostFunction.h"

#include <viennacl/matrix.hpp>

template class CostFunction<float>;
template class CostFunction<double>;

template<class Scalar>
CostFunction<Scalar>::CostFunction(
        const viennacl::matrix<Scalar> &X,
        const viennacl::vector<Scalar> &y): X_(X), y_(y) {
    assert(CostFunction<Scalar>::y_.size() == CostFunction<Scalar>::X_.size1());
}

/**
 *  How far hypothesis \f$h_\theta(X)\f$ misses training examples \f$y\f$
 *  \param theta
 */
template <typename Scalar>
viennacl::vector<Scalar>
CostFunction<Scalar>::deviation(const viennacl::vector<Scalar> &theta) const {
    return h_theta(theta) - y();
}
