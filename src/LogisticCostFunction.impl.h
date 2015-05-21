//
// Created by lene on 11.05.15.
//

#ifndef LINA_LOGISTICCOSTFUNCTION_IMPL_H
#define LINA_LOGISTICCOSTFUNCTION_IMPL_H

#include "LogisticCostFunction.h"

#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/prod.hpp>
#include <viennacl/matrix.hpp>

#include <boost/numeric/ublas/vector.hpp>
using namespace boost::numeric;

using namespace viennacl::linalg;

template<typename Scalar>
LogisticCostFunction<Scalar>::LogisticCostFunction(
        const viennacl::matrix<Scalar> &X,const viennacl::vector<Scalar> &y
): CostFunction<Scalar>(X, y), one_(X.size1()) {
    viennacl::copy(ublas::scalar_vector<Scalar>(X.size1(), 1), one_);
}

template<typename Scalar>
viennacl::vector<Scalar> LogisticCostFunction<Scalar>::sigmoid(const viennacl::vector<Scalar> &v) const {
    assert(v.size() == X().size1());
    viennacl::vector<Scalar> e = element_exp(-v);
    viennacl::vector<Scalar> ep1 = one_+e;
    return element_div(one_, ep1);
}

template<typename Scalar>
viennacl::vector<Scalar> LogisticCostFunction<Scalar>::h_theta(const viennacl::vector<Scalar> &theta) const {
    assert(theta.size() == X().size2());
    viennacl::vector<Scalar> product = prod(X(), theta);
    return sigmoid(product);
}

template<typename Scalar>
viennacl::scalar<Scalar> LogisticCostFunction<Scalar>::operator()(const viennacl::vector<Scalar> &theta) const {
    // -y'*log(h_theta) - (1-y)'*log(1-h_theta)
    assert(theta.size() == X().size2());

    viennacl::vector<Scalar> htheta = h_theta(theta);
    viennacl::scalar<Scalar> c1 = -inner_prod(y(), element_log(htheta));
    viennacl::scalar<Scalar> c2 = -inner_prod(one_ - y(), element_log(one_ - htheta));

    return (c1+c2)/Scalar(y().size());
}

template<typename Scalar>
viennacl::vector<Scalar> LogisticCostFunction<Scalar>::gradient(const viennacl::vector<Scalar> &theta) const {
    // (h_theta-y)'*X/m
    return prod(trans(X()), deviation(theta))/ y().size();
}

#endif //LINA_LOGISTICCOSTFUNCTION_IMPL_H
