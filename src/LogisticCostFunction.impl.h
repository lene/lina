//
// Created by lene on 11.05.15.
//

#ifndef LINA_LOGISTICCOSTFUNCTION_IMPL_H
#define LINA_LOGISTICCOSTFUNCTION_IMPL_H

#include "LogisticCostFunction.h"

#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/prod.hpp"
#include <boost/numeric/ublas/vector.hpp>
using namespace boost::numeric;

using namespace viennacl::linalg;

template<typename Scalar>
LogisticCostFunction<Scalar>::LogisticCostFunction(
        const viennacl::matrix<Scalar> &X,const viennacl::vector<Scalar> &y
): LinearCostFunction<Scalar>(X, y), one_(X.size1()) {
    assert(y.size() == X.size1());
    viennacl::copy(ublas::scalar_vector<Scalar>(X.size1(), 1), one_);
}

template<typename Scalar>
viennacl::vector<Scalar> LogisticCostFunction<Scalar>::sigmoid(const viennacl::vector<Scalar> &v) const {
    assert(v.size() == LinearCostFunction<Scalar>::X_.size1());
    viennacl::vector<Scalar> e = element_exp(-v);
    viennacl::vector<Scalar> ep1 = one_+e;
    return element_div(one_, ep1);
}

template<typename Scalar>
viennacl::vector<Scalar> LogisticCostFunction<Scalar>::h_theta(const viennacl::vector<Scalar> &theta) const {
    assert(theta.size() == LinearCostFunction<Scalar>::X_.size2());
    viennacl::vector<Scalar> product = prod(LinearCostFunction<Scalar>::X_, theta);
    return sigmoid(product);
}

template<typename Scalar>
viennacl::scalar<Scalar> LogisticCostFunction<Scalar>::operator()(const viennacl::vector<Scalar> &theta) const {
    // -y'*log(h_theta) - (1-y)'*log(1-h_theta)
    assert(theta.size() == LinearCostFunction<Scalar>::X_.size2());

    viennacl::vector<Scalar> htheta = h_theta(theta);
    viennacl::scalar<Scalar> c1 = -inner_prod(LinearCostFunction<Scalar>::y_, element_log(htheta));
    viennacl::vector<Scalar> one_y = one_ - LinearCostFunction<Scalar>::y_;
    viennacl::vector<Scalar> one_htheta = element_log(one_ - htheta);
    viennacl::scalar<Scalar> c2 = -inner_prod(one_y, one_htheta);
    auto ret = (c1+c2)/Scalar(LinearCostFunction<Scalar>::y_.size());
#   ifdef DEBUG_LOGISTIC_REGRESSION
    std::cout<< *this << " *** cost(" << theta << ") = " << ret << std::endl;
#   endif
    return ret;
}

template<typename Scalar>
viennacl::vector<Scalar> LogisticCostFunction<Scalar>::gradient(const viennacl::vector<Scalar> &theta) const {
    // (h_theta-y)'*X/m
    viennacl::vector<Scalar> deviation = h_theta(theta)- LinearCostFunction<Scalar>::y_;
    return prod(trans(LinearCostFunction<Scalar>::X_), deviation)/ LinearCostFunction<Scalar>::y_.size();
//    return trans(LinearCostFunction<Scalar>::X_) * deviation/LinearCostFunction<Scalar>::y_.size();
}

#endif //LINA_LOGISTICCOSTFUNCTION_IMPL_H
