//
// Created by lene on 11.05.15.
//

#ifndef LINA_LOGISTICCOSTFUNCTION_IMPL_H
#define LINA_LOGISTICCOSTFUNCTION_IMPL_H

#include "LogisticCostFunction.h"

#include "viennacl/linalg/inner_prod.hpp"
#include <boost/numeric/ublas/vector.hpp>
using namespace boost::numeric;

using namespace viennacl::linalg;

template<typename Scalar>
viennacl::vector<Scalar> LogisticCostFunction<Scalar>::sigmoid(const viennacl::vector<Scalar> &v) {
    viennacl::vector<Scalar> ret(v.size());
    for (unsigned i = 0; i < v.size(); ++i) {
        ret(i) = 1./(1.+exp(-v(i)));
    }
    return ret;
}

template<typename Scalar>
viennacl::vector<Scalar> LogisticCostFunction<Scalar>::log(const viennacl::vector<Scalar> &v) {
    viennacl::vector<Scalar> ret(v.size());
    for (unsigned i = 0; i < v.size(); ++i) {
        ret(i) = ::log(Scalar(v(i)));
    }
    return ret;
}

template<typename Scalar>
viennacl::vector<Scalar> LogisticCostFunction<Scalar>::h_theta(const viennacl::vector<Scalar> &theta) const {
    return sigmoid(CostFunction<Scalar>::h_theta(theta));
}

template<typename Scalar>
viennacl::scalar<Scalar> LogisticCostFunction<Scalar>::operator()(const viennacl::vector<Scalar> &theta) const {
    // -y'*log(h_theta) - (1-y)'*log(1-h_theta)
    viennacl::scalar<Scalar> c1 = -inner_prod(CostFunction<Scalar>::y_, log(h_theta(theta)));
    viennacl::vector<Scalar> one(CostFunction<Scalar>::y_.size());
    viennacl::copy(ublas::scalar_vector<Scalar>(CostFunction<Scalar>::y_.size(), 1), one);
    viennacl::vector<Scalar> one_y = one - CostFunction<Scalar>::y_;
    viennacl::vector<Scalar> one_htheta = log(one - h_theta(theta));
    viennacl::scalar<Scalar> c2 = inner_prod(one_y, one_htheta);
    std::cout
    << "y" << CostFunction<Scalar>::y_
    << "1-y" << one_y
    << "1-h" << one_htheta
    << "c1: " << c1 << " c2:" << c2 << std::endl;
    return c1-c2;
}

template<typename Scalar>
viennacl::vector<Scalar> LogisticCostFunction<Scalar>::gradient(const viennacl::vector<Scalar> &theta) const {
    return CostFunction<Scalar>::gradient(theta);
}

#endif //LINA_LOGISTICCOSTFUNCTION_IMPL_H
