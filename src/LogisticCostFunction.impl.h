//
// Created by lene on 11.05.15.
//

#ifndef LINA_LOGISTICCOSTFUNCTION_IMPL_H
#define LINA_LOGISTICCOSTFUNCTION_IMPL_H

#include "LogisticCostFunction.h"

template<typename Scalar>
viennacl::vector<Scalar> LogisticCostFunction<Scalar>::h_theta(const viennacl::vector<Scalar> &theta) const {
    return sigmoid(CostFunction<Scalar>::h_theta(theta));
}

template<typename Scalar>
viennacl::scalar<Scalar> LogisticCostFunction<Scalar>::operator()(const viennacl::vector<Scalar> &theta) const {
    return CostFunction<Scalar>::operator()(theta);
}

template<typename Scalar>
viennacl::vector<Scalar> LogisticCostFunction<Scalar>::gradient(const viennacl::vector<Scalar> &theta) const {
    return CostFunction<Scalar>::gradient(theta);
}

#endif //LINA_LOGISTICCOSTFUNCTION_IMPL_H
