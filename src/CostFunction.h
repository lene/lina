//
// Created by lene on 04.05.15.
//

#ifndef TAVSIYE_COSTFUNCTION_H
#define TAVSIYE_COSTFUNCTION_H


#include <viennacl/vector.hpp>

template<typename Scalar>
class CostFunction {

public:

    CostFunction(
            const viennacl::matrix<Scalar> &X,
            const viennacl::vector<Scalar> &y
    );

    viennacl::vector<Scalar> h_theta(const viennacl::vector<Scalar> &theta) const;
    viennacl::vector<Scalar> deviation(const viennacl::vector<Scalar> &theta) const;
    viennacl::scalar<Scalar> operator()(const viennacl::vector<Scalar> &theta) const;
    viennacl::vector<Scalar> gradient(const viennacl::vector<Scalar> &theta) const;

    const viennacl::matrix<Scalar> &X() { return X_; }

private:
    const viennacl::matrix<Scalar> &X_;
    const viennacl::vector<Scalar> &y_;
};

#include "CostFunction.impl.h"

#endif //TAVSIYE_COSTFUNCTION_H
