//
// Created by lene on 04.05.15.
//

#ifndef TAVSIYE_COSTFUNCTION_H
#define TAVSIYE_COSTFUNCTION_H


#include <viennacl/vector.hpp>

template<typename ScalarType>
class CostFunction {

public:
    /**
     *  X: features
     *  y: training examples (y_i = f(X_i))
     */
    CostFunction(
            const viennacl::matrix<ScalarType> &X,
            const viennacl::vector<ScalarType> &y
    );

    viennacl::vector<ScalarType> h_theta(const viennacl::vector<ScalarType> &theta) const;
    viennacl::vector<ScalarType> deviation(const viennacl::vector<ScalarType> &theta) const;
    viennacl::scalar<ScalarType> operator()(const viennacl::vector<ScalarType> &theta) const;
    viennacl::vector<ScalarType> gradient(const viennacl::vector<ScalarType> &theta) const;

    const viennacl::matrix<ScalarType> &X() { return X_; }

private:
    const viennacl::matrix<ScalarType> &X_;
    const viennacl::vector<ScalarType> &y_;
};

#include "CostFunction.impl.h"

#endif //TAVSIYE_COSTFUNCTION_H
