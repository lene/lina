//
// Created by lene on 04.05.15.
//

#ifndef LINA_LINEARCOSTFUNCTION_H
#define LINA_LINEARCOSTFUNCTION_H

#include "MatrixPrinter.h"
#include "VectorPrinter.h"
#include "CostFunction.h"

#include <viennacl/vector.hpp>
#include <typeinfo>

template<typename Scalar>
class LinearCostFunction: public CostFunction<Scalar> {

    static_assert(std::is_floating_point<Scalar>::value, "Scalar is not a floating point type");

public:

    LinearCostFunction(
            const viennacl::matrix<Scalar> &X,
            const viennacl::vector<Scalar> &y
    );

    virtual viennacl::vector<Scalar> h_theta(const viennacl::vector<Scalar> &theta) const;
    virtual viennacl::scalar<Scalar> cost(const viennacl::vector<Scalar> &theta) const;
    virtual viennacl::vector<Scalar> gradient(const viennacl::vector<Scalar> &theta) const;

    using CostFunction<Scalar>::X;
    using CostFunction<Scalar>::y;
    using CostFunction<Scalar>::deviation;

};

#endif //LINA_COSTFUNCTION_H
