//
// Created by lene on 21.05.15.
//

#ifndef LINA_COSTFUNCTION_H
#define LINA_COSTFUNCTION_H

#include <viennacl/vector.hpp>


template<typename Scalar>
class CostFunction {

    static_assert(std::is_floating_point<Scalar>::value, "Scalar is not a floating point type");

public:

    CostFunction(
            const viennacl::matrix<Scalar> &X,
            const viennacl::vector<Scalar> &y
    );

    virtual viennacl::vector<Scalar> h_theta(const viennacl::vector<Scalar> &theta) const = 0;
    virtual viennacl::scalar<Scalar> cost(const viennacl::vector<Scalar> &theta) const = 0;
    virtual viennacl::vector<Scalar> gradient(const viennacl::vector<Scalar> &theta) const = 0;

    viennacl::vector<Scalar> deviation(const viennacl::vector<Scalar> &theta) const;

    const viennacl::matrix<Scalar> &X() const { return X_; }
    const viennacl::vector<Scalar> &y() const { return y_; }

    friend std::ostream& operator<<(std::ostream& os, const CostFunction<Scalar>& cost) {
        os
#       ifdef DEBUG_LOGISTIC_REGRESSION
            << typeid(cost).name()
#       endif
        << cost.X_ << cost.y_;
        return os;
    }

private:
    const viennacl::matrix<Scalar> &X_;
    const viennacl::vector<Scalar> &y_;

};


#endif //LINA_COSTFUNCTION_H
