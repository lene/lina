//
// Created by lene on 04.05.15.
//

#ifndef LINA_COSTFUNCTION_H
#define LINA_COSTFUNCTION_H

#include "MatrixPrinter.h"
#include "VectorPrinter.h"

#include <viennacl/vector.hpp>

template<typename Scalar>
class CostFunction {

public:

    CostFunction(
            const viennacl::matrix<Scalar> &X,
            const viennacl::vector<Scalar> &y
    );

    virtual viennacl::vector<Scalar> h_theta(const viennacl::vector<Scalar> &theta) const;
    viennacl::vector<Scalar> deviation(const viennacl::vector<Scalar> &theta) const;
    virtual viennacl::scalar<Scalar> operator()(const viennacl::vector<Scalar> &theta) const;
    virtual viennacl::vector<Scalar> gradient(const viennacl::vector<Scalar> &theta) const;

    const viennacl::matrix<Scalar> &X() { return X_; }

    friend std::ostream& operator<<(std::ostream& os, const CostFunction<Scalar>& cost) {
        os << cost.X_ << cost.y_;
        return os;
    }

private:
    const viennacl::matrix<Scalar> &X_;
    const viennacl::vector<Scalar> &y_;
};

#endif //LINA_COSTFUNCTION_H
