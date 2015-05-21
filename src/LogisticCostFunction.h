//
// Created by lene on 11.05.15.
//

#ifndef LINA_LOGISTICCOSTFUNCTION_H
#define LINA_LOGISTICCOSTFUNCTION_H


#include "CostFunction.h"

template<typename Scalar>
class LogisticCostFunction: public CostFunction<Scalar> {

public:
    LogisticCostFunction(
            const viennacl::matrix<Scalar> &X,
            const viennacl::vector<Scalar> &y
    );

    viennacl::vector<Scalar> sigmoid(const viennacl::vector<Scalar> &v) const;

    virtual viennacl::vector<Scalar> h_theta(const viennacl::vector<Scalar> &theta) const override;

    virtual viennacl::scalar<Scalar> operator()(const viennacl::vector<Scalar> &theta) const override;

    virtual viennacl::vector<Scalar> gradient(const viennacl::vector<Scalar> &theta) const override;

    using CostFunction<Scalar>::X;
    using CostFunction<Scalar>::y;

private:
    viennacl::vector<Scalar> one_;

};


#endif //LINA_LOGISTICCOSTFUNCTION_H
