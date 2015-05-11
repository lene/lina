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
    ) : CostFunction<Scalar>(X, y) { }

};


#endif //LINA_LOGISTICCOSTFUNCTION_H
