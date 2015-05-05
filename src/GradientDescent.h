//
// Created by lene on 05.05.15.
//

#ifndef LINA_GRADIENTDESCENT_H
#define LINA_GRADIENTDESCENT_H

#include "CostFunction.h"

template <typename Scalar>
class GradientDescent {
public:

    const unsigned DEFAULT_NUM_ITER = 10000;
    const Scalar DEFAULT_LEARNING_RATE = 0.1;

    GradientDescent(const CostFunction &function):
            func_(function),
            alpha_(DEFAULT_LEARNING_RATE),
            num_iter_(DEFAULT_NUM_ITER) {}

    bool optimize(const viennacl::vector<Scalar> &initial_guess);

    viennacl::vector<Scalar> getMinimum();

    void setLearningRate(Scalar alpha) {
        assert(alpha > 0);
        alpha_ = alpha;
    }

    void setNumIter(unsigned int num_iter) {
        num_iter_ = num_iter;
    }

private:
    const CostFunction &func_;
    viennacl::vector<Scalar> theta_;
    Scalar alpha_;
    unsigned num_iter_;

};

#include "GradientDescent.impl.h"

#endif //LINA_GRADIENTDESCENT_H
