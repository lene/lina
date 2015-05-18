//
// Created by lene on 05.05.15.
//

#ifndef LINA_GRADIENTDESCENT_H
#define LINA_GRADIENTDESCENT_H

#include <memory>
#include "CostFunction.h"

template <typename Scalar>
class GradientDescent {
public:

    const unsigned DEFAULT_NUM_ITER = 10000;
    const Scalar DEFAULT_LEARNING_RATE = 1;

    GradientDescent(const CostFunction<Scalar> &function);

    bool optimize(const viennacl::vector<Scalar> &initial_guess);

    viennacl::vector<Scalar> getMinimum() const { return theta_; }

    void setLearningRate(Scalar alpha) {
        assert(alpha > 0);
        alpha_ = alpha;
    }

    void setMaxIter(unsigned int max_iter) {
        max_iter_ = max_iter;
    }

    unsigned getIterations() const { return iter_; }

    std::vector<std::pair<viennacl::vector<Scalar>, Scalar>> getHistory2() const { return history2_; }

private:

    void updateHistory();
    bool hasConverged();
    void adjustLearningRate();

    std::shared_ptr<const CostFunction<Scalar>> func_;
    viennacl::vector<Scalar> theta_;
    Scalar alpha_;
    unsigned max_iter_;
    unsigned iter_;
    std::vector<std::pair<viennacl::vector<Scalar>, Scalar>> history2_;

};

#endif //LINA_GRADIENTDESCENT_H
