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

    std::vector<std::pair<viennacl::vector<Scalar>, Scalar>> getHistory() const { return history_; }

    void setScaleStepUpFactor(Scalar scale_step_up_factor) { scale_step_up_factor_ = scale_step_up_factor; }
    void setScaleStepDownFactor(Scalar scale_step_down_factor) { scale_step_down_factor_ = scale_step_down_factor; }
    void setSkipConvergenceTest(bool skip) { skip_convergence_test_ = skip; }

private:

    void updateHistory();
    bool hasConverged();
    void adjustLearningRate();

//    std::shared_ptr<const CostFunction<Scalar>> func_;
    const CostFunction<Scalar> *func_;
    viennacl::vector<Scalar> theta_;
    Scalar alpha_;
    unsigned max_iter_;
    unsigned iter_;
    std::vector<std::pair<viennacl::vector<Scalar>, Scalar>> history_;

    Scalar scale_step_up_factor_;
    Scalar scale_step_down_factor_;

    bool skip_convergence_test_;

    /**
     *  Scaling up is done by a smaller factor than scaling down (let's not get too enthusiastic!).
     *  Choose the scale up and down factors to be mutually prime to avoid cycles.
     */
    static constexpr Scalar DEFAULT_SCALE_STEP_UP_FACTOR = Scalar(1.2);
    static constexpr Scalar DEFAULT_SCALE_STEP_DOWN_FACTOR = Scalar(2.0);

};

#endif //LINA_GRADIENTDESCENT_H
