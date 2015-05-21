//
// Created by lene on 07.05.15.
//

#include "GradientDescent.h"

template class GradientDescent<float>;
template class GradientDescent<double>;

template <typename Scalar>
GradientDescent<Scalar>::GradientDescent(const CostFunction<Scalar> &function):
//        func_(std::make_shared<const LinearCostFunction<Scalar>>(&function)),
        func_(&function),
        alpha_(DEFAULT_LEARNING_RATE),
        max_iter_(DEFAULT_NUM_ITER),
        iter_(0), history_(),
        scale_step_up_factor_(DEFAULT_SCALE_STEP_UP_FACTOR), scale_step_down_factor_(DEFAULT_SCALE_STEP_DOWN_FACTOR),
        skip_convergence_test_(false) { }

template <typename Scalar>
bool GradientDescent<Scalar>::optimize(const viennacl::vector<Scalar> &initial_guess) {
    theta_ = initial_guess;
    for (iter_ = 0; iter_ < max_iter_; ++iter_) {
        updateHistory();
        if (hasConverged()) return true;
        adjustLearningRate();
        viennacl::vector<Scalar> temp = func_->gradient(theta_);
        theta_ -= alpha_*temp;
    }
    return false;
}

/**
 *  If the function values grow, the learning rate is too big - scale it down.
 *  If function values drop, maybe we can achieve faster convergence with a bigger learning rate.
 */
template <typename Scalar>
void GradientDescent<Scalar>::adjustLearningRate() {
    if (history_.back().second > history_[history_.size()-2].second) alpha_ /= scale_step_down_factor_;
    else alpha_ *= scale_step_up_factor_;
}

template <typename Scalar>
void GradientDescent<Scalar>::updateHistory() {
    history_.push_back(std::make_pair(theta_, func_->cost(theta_)));
}

/**
 *  Compares the last function value with the third last (not the second last to avoid falling
 *  into a cycle). If the function has not changed it will not change on further iterations.
 */
template <typename Scalar>
bool GradientDescent<Scalar>::hasConverged() {
    if (skip_convergence_test_) return false;
    if (history_.size() < 3) return false;
    return (history_.back().second == history_[history_.size()-2].second);
}
