//
// Created by lene on 05.05.15.
//

#include "GradientDescent.h"

template <typename Scalar>
GradientDescent<Scalar>::GradientDescent(const CostFunction<Scalar> &function):
        func_(function),
        alpha_(DEFAULT_LEARNING_RATE),
        max_iter_(DEFAULT_NUM_ITER),
        iter_(0), history_() {}

template <typename Scalar>
bool GradientDescent<Scalar>::optimize(const viennacl::vector<Scalar> &initial_guess) {
    theta_ = initial_guess;
    for (iter_ = 0; iter_ < max_iter_; ++iter_) {
        updateHistory();
        if (hasConverged()) return true;
        adjustLearningRate();
        viennacl::vector<Scalar> temp = func_.gradient(theta_);
        theta_ -= alpha_*temp;
    }
    return false;
}

/**
 *  If the function values grow, the learning rate is too big - scale it down.
 *  If function values drop, maybe we can achieve faster convergence with a bigger learning rate.
 *  Scaling up is done by a smaller factor than scaling down (let's not get too enthusiastic!).
 *  Choose the scale up and down factors to be mutually prime to avoid cycles.
 */
template <typename Scalar>
void GradientDescent<Scalar>::adjustLearningRate() {
    if (history_.back() > history_[history_.size()-2]) alpha_ /= 2;
    else alpha_ *= 1.3;
}

template <typename Scalar>
void GradientDescent<Scalar>::updateHistory() {
    history_.push_back(func_(theta_));
}

/**
 *  Compares the last function value with the third last (not the second last to avoid falling
 *  into a cycle). If the function has not changed it will not change on further iterations.
 */
template <typename Scalar>
bool GradientDescent<Scalar>::hasConverged() {
    if (history_.size() < 3) return false;
    return history_.back() == history_[history_.size()-3];
}