//
// Created by lene on 05.05.15.
//

#include "GradientDescent.h"
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

template <typename Scalar>
void GradientDescent<Scalar>::adjustLearningRate() {
    if (history_.back() > history_[history_.size()-2]) alpha_ /= 2;
    else alpha_ *= 1.3;
}

template <typename Scalar>
void GradientDescent<Scalar>::updateHistory() {
    history_.push_back(func_(theta_));
}

template <typename Scalar>
bool GradientDescent<Scalar>::hasConverged() {
    return history_.back() == history_[history_.size()-2];
}