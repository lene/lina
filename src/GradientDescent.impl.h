//
// Created by lene on 05.05.15.
//

#include "GradientDescent.h"
template <typename Scalar>
bool GradientDescent<Scalar>::optimize(const viennacl::vector<Scalar> &initial_guess) {
    theta_ = initial_guess;
    for (unsigned iter = 0; iter < num_iter_; ++iter) {
        viennacl::vector<Scalar> temp = func_.gradient(theta_);
        theta_ -= alpha_*temp;
    }
}