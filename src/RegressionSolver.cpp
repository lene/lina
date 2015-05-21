//
// Created by lene on 07.05.15.
//

#include "RegressionSolver.h"

template class RegressionSolver<float>;
template class RegressionSolver<double>;

#include "FeatureNormalize.h"
#include "FileReader.h"

template <typename Scalar, typename Function>
RegressionSolver<Scalar, Function>::RegressionSolver(const Matrix &X, const Vector &y):
        X_(X), y_(y), Xnorm_(X.size1(), X.size2()), Xbias_(X.size1(), X.size2()+1) {
    calculateNormalizedMatrix();
    calculateBiasedMatrix();
    cost_ = std::make_shared<Function>(Xbias_, y_);
    grad_ = std::make_shared<GradientDescent<Scalar>>(*cost_);

}

template <typename Scalar, typename Function>
bool RegressionSolver<Scalar, Function>::optimize(const Vector &theta) {
    return grad_->optimize(theta);
}

template <typename Scalar, typename Function>
viennacl::scalar<Scalar> RegressionSolver<Scalar, Function>::operator()(const viennacl::vector<Scalar> &theta) const {
    return cost_->cost(theta);
}

template <typename Scalar, typename Function>
typename RegressionSolver<Scalar, Function>::Vector RegressionSolver<Scalar, Function>::minTheta() {
    return grad_->getMinimum();
}

template <typename Scalar, typename Function>
void RegressionSolver<Scalar, Function>::calculateNormalizedMatrix() {
    FeatureNormalize<Scalar> normalize(X_);
    auto Xnorm = normalize.normalize();
    copy(Xnorm, Xnorm_);
}

template <typename Scalar, typename Function>
void RegressionSolver<Scalar, Function>::calculateBiasedMatrix() {
    Xbias_ = FileReader::add_bias_column(Xnorm_);
}
