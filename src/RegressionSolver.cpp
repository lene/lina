//
// Created by lene on 07.05.15.
//

#include "RegressionSolver.h"

template class RegressionSolver<float>;
template class RegressionSolver<double>;

#include "FeatureNormalize.h"
#include "FileReader.h"
#include "LinearCostFunction.h"
#include "GradientDescent.h"

template <typename Scalar>
RegressionSolver<Scalar>::RegressionSolver(const Matrix &X, const Vector &y):
        X_(X), y_(y), Xnorm_(X.size1(), X.size2()), Xbias_(X.size1(), X.size2()+1) {
    calculateNormalizedMatrix();
    calculateBiasedMatrix();
    cost_ = std::make_shared<LinearCostFunction<Scalar>>(Xbias_, y_);
    grad_ = std::make_shared<GradientDescent<Scalar>>(*cost_);

}

template <typename Scalar>
bool RegressionSolver<Scalar>::optimize(const Vector &theta) {
    return grad_->optimize(theta);
}

template <typename Scalar>
viennacl::scalar<Scalar> RegressionSolver<Scalar>::operator()(const viennacl::vector<Scalar> &theta) const {
    return cost_->cost(theta);
}

template <typename Scalar>
typename RegressionSolver<Scalar>::Vector RegressionSolver<Scalar>::minTheta() {
    return grad_->getMinimum();
}

template <typename Scalar>
void RegressionSolver<Scalar>::calculateNormalizedMatrix() {
    FeatureNormalize<Scalar> normalize(X_);
    auto Xnorm = normalize.normalize();
    copy(Xnorm, Xnorm_);
}

template <typename Scalar>
void RegressionSolver<Scalar>::calculateBiasedMatrix() {
    Xbias_ = FileReader::add_bias_column(Xnorm_);
}
