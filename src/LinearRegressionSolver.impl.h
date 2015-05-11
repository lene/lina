//
// Created by lene on 07.05.15.
//

#ifndef LINA_LINEARREGRESSIONSOLVER_IMPL_H
#define LINA_LINEARREGRESSIONSOLVER_IMPL_H

#include "LinearRegressionSolver.h"

#include "FeatureNormalize.h"
#include "FileReader.h"
#include "CostFunction.h"
#include "GradientDescent.h"

template <typename Scalar>
LinearRegressionSolver<Scalar>::LinearRegressionSolver(const Matrix &X, const Vector &y):
        X_(X), y_(y), Xnorm_(X.size1(), X.size2()), Xbias_(X.size1(), X.size2()+1) {
    calculateNormalizedMatrix();
    calculateBiasedMatrix();
    cost_ = std::make_shared<CostFunction<Scalar>>(Xbias_, y_);
    grad_ = std::make_shared<GradientDescent<Scalar>>(*cost_);

}

template <typename Scalar>
bool LinearRegressionSolver<Scalar>::optimize(const Vector &theta) {
    return grad_->optimize(theta);
}

template <typename Scalar>
viennacl::scalar<Scalar> LinearRegressionSolver<Scalar>::operator()(const viennacl::vector<Scalar> &theta) const {
    return cost_->operator()(theta);
}

template <typename Scalar>
typename LinearRegressionSolver<Scalar>::Vector LinearRegressionSolver<Scalar>::minTheta() {
    return grad_->getMinimum();
}

template <typename Scalar>
void LinearRegressionSolver<Scalar>::calculateNormalizedMatrix() {
    FeatureNormalize<Scalar> normalize(X_);
    auto Xnorm = normalize.normalize();
    copy(Xnorm, Xnorm_);
}

template <typename Scalar>
void LinearRegressionSolver<Scalar>::calculateBiasedMatrix() {
    Xbias_ = FileReader::add_bias_column(Xnorm_);
}

#endif //LINA_LINEARREGRESSIONSOLVER_IMPL_H