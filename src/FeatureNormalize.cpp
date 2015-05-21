//
// Created by lene on 04.05.15.
//

#include "FeatureNormalize.h"

template class FeatureNormalize<float>;
template class FeatureNormalize<double>;

template <typename Scalar>
FeatureNormalize<Scalar>::FeatureNormalize(const viennacl::matrix<Scalar> &X):
        blas_matrix_(X.size1(), X.size2()),
        normalized_matrix_(X.size1(), X.size2()),
        mu_(X.size2()), sigma_(X.size2()) {
    viennacl::copy(X, blas_matrix_);
    runNormalization();
}
