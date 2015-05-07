//
// Created by lene on 07.05.15.
//

#ifndef LINA_FEATURENORMALIZE_IMPL_H
#define LINA_FEATURENORMALIZE_IMPL_H

#include "FeatureNormalize.h"

template <typename Scalar>
FeatureNormalize<Scalar>::FeatureNormalize(const viennacl::matrix<Scalar> &X):
        blas_matrix_(X.size1(), X.size2()),
        normalized_matrix_(X.size1(), X.size2()),
        mu_(X.size2()), sigma_(X.size2()) {
    viennacl::copy(X, blas_matrix_);
    runNormalization();
}

#endif //LINA_FEATURENORMALIZE_IMPL_H
