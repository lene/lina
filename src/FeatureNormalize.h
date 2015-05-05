//
// Created by lene on 04.05.15.
//

#ifndef TAVSIYE_FEATURENORMALIZE_H
#define TAVSIYE_FEATURENORMALIZE_H

#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/linalg/inner_prod.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
using namespace boost::numeric;

template <typename ScalarType>
class FeatureNormalize {
public:
    FeatureNormalize(const viennacl::matrix<ScalarType> &X):
            blas_matrix_(X.size1(), X.size2()),
            normalized_matrix_(X.size1(), X.size2()),
            mu_(X.size1()), sigma_(X.size2()) {
        viennacl::copy(X, blas_matrix_);
        runNormalization();
    }

    const ublas::matrix<ScalarType> &normalize() {
        return normalized_matrix_;
    }

    viennacl::matrix<ScalarType> restore(const viennacl::matrix<ScalarType> &matrix);

private:

    ScalarType sum(const ublas::vector<ScalarType> &v) {
        ublas::vector<ScalarType> ones = ublas::scalar_vector<ScalarType>(v.size(), 1.0);
        return ublas::inner_prod(v, ones);
    }

    ScalarType mean(const ublas::vector<ScalarType> &v) {
        return sum(v) / v.size();
    }

    ScalarType variance(const ublas::vector<ScalarType> &v) {
        ublas::vector<ScalarType> mean_val = ublas::scalar_vector<ScalarType>(v.size(), mean(v));
        ublas::vector<ScalarType> difference = v - mean_val;
        for (auto x: difference) x *= x;
        return mean(difference);
    }

    ScalarType stddev(const ublas::vector<ScalarType> &v) {
        return sqrt(variance(v));
    }

    void runNormalization() {
        unsigned long len = blas_matrix_.size2();
        for (unsigned long i = 0; i < len; ++i) {
            ublas::vector<ScalarType> column = ublas::column(blas_matrix_, i);
            mu_(i) = mean(column);
            ublas::vector<ScalarType> normalized_column = ublas::column(blas_matrix_, i);
            normalized_column -= ScalarType(mu_(i)) * ublas::scalar_vector<ScalarType>(normalized_column.size(), 1.0);
            sigma_(i) = stddev(normalized_column);
            normalized_column /= sigma_(i);
        }

    }

    ublas::matrix<ScalarType> blas_matrix_;
    ublas::matrix<ScalarType> normalized_matrix_;
    ublas::vector<ScalarType> mu_;
    ublas::vector<ScalarType> sigma_;
};


#endif //TAVSIYE_FEATURENORMALIZE_H
