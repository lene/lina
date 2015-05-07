//
// Created by lene on 04.05.15.
//

#ifndef LINA_FEATURENORMALIZE_H
#define LINA_FEATURENORMALIZE_H

#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/linalg/inner_prod.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include "VectorPrinter.h"
#include "MatrixPrinter.h"

using namespace boost::numeric;

template <typename Scalar>
class FeatureNormalize {
public:
    FeatureNormalize(const viennacl::matrix<Scalar> &X);

    const ublas::matrix<Scalar> &normalize() const {
        return normalized_matrix_;
    }

    const ublas::vector<Scalar> &mu() const {
        return mu_;
    }

    const ublas::vector<Scalar> &sigma() const {
        return sigma_;
    }

    viennacl::matrix<Scalar> restore(const viennacl::matrix<Scalar> &matrix) {
        assert(matrix.size1() == normalized_matrix_.size1());
        assert(matrix.size2() == normalized_matrix_.size2());
        viennacl::matrix<Scalar> restored(normalized_matrix_.size1(), normalized_matrix_.size2());

        for (unsigned long i = 0; i < normalized_matrix_.size2(); ++i) {
            for (unsigned j = 0; j < normalized_matrix_.size1(); ++j) {
                restored(j, i) = matrix(j, i)*sigma_(i)+mu_(i);
            }
        }

        return restored;
    }

    static Scalar sum(const ublas::vector<Scalar> &v) {
        ublas::vector<Scalar> ones = ublas::scalar_vector<Scalar>(v.size(), 1.0);
        return ublas::inner_prod(v, ones);
    }

private:

    Scalar mean(const ublas::vector<Scalar> &v) {
        return sum(v) / v.size();
    }

    Scalar variance(const ublas::vector<Scalar> &v) {
        ublas::vector<Scalar> sqdiff = element_prod(v, v);
        return mean(sqdiff);
    }

    Scalar stddev(const ublas::vector<Scalar> &v) {
        return sqrt(variance(v));
    }

    void runNormalization() {
        for (unsigned long i = 0; i < blas_matrix_.size2(); ++i) {
            ublas::vector<Scalar> column = ublas::column(blas_matrix_, i);
            mu_(i) = mean(column);
            column -= Scalar(mu_(i)) * ublas::scalar_vector<Scalar>(column.size(), 1.0);
            sigma_(i) = stddev(column);
            column /= sigma_(i);
            for (unsigned j = 0; j < blas_matrix_.size1(); ++j) {
                normalized_matrix_(j, i) = column(j);
            }
        }
    }

    ublas::matrix<Scalar> blas_matrix_;
    ublas::matrix<Scalar> normalized_matrix_;
    ublas::vector<Scalar> mu_;
    ublas::vector<Scalar> sigma_;
};


#endif //LINA_FEATURENORMALIZE_H
