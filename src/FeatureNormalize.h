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
#include "VectorPrinter.h"
#include "MatrixPrinter.h"

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
        ublas::vector<ScalarType> sqdiff = element_prod(v, v);
        return mean(sqdiff);
    }

    ScalarType stddev(const ublas::vector<ScalarType> &v) {
        return sqrt(variance(v));
    }

    void runNormalization() {
        unsigned long len = blas_matrix_.size2();
        for (unsigned long i = 0; i < len; ++i) {
            ublas::vector<ScalarType> column = ublas::column(blas_matrix_, i);
            VectorPrinter<ublas::vector<ScalarType>> pc(column);
            pc.print("col");
            mu_(i) = mean(column);
            std::cout << "mu: " << mu_(i) << std::endl;
            ublas::vector<ScalarType> normalized_column = ublas::column(blas_matrix_, i);
            normalized_column -= ScalarType(mu_(i)) * ublas::scalar_vector<ScalarType>(normalized_column.size(), 1.0);
            VectorPrinter<ublas::vector<ScalarType>> pnc(normalized_column);
            pnc.print("normalized 1");
            sigma_(i) = stddev(normalized_column);
            std::cout << "sigma: " << sigma_(i) << std::endl;
            normalized_column /= sigma_(i);
            VectorPrinter<ublas::vector<ScalarType>> pnc2(normalized_column);
            pnc2.print("normalized 2");
            for (unsigned j = 0; j < blas_matrix_.size1(); ++j) {
                normalized_matrix_(j, i) = normalized_column(j);
            }
        }
        MatrixPrinter<ublas::matrix<ScalarType>> p(normalized_matrix_);
        p.print("normalized");

    }

    ublas::matrix<ScalarType> blas_matrix_;
    ublas::matrix<ScalarType> normalized_matrix_;
    ublas::vector<ScalarType> mu_;
    ublas::vector<ScalarType> sigma_;
};


#endif //TAVSIYE_FEATURENORMALIZE_H
