//
// Created by lene on 04.05.15.
//

#ifndef LINA_FILEREADER_IMPL_H
#define LINA_FILEREADER_IMPL_H

#include "FileReader.h"
#include "MatrixPrinter.h"


#include <boost/numeric/ublas/matrix.hpp>
using namespace boost::numeric;

namespace FileReader {

    template<typename Scalar>
    viennacl::matrix<Scalar> read_matrix(const std::string &filename) {
        std::ifstream stream(filename);
        if (!stream.good()) throw std::invalid_argument("unable to open \""+filename+"\"");
        return read_matrix<Scalar>(stream);
    }

    template<typename Scalar>
    viennacl::vector<Scalar> read_vector(const std::string &filename) {
        std::ifstream stream(filename);
        if (!stream.good()) throw std::invalid_argument("unable to open \""+filename+"\"");
        return read_vector<Scalar>(stream);
    }

    template<typename Scalar> viennacl::matrix<Scalar> read_matrix(std::istream &in) {
        unsigned size1, size2;
        in >> size1 >> size2;
        if (size1*size2 == 0) throw std::invalid_argument("bad size");
        ublas::matrix<Scalar> X(size1, size2);
        for (unsigned i = 0; i < size1; ++i)
            for (unsigned j = 0; j < size2; ++j)
                in >> X(i, j);

        viennacl::matrix<Scalar> ret(X.size1(), X.size2());
        viennacl::copy(X, ret);

        return ret;
    }

    template<typename Scalar> viennacl::vector<Scalar> read_vector(std::istream &in) {
        unsigned size;
        in >> size;
        ublas::vector<Scalar> v(size);
        for (unsigned i = 0; i < size; ++i)
            in >> v(i);

        viennacl::vector<Scalar> ret(v.size());
        viennacl::copy(v, ret);

        return ret;
    }

    template<typename Scalar> viennacl::matrix<Scalar> add_bias_column(const viennacl::matrix<Scalar> &M) {
        viennacl::matrix<Scalar> augmented(M.size1(), M.size2()+1);
        for(int i = M.size1(); i >= 0; --i) {
            for (int j = M.size2(); j > 0; --j) {
                augmented(i, j) = M(i, j-1);
            }
            augmented(i,0) = 1;
        }

//        MatrixPrinter<viennacl::matrix<Scalar>> p(augmented);
//        p.print();

        return augmented;
    }
}

#endif