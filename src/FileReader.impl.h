//
// Created by lene on 04.05.15.
//

#include "FileReader.h"


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
}
