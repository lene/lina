//
// Created by lene on 04.05.15.
//

#ifndef TAVSIYE_FILEREADER_H
#define TAVSIYE_FILEREADER_H


#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

namespace FileReader {

    template<typename Scalar> viennacl::matrix<Scalar> read_matrix(const std::string &filename);
    template<typename Scalar> viennacl::matrix<Scalar> read_matrix(std::istream &in);
    template<typename Scalar> viennacl::vector<Scalar> read_vector(const std::string &filename);
    template<typename Scalar> viennacl::vector<Scalar> read_vector(std::istream &in);

};

#include "FileReader.impl.h"

#endif //TAVSIYE_FILEREADER_H
