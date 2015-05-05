//
// Created by lene on 04.05.15.
//

#ifndef TAVSIYE_FILEREADER_H
#define TAVSIYE_FILEREADER_H


#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

namespace FileReader {

    template<typename ScalarType> viennacl::matrix<ScalarType> X(const std::string &filename);
    template<typename ScalarType> viennacl::vector<ScalarType> y(const std::string &filename);

};

#include "FileReader.impl.h"

#endif //TAVSIYE_FILEREADER_H
