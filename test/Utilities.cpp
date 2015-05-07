//
// Created by lene on 07.05.15.
//

#include "Utilities.h"

#include "FileReader.h"

#include <viennacl/matrix.hpp>

Utilities::vVector Utilities::vectorFixture(const std::string &vec_data) {
    std::stringstream stream(vec_data);
    return FileReader::read_vector<float>(stream);
}

Utilities::vMatrix Utilities::matrixFixture(const std::string &mat_data) {
    std::stringstream stream(mat_data);
    return FileReader::read_matrix<float>(stream);
}

CostFunction<float> Utilities::costFunctionFixture(const std::string &mat_data, const std::string &vec_data) {
    return CostFunction<float>(matrixFixture(mat_data), vectorFixture(vec_data));
}

