//
// Created by lene on 07.05.15.
//

#include "Utilities.h"

#include "FileReader.h"

#include <viennacl/matrix.hpp>
#include <LogisticCostFunction.h>

std::vector<Utilities::vVector> Utilities::persistentVectors_ = std::vector<Utilities::vVector>();
std::vector<Utilities::vMatrix> Utilities::persistentMatrices_ = std::vector<Utilities::vMatrix>();

Utilities::vVector Utilities::vectorFixture(const std::string &vec_data) {
    std::stringstream stream(vec_data);
    return FileReader::read_vector<float>(stream);
}

Utilities::vMatrix Utilities::matrixFixture(const std::string &mat_data) {
    std::stringstream stream(mat_data);
    return FileReader::read_matrix<float>(stream);
}

LinearCostFunction<float> Utilities::costFunctionFixture(const std::string &mat_data, const std::string &vec_data) {
    persistentMatrices_.push_back(matrixFixture(mat_data));
    persistentVectors_.push_back(vectorFixture(vec_data));
    return LinearCostFunction<float>(persistentMatrices_.back(), persistentVectors_.back());
}

LogisticCostFunction<float> Utilities::logisticCostFunctionFixture(
        const std::string &mat_data, const std::string &vec_data
) {
    persistentMatrices_.push_back(matrixFixture(mat_data));
    persistentVectors_.push_back(vectorFixture(vec_data));
    return LogisticCostFunction<float>(persistentMatrices_.back(), persistentVectors_.back());
}

RegressionSolver<float> Utilities::linearRegressionSolverFixture(
        const std::string &mat_data, const std::string &vec_data
) {
    persistentMatrices_.push_back(matrixFixture(mat_data));
    persistentVectors_.push_back(vectorFixture(vec_data));
    return RegressionSolver<float, LinearCostFunction<float>>(persistentMatrices_.back(), persistentVectors_.back());
}

RegressionSolver<float, LogisticCostFunction<float>> Utilities::logisticRegressionSolverFixture(
        const std::string &mat_data, const std::string &vec_data
) {
    persistentMatrices_.push_back(matrixFixture(mat_data));
    persistentVectors_.push_back(vectorFixture(vec_data));
    return RegressionSolver<float, LogisticCostFunction<float>>(persistentMatrices_.back(), persistentVectors_.back());
}
