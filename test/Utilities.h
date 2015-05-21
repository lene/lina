//
// Created by lene on 07.05.15.
//

#ifndef LINA_UTILITIES_H
#define LINA_UTILITIES_H

#include "LinearCostFunction.h"
#include "RegressionSolver.h"
#include "LogisticCostFunction.h"

#include <viennacl/matrix.hpp>

#include <boost/numeric/ublas/matrix.hpp>

using namespace boost::numeric;

class Utilities {

public:

    typedef ublas::vector<float> uVector;
    typedef viennacl::vector<float> vVector;
    typedef ublas::matrix<float> uMatrix;
    typedef viennacl::matrix<float> vMatrix;

    static vVector vectorFixture(const std::string &vec_data = "3\n0.1 0.2 0.3");
    static vMatrix matrixFixture(const std::string &mat_data = "3 2\n0.1 0.2\n1.0 2.0\n10.0 20.0");
    static LinearCostFunction<float> costFunctionFixture(
            const std::string &mat_data = "3 2\n0.1 0.2\n1.0 2.0\n10.0 20.0",
            const std::string &vec_data = "3\n0.1 0.2 0.3"
    );
    static LogisticCostFunction<float> logisticCostFunctionFixture(
            const std::string &mat_data = "3 2\n0.1 0.2\n1.0 2.0\n10.0 20.0",
            const std::string &vec_data = "3\n0 1 0"
    );
    static RegressionSolver<float> linearRegressionSolverFixture(
            const std::string &mat_data = "3 2\n0.1 0.2\n1.0 2.0\n10.0 20.0",
            const std::string &vec_data = "3\n0.1 0.2 0.3"
    );


private:
    static std::vector<vVector> persistentVectors_;
    static std::vector<vMatrix> persistentMatrices_;
};


#endif //LINA_UTILITIES_H
