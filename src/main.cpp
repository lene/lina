
#include "FileReader.h"
#include "CostFunction.h"
#include "FeatureNormalize.h"

#define VIENNACL_WITH_UBLAS 1
#define VIENNACL_WITH_OPENCL 0

#ifndef NDEBUG
 #define NDEBUG
#endif

typedef float ScalarType;
typedef viennacl::vector<ScalarType> VectorType;
typedef viennacl::matrix<ScalarType> MatrixType;

int main() {

    MatrixType X = FileReader::X<ScalarType>(std::string(""));
    VectorType y = FileReader::y<ScalarType>(std::string(""));
    VectorType theta(X.size1());

//    FeatureNormalize<ScalarType> normalize(X);
//    auto throwaway = normalize.normalize();

    CostFunction<ScalarType> cost_function(X, y);
    ScalarType cost = cost_function(theta);

    std::cout << "Cost: " << cost << " dot: " << viennacl::linalg::inner_prod(y, y);

    return 0;
}
