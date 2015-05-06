
#include "FileReader.h"
#include "CostFunction.h"
#include "FeatureNormalize.h"
#include "GradientDescent.h"
#include "VectorPrinter.h"

#include <sstream>

#define VIENNACL_WITH_UBLAS 1
#define VIENNACL_WITH_OPENCL 0

#ifndef NDEBUG
 #define NDEBUG
#endif

typedef float ScalarType;
typedef viennacl::vector<ScalarType> VectorType;
typedef viennacl::matrix<ScalarType> MatrixType;

int main() {

    std::stringstream mstream(FileReader::testmatrix);
    MatrixType X = FileReader::add_bias_column(FileReader::read_matrix<ScalarType>(mstream));
    std::stringstream vstream(FileReader::testvector);
    VectorType y = FileReader::read_vector<ScalarType>(vstream);
    VectorType theta(X.size2());
    theta.clear();              // theta = (0,0,...,0)
MatrixPrinter<MatrixType>p(X);
    p.print("X");

VectorPrinter<VectorType> pv(y);
    pv.print("y");

//    FeatureNormalize<ScalarType> normalize(X);
//    auto throwaway = normalize.normalize();

    CostFunction<ScalarType> cost_function(X, y);
    ScalarType cost = cost_function(theta);

    std::cout << "Cost: " << cost << " dot: " << viennacl::linalg::inner_prod(y, y) << std::endl;

    GradientDescent<ScalarType> grad(cost_function);
    return 0;
    grad.optimize(theta);

    theta = grad.getMinimum();
    VectorPrinter<VectorType> printer(theta);
    printer.print("Optimal theta:");
    std::cout << "Cost: " << cost_function(theta) << std::endl;
    std::copy(
            grad.getHistory().begin(), grad.getHistory().end(),
            std::ostream_iterator<ScalarType >(std::cout, " ")
    );

}
