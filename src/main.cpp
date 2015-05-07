
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

typedef double Scalar;
typedef viennacl::vector<Scalar> Vector;
typedef viennacl::matrix<Scalar> Matrix;

void debugGradientDescent(const GradientDescent<Scalar> &grad, const CostFunction<Scalar> &cost, const Vector &theta) {
    std::cout << grad.getHistory().size() << " steps: ";
    for (auto val: grad.getHistory()) std::cout << val << " ";
    std::cout << std::endl;

}

Vector optimalTheta(const Matrix &X, const Vector &y) {
    FeatureNormalize<Scalar> normalize(X);
    auto Xnorm = normalize.normalize();
    viennacl::matrix<Scalar> vXnorm(Xnorm.size1(), Xnorm.size2());
    copy(Xnorm, vXnorm);
    auto Xbias = FileReader::add_bias_column(vXnorm);

    Vector theta(Xbias.size2());
    theta.clear();              // theta = (0,0,...,0)

    CostFunction<Scalar> cost_function(Xbias, y);

    GradientDescent<Scalar> grad(cost_function);
    grad.optimize(theta);

    debugGradientDescent(grad, cost_function, theta);

    return grad.getMinimum();
}

int main() {

    std::stringstream mstream(FileReader::testmatrix);
    Matrix Xorig = FileReader::read_matrix<Scalar>(mstream);
    std::stringstream vstream(FileReader::testvector);
    Vector y = FileReader::read_vector<Scalar>(vstream);

    Vector theta = optimalTheta(Xorig, y);

    VectorPrinter<Vector> printer(theta);
    printer.print("Optimal theta:");

    return EXIT_SUCCESS;

}
