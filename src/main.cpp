
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
typedef viennacl::vector<Scalar> VectorType;
typedef viennacl::matrix<Scalar> MatrixType;

int main() {

    std::stringstream mstream(FileReader::testmatrix);
    MatrixType Xorig = FileReader::read_matrix<Scalar>(mstream);
    std::stringstream vstream(FileReader::testvector);
    VectorType y = FileReader::read_vector<Scalar>(vstream);

    MatrixPrinter<MatrixType>p(Xorig);
    p.print("X");

    VectorPrinter<VectorType> pv(y);
    pv.print("y");

    FeatureNormalize<Scalar> normalize(Xorig);
    auto Xnorm = normalize.normalize();
    viennacl::matrix<Scalar> vXnorm(Xnorm.size1(), Xnorm.size2());
    copy(Xnorm, vXnorm);
    auto Xbias = FileReader::add_bias_column(vXnorm);

    MatrixPrinter<MatrixType>pb(Xbias);
    pb.print("Xbias");

    VectorType theta(Xbias.size2());
    theta.clear();              // theta = (0,0,...,0)

    CostFunction<Scalar> cost_function(Xbias, y);
    Scalar cost = cost_function(theta);

    std::cout << "Cost: " << cost << " dot: " << viennacl::linalg::inner_prod(y, y) << std::endl;

    GradientDescent<Scalar> grad(cost_function);
    grad.setLearningRate(1);
    grad.optimize(theta);

    theta = grad.getMinimum();
    VectorPrinter<VectorType> printer(theta);
    printer.print("Optimal theta:");
    std::cout << "Cost: " << cost_function(theta) << std::endl;
    std::cout << grad.getHistory().size() << " steps: ";
    for (auto val: grad.getHistory()) std::cout << val << " ";
    std::cout << std::endl;
    return 0;

}
