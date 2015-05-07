
#include "FileReader.h"
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

const std::string testmatrix = "47 2 \
2104 3  1600 3  2400 3  1416 2  3000 4 \
1985 4  1534 3  1427 3  1380 3  1494 3 \
1940 4  2000 3  1890 3  4478 5  1268 3 \
2300 4  1320 2  1236 3  2609 4  3031 4 \
1767 3  1888 2  1604 3  1962 4  3890 3 \
1100 3  1458 3  2526 3  2200 3  2637 3 \
1839 2  1000 1  2040 4  3137 3  1811 4 \
1437 3  1239 3  2132 4  4215 4  2162 4 \
1664 2  2238 3  2567 4  1200 3   852 2 \
1852 4  1203 3";
const std::string testvector ="47 \
399900 329900 369000 232000 539900 299900 314900 198999 212000 242500 239999 347000 329999 699900 259900 449900 299900 199900 499998 599000 252900 255000 242900 259900 573900 249900 464500 469000 475000 299900 349900 169900 314900 579900 285900 249900 229900 345000 549000 287000 368500 329900 314000 299000 179900 299900 239500";

void debugGradientDescent(const GradientDescent<Scalar> &grad) {
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

    debugGradientDescent(grad);

    return grad.getMinimum();
}

int main() {

    std::stringstream mstream(testmatrix);
    Matrix Xorig = FileReader::read_matrix<Scalar>(mstream);
    std::stringstream vstream(testvector);
    Vector y = FileReader::read_vector<Scalar>(vstream);

    Vector theta = optimalTheta(Xorig, y);

    VectorPrinter<Vector> printer(theta);
    printer.print("Optimal theta:");

    return EXIT_SUCCESS;

}
