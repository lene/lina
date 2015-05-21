//
// Created by lene on 07.05.15.
//

#ifndef LINA_LINEARREGRESSIONSOLVER_H
#define LINA_LINEARREGRESSIONSOLVER_H

#include "GradientDescent.h"
#include "CostFunction.h"

#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>

#include <memory>

template <typename Scalar, typename Function = LinearCostFunction<Scalar>>
class RegressionSolver {

    static_assert(std::is_floating_point<Scalar>::value, "Scalar is not a floating point type");
    static_assert(std::is_base_of<CostFunction<Scalar>, Function>::value, "Function does not implement CostFunction");

protected:
    typedef viennacl::vector<Scalar> Vector;
    typedef viennacl::matrix<Scalar> Matrix;

public:
    RegressionSolver(
            const Matrix &X, const Vector &y
    );

    bool optimize(const Vector &theta);
    viennacl::scalar<Scalar> operator()(const viennacl::vector<Scalar> &theta) const;

    Vector minTheta();

    Matrix Xbias() { return Xbias_; }

private:

    void calculateNormalizedMatrix();
    void calculateBiasedMatrix();

    const Matrix &X_;
    Matrix Xnorm_;
    Matrix Xbias_;
    const Vector &y_;
    std::shared_ptr<LinearCostFunction<Scalar>> cost_;
    std::shared_ptr<GradientDescent<Scalar>> grad_;
};


#endif //LINA_LINEARREGRESSIONSOLVER_H
