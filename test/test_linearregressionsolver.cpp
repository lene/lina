#include "LinearRegressionSolver.h"
#include "Utilities.h"

#include <gtest/gtest.h>

class LinearRegressionSolverTest : public ::testing::Test {
protected:
    const std::string X_ = "3 2 1 1 .1 0 1 0";
    const std::string y_ = "3 1 0 0";
};

TEST_F(LinearRegressionSolverTest, ConvergesOnSimpleSystem) {
    auto solver = Utilities::linearRegressionSolverFixture(X_, y_);
    ASSERT_TRUE(solver.optimize(Utilities::vectorFixture("3 0 0 0")));
}

TEST_F(LinearRegressionSolverTest, SolvesSimpleSystem) {
    auto solver = Utilities::linearRegressionSolverFixture(X_, y_);
    solver.optimize(Utilities::vectorFixture("3 0 0 0"));
    auto theta = solver.minTheta();
    ASSERT_EQ(3, theta.size());
    for (unsigned i = 0; i < 2; ++i)
            ASSERT_NEAR(Utilities::vectorFixture("3 0.333 0 0.471")(i), theta(i), 1e-3);
}

TEST_F(LinearRegressionSolverTest, SimpleSystemConvergesToCostZero) {
    auto solver = Utilities::linearRegressionSolverFixture(X_, y_);
    solver.optimize(Utilities::vectorFixture("3 0 0 0"));
    auto theta = solver.minTheta();
    ASSERT_EQ(0, solver(theta));
}
