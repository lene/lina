//
// Created by lene on 07.05.15.
//

#include "GradientDescent.h"
#include "Utilities.h"
#include "CostFunction.h"

#include <gtest/gtest.h>

class GradDescTest: public ::testing::Test {
protected:
};

TEST_F(GradDescTest, Initializes) {
    auto cost = Utilities::costFunctionFixture("3 2 1 1 1 0 1 0", "3 1 0 0");
    GradientDescent<float> grad(cost);
}

TEST_F(GradDescTest, ConvergesOnSimpleSystem) {
    auto cost = Utilities::costFunctionFixture("3 2 1 1 1 0 1 0", "3 1 0 0");
    GradientDescent<float> grad(cost);
    ASSERT_TRUE(grad.optimize(Utilities::vectorFixture("2 0 0")));
}

TEST_F(GradDescTest, SolvesSimpleSystem) {
    auto cost = Utilities::costFunctionFixture("3 2 1 1 1 0 1 0", "3 1 0 0");
    GradientDescent<float> grad(cost);
    grad.optimize(Utilities::vectorFixture("2 0 0"));
    auto theta = grad.getMinimum();
    ASSERT_EQ(2, theta.size());
    for (unsigned i = 0; i < 2; ++i)
        ASSERT_NEAR(Utilities::vectorFixture("2 0 1")(i), theta(i), 1e-6);
}

TEST_F(GradDescTest, SimpleSystemConvergesToCostZero) {
    auto cost = Utilities::costFunctionFixture("3 2 1 1 1 0 1 0", "3 1 0 0");
    GradientDescent<float> grad(cost);
    grad.optimize(Utilities::vectorFixture("2 0 0"));
    auto theta = grad.getMinimum();
    ASSERT_EQ(0, cost(theta));
}

TEST_F(GradDescTest, IterationsAreOneLessThanHistorySizeCauseTheyStartAtZero) {
    auto cost = Utilities::costFunctionFixture("3 2 1 1 1 0 1 0", "3 1 0 0");
    GradientDescent<float> grad(cost);
    grad.optimize(Utilities::vectorFixture("2 0 0"));
    ASSERT_EQ(grad.getIterations()+1, grad.getHistory().size());
}

TEST_F(GradDescTest, LastThreeIterationsEqual) {
    auto cost = Utilities::costFunctionFixture("3 2 1 1 1 0 1 0", "3 1 0 0");
    GradientDescent<float> grad(cost);
    grad.optimize(Utilities::vectorFixture("2 0 0"));
    auto history = grad.getHistory();
    ASSERT_EQ(0, history.back());
    ASSERT_EQ(0, history[history.size()-2]);
    ASSERT_EQ(0, history[history.size()-3]);
}

TEST_F(GradDescTest, FourthLastIterationNotEqual) {
    auto cost = Utilities::costFunctionFixture("3 2 1 1 1 0 1 0", "3 1 0 0");
    GradientDescent<float> grad(cost);
    grad.optimize(Utilities::vectorFixture("2 0 0"));
    auto history = grad.getHistory();
    ASSERT_NE(history.back(), history[history.size()-4]);
}

TEST_F(GradDescTest, ConvergesImmediatelyOnCorrectGuess) {
    auto cost = Utilities::costFunctionFixture("3 2 1 1 1 0 1 0", "3 1 0 0");
    GradientDescent<float> grad(cost);
    grad.optimize(Utilities::vectorFixture("2 0 1"));
    auto theta = grad.getMinimum();
    ASSERT_EQ(0, cost(theta));
    for (auto x: grad.getHistory()) ASSERT_EQ(0, x);
}

TEST_F(GradDescTest, DoesNotConvergeIfMaxIterTooLow) {
    auto cost = Utilities::costFunctionFixture("3 2 1 1 1 0 1 0", "3 1 0 0");
    GradientDescent<float> grad(cost);
    grad.setMaxIter(5);
    ASSERT_FALSE(grad.optimize(Utilities::vectorFixture("2 0 0")));
    auto theta = grad.getMinimum();
    ASSERT_NE(0, cost(theta));
}

TEST_F(GradDescTest, CanRestartIfNotConverged) {
    auto cost = Utilities::costFunctionFixture("3 2 1 1 1 0 1 0", "3 1 0 0");
    GradientDescent<float> grad(cost);
    grad.setMaxIter(5);
    grad.optimize(Utilities::vectorFixture("2 0 0"));
    auto theta = grad.getMinimum();
    grad.optimize(theta);
    ASSERT_LT(cost(grad.getMinimum()), cost(theta));
}
