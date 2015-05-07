//
// Created by lene on 05.05.15.
//

#include "CostFunction.h"
#include "Utilities.h"

#include <gtest/gtest.h>

#include <viennacl/matrix.hpp>
#include "viennacl/linalg/inner_prod.hpp"

#include <boost/numeric/ublas/matrix.hpp>
using namespace boost::numeric;

class CostFunctionTest: public ::testing::Test {

protected:

    virtual void SetUp() {
        X_ = Utilities::matrixFixture(x_data_);
        y_ = Utilities::vectorFixture(y_data_);
        theta_ = Utilities::vectorFixture(theta_data);
    }

    viennacl::matrix<float> X_;
    viennacl::vector<float> y_, theta_;

    const std::string x_data_ = "3 2 1 1 1 0 1 0";
    const std::string y_data_ = "3 1 0 0";
    const std::string theta_data = "2 0 0";
};

TEST_F(CostFunctionTest, WorksAtAll) {
    ASSERT_TRUE(true);
}

TEST_F(CostFunctionTest, ElementsGetSetUp) {
    ASSERT_EQ(X_(0,0), 1.f);
    ASSERT_EQ(X_(1,0), 1.f);
    ASSERT_EQ(X_(2,0), 1.f);
    ASSERT_EQ(X_(0,1), 1.f);
    ASSERT_EQ(X_(1,1), 0.f);
    ASSERT_EQ(X_(2,1), 0.f);

    ASSERT_EQ(y_(0), 1.f);
    ASSERT_EQ(viennacl::linalg::inner_prod(y_, y_), 1.f);

    ASSERT_EQ(viennacl::linalg::inner_prod(theta_, theta_), 0.f);
}

TEST_F(CostFunctionTest, CostFunctionInitializes) {
    CostFunction<float> cost(X_, y_);
}

TEST_F(CostFunctionTest, CostFunctionRuns) {
    CostFunction<float> cost(X_, y_);
    cost(theta_);
}

TEST_F(CostFunctionTest, CostFunctionEvaluates) {
    CostFunction<float> cost(X_, y_);
    ASSERT_FLOAT_EQ(1.f/6.f, cost(theta_));
}

TEST_F(CostFunctionTest, CostFunctionIdealTheta) {
    CostFunction<float> cost(X_, y_);
    ASSERT_EQ(0.f, cost(Utilities::vectorFixture("2 0 1")));
}

TEST_F(CostFunctionTest, CostFunctionBadTheta) {
    CostFunction<float> cost(X_, y_);
    ASSERT_LT(cost(theta_), cost(Utilities::vectorFixture("2 0 10")));
}
