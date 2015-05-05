//
// Created by lene on 05.05.15.
//

#include "CostFunction.h"

#include <gtest/gtest.h>

#include <viennacl/matrix.hpp>

#include <boost/numeric/ublas/matrix.hpp>
using namespace boost::numeric;

class CostFunctionTest: public ::testing::Test {
protected:
    CostFunctionTest():
            X_(viennacl::matrix<float>(3, 2)),
            y_(viennacl::vector<float>(3)),
            theta_(viennacl::vector<float>(3)) { }

    virtual void SetUp() {
        ublas::matrix<float> X(3, 2);
        ublas::vector<float> y(3);
        ublas::vector<float> theta(X.size1());
        X(0,0) = X(1,0) = X(2,0) = 1.f;
        X(0,1) = 1.f;
        X(1,1) = X(2,1) = 0.f;
        y(0) = 1.f;
        y(1) = y(2) = 0.f;
        theta(0) = theta(1) = theta(2) = 0.f;

        viennacl::copy(X, X_);
        viennacl::copy(y, y_);
        viennacl::copy(theta, theta_);
    }

    viennacl::matrix<float> X_;
    viennacl::vector<float> y_, theta_;

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
