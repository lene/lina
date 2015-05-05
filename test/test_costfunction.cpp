//
// Created by lene on 05.05.15.
//

#include "CostFunction.h"

#include <gtest/gtest.h>

#include <viennacl/matrix.hpp>

#include <boost/numeric/ublas/matrix.hpp>
using namespace boost::numeric;

class CostFunctionTest: public ::testing::Test {

public:
    template <typename Scalar, typename Container = std::vector<Scalar> >
    ublas::matrix<float> umat(const Container &values, int size1, int size2) {
        ublas::matrix<float> X(size1, size2);
        for (int i = 0; i < size1; ++i)
            for(int j = 0; j < size2; ++j)
                X(i, j) = values[i*size2+j];
        return X;
    }

    template <typename Scalar, typename Container = std::vector<Scalar> >
    void vmat(viennacl::matrix<Scalar> &mat, const Container &values, int size1, int size2) {
        viennacl::copy(umat<Scalar>(values, size1, size2), mat);
    }

    template <typename Scalar, typename Container = std::vector<Scalar> >
    ublas::vector<Scalar> uvec(const Container &values, int size) {
        ublas::vector<Scalar> v(size);
        for (int i = 0; i < size; ++i)
            v(i) = values[i];
        return v;
    }

    template <typename Scalar, typename Container = std::vector<Scalar> >
    void vvec(viennacl::vector<Scalar> &vec, const Container &values, int size) {
        viennacl::copy(uvec<Scalar>(values, size), vec);
    }

protected:
    CostFunctionTest():
            X_(viennacl::matrix<float>(3, 2)),
            y_(viennacl::vector<float>(2)),
            theta_(viennacl::vector<float>(3)) { }

    virtual void SetUp() {
        std::vector<float> data = { 1.f, 1.f,  1.f, 0.f,  1.f, 0.f };
        vmat<float>(X_, data, 3, 2);
        data = { 1.f, 0.f };
        vvec<float>(y_, data, 2);
        data = { 0.f, 0.f, 0.f };
        vvec<float>(theta_, data, 3);
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

TEST_F(CostFunctionTest, CostFunctionInitializes) {
    CostFunction<float> cost(X_, y_);
}

TEST_F(CostFunctionTest, CostFunctionRuns) {
    CostFunction<float> cost(X_, y_);
    cost(theta_);
}

TEST_F(CostFunctionTest, CostFunctionEvaluates) {
    CostFunction<float> cost(X_, y_);
    ASSERT_EQ(cost(theta_), 1.f);
}