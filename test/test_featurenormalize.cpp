//
// Created by lene on 06.05.15.
//

#include "FeatureNormalize.h"
#include "FileReader.h"
#include "Utilities.h"

#include <gtest/gtest.h>

class FeatureNormalizeTest: public ::testing::Test {
protected:
    FeatureNormalizeTest() { }

    const std::string mat_data_ = "3 2\n0.1 0.2\n1.0 2.0\n10.0 20.0";
};

TEST_F(FeatureNormalizeTest, RunsAtAll) {
    FeatureNormalize<float> F(Utilities::matrixFixture(mat_data_));
}

TEST_F(FeatureNormalizeTest, NormalizedMatrixSize) {
    viennacl::matrix<float> M = Utilities::matrixFixture(mat_data_);
    FeatureNormalize<float> F(M);
    ublas::matrix<float> normalized = F.normalize();
    ASSERT_EQ(M.size1(), normalized.size1());
    ASSERT_EQ(M.size2(), normalized.size2());
}

TEST_F(FeatureNormalizeTest, NormalizedColumsAverageZero) {
    FeatureNormalize<float> F(Utilities::matrixFixture(mat_data_));
    ublas::matrix<float> normalized = F.normalize();
    for (unsigned long i = 0; i < normalized.size2(); ++i) {
        ublas::vector<float> column = ublas::column(normalized, i);
        ASSERT_FLOAT_EQ(0., FeatureNormalize<float>::sum(column));
    }
}

TEST_F(FeatureNormalizeTest, MeanValues) {
    FeatureNormalize<float> F(Utilities::matrixFixture(mat_data_));
    ublas::vector<float> means = F.mu();
    ASSERT_EQ(2, means.size());
    ASSERT_FLOAT_EQ(2*means(0), means(1));
}

TEST_F(FeatureNormalizeTest, SigmaValues) {
    FeatureNormalize<float> F(Utilities::matrixFixture(mat_data_));
    ublas::vector<float> sigma = F.sigma();
    ASSERT_EQ(2, sigma.size());
    ASSERT_FLOAT_EQ(2*sigma(0), sigma(1));
}

TEST_F(FeatureNormalizeTest, Restore) {
    viennacl::matrix<float> original = Utilities::matrixFixture(mat_data_);
    FeatureNormalize<float> F(original);
    ublas::matrix<float> normalized = F.normalize();
    viennacl::matrix<float> normalized_v(normalized.size1(), normalized.size2());
    copy(normalized, normalized_v);
    viennacl::matrix<float> restored = F.restore(normalized_v);
    ASSERT_EQ(original.size1(), restored.size1());
    ASSERT_EQ(original.size2(), restored.size2());
    for (unsigned i = 0; i < original.size1(); ++i)
        for (unsigned j = 0; j < original.size2(); ++j)
            ASSERT_NEAR(original(i, j), restored(i, j), 1e-6f);
}
