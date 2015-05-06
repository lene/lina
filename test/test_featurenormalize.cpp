//
// Created by lene on 06.05.15.
//

#include "FeatureNormalize.h"
#include "FileReader.h"

#include <gtest/gtest.h>

class FeatureNormalizeTest: public ::testing::Test {
protected:
    FeatureNormalizeTest(): mat_stream_(mat_data_) { }
    const std::string mat_data_ = "3 2\n0.1 0.2\n1.0 2.0\n10.0 20.0";
    std::stringstream mat_stream_;
};

TEST_F(FeatureNormalizeTest, RunsAtAll) {
    viennacl::matrix<float> M = FileReader::read_matrix<float>(mat_stream_);
    FeatureNormalize<float> F(M);
}

TEST_F(FeatureNormalizeTest, NormalizedMatrix) {
    viennacl::matrix<float> M = FileReader::read_matrix<float>(mat_stream_);
    MatrixPrinter<viennacl::matrix<float>> po(M);
    po.print();
    FeatureNormalize<float> F(M);
    ublas::matrix<float> normalized = F.normalize();
    MatrixPrinter<ublas::matrix<float>> pn(normalized);
    pn.print();
}
