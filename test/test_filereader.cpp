//
// Created by lene on 05.05.15.
//


#include "FileReader.h"

#include <sstream>

#include <gtest/gtest.h>
#include <MatrixPrinter.h>

class FileReaderTest: public ::testing::Test {
protected:
    std::string filename_ = "";
};

TEST_F(FileReaderTest, WorksAtAll) {
    ASSERT_THROW(FileReader::read_matrix<float>(filename_), std::invalid_argument);
}

TEST_F(FileReaderTest, WorksAtAllForStream) {
    std::stringstream stream("1 1");
    FileReader::read_matrix<float>(stream);
}

TEST_F(FileReaderTest, ReadsCorrectMatrixSize) {
    std::stringstream stream("3 2");
    viennacl::matrix<float> M = FileReader::read_matrix<float>(stream);
    ASSERT_EQ(3, M.size1());
    ASSERT_EQ(2, M.size2());
}

TEST_F(FileReaderTest, ReadsCorrectMatrixData) {
    std::stringstream stream("3 2\n0.1 0.2\n1.0 2.0\n10.0 20.0");
    viennacl::matrix<float> M = FileReader::read_matrix<float>(stream);
    ASSERT_FLOAT_EQ(0.1, M(0,0));
    ASSERT_FLOAT_EQ(0.2, M(0,1));
    ASSERT_FLOAT_EQ(1.0, M(1,0));
    ASSERT_FLOAT_EQ(2.0, M(1,1));
    ASSERT_FLOAT_EQ(10.0, M(2,0));
    ASSERT_FLOAT_EQ(20.0, M(2,1));
}

TEST_F(FileReaderTest, ReadsCorrectVectorSize) {
    std::stringstream stream("3");
    viennacl::vector<float> v = FileReader::read_vector<float>(stream);
    ASSERT_EQ(3, v.size());
}

TEST_F(FileReaderTest, ReadsCorrectVectorData) {
    std::stringstream stream("3\n0.1 0.2 0.3");
    viennacl::vector<float> v = FileReader::read_vector<float>(stream);
    ASSERT_FLOAT_EQ(0.1, v(0));
    ASSERT_FLOAT_EQ(0.2, v(1));
    ASSERT_FLOAT_EQ(0.3, v(2));
}
