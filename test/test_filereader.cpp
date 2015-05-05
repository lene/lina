//
// Created by lene on 05.05.15.
//


#include "FileReader.h"

#include <gtest/gtest.h>

class FileReaderTest: public ::testing::Test {
protected:
    std::string filename_ = "";
};

TEST_F(FileReaderTest, WorksAtAll) {
    FileReader::read_matrix<float>(filename_);
}
