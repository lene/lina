//
// Created by lene on 05.05.15.
//


#include "FileReader.h"

#include <gtest/gtest.h>

class FileReaderTest: public ::testing::Test {

};

TEST_F(FileReaderTest, WorksAtAll) {
    FileReader::X<float>(std::string(""));
}
