//
// Created by lene on 05.05.15.
//

#include <gtest/gtest.h>

TEST(DummyTest, WorksAtAll) {
    ASSERT_TRUE(true);
    ASSERT_FALSE(false);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
