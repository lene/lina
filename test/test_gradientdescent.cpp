//
// Created by lene on 07.05.15.
//

#include "GradientDescent.h"
#include "Utilities.h"

#include <gtest/gtest.h>

class GradDescTest: public ::testing::Test {
protected:

};

TEST_F(GradDescTest, Initializes) {
    CostFunction<float> cost = Utilities::costFunctionFixture("3 2 1 1 1 0 1 0", "3 1 0 0");
    GradientDescent<float> grad(cost);
}