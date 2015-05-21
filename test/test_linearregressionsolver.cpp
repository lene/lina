#include "RegressionSolver.h"
#include "Utilities.h"

#include <gtest/gtest.h>

class LinearRegressionSolverTest : public ::testing::Test {
protected:
    const std::string simple_X_ = "3 2 1 1 .1 0 1 0";
    const std::string simple_y_ = "3 1 0 0";

    const std::string coursera_X_ = "47 2 \
2104 3  1600 3  2400 3  1416 2  3000 4 \
1985 4  1534 3  1427 3  1380 3  1494 3 \
1940 4  2000 3  1890 3  4478 5  1268 3 \
2300 4  1320 2  1236 3  2609 4  3031 4 \
1767 3  1888 2  1604 3  1962 4  3890 3 \
1100 3  1458 3  2526 3  2200 3  2637 3 \
1839 2  1000 1  2040 4  3137 3  1811 4 \
1437 3  1239 3  2132 4  4215 4  2162 4 \
1664 2  2238 3  2567 4  1200 3   852 2 \
1852 4  1203 3";
    const std::string coursera_y_ ="47 \
399900 329900 369000 232000 539900 299900 314900 198999 212000 242500 239999 347000 329999 699900 259900 449900 299900 199900 499998 599000 252900 255000 242900 259900 573900 249900 464500 469000 475000 299900 349900 169900 314900 579900 285900 249900 229900 345000 549000 287000 368500 329900 314000 299000 179900 299900 239500";
    const float thetamin_[3] = { 340413., 109448., -6578.35 };
};

TEST_F(LinearRegressionSolverTest, ConvergesOnSimpleSystem) {
    auto solver = Utilities::linearRegressionSolverFixture(simple_X_, simple_y_);
    ASSERT_TRUE(solver.optimize(Utilities::vectorFixture("3 0 0 0")));
}

TEST_F(LinearRegressionSolverTest, SolvesSimpleSystem) {
    auto solver = Utilities::linearRegressionSolverFixture(simple_X_, simple_y_);
    solver.optimize(Utilities::vectorFixture("3 0 0 0"));
    auto theta = solver.minTheta();
    ASSERT_EQ(3, theta.size());
    for (unsigned i = 0; i < 2; ++i)
            ASSERT_NEAR(Utilities::vectorFixture("3 0.333 0 0.471")(i), theta(i), 1e-3);
}

TEST_F(LinearRegressionSolverTest, SimpleSystemConvergesToCostZero) {
    auto solver = Utilities::linearRegressionSolverFixture(simple_X_, simple_y_);
    solver.optimize(Utilities::vectorFixture("3 0 0 0"));
    auto theta = solver.minTheta();
    ASSERT_EQ(0, solver(theta));
}

TEST_F(LinearRegressionSolverTest, CourseraData) {
    auto solver = Utilities::linearRegressionSolverFixture(coursera_X_, coursera_y_);
    solver.optimize(Utilities::vectorFixture("3 0 0 0"));
    auto theta = solver.minTheta();
    for (auto i = 0; i < 3; ++i)
        ASSERT_NEAR(thetamin_[i], theta(i), 1.);
}
