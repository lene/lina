#include "LogisticCostFunction.h"
#include "Utilities.h"
#include "FileReader.h"
#include "FeatureNormalize.h"

#include "viennacl/linalg/inner_prod.hpp"
#include <gtest/gtest.h>

#define DEBUG_LOGISTIC_REGRESSION 1

class LogisticCostFunctionTest : public ::testing::Test {
protected:
    const std::string X_ = "3 2 1 1 1 0 1 0";
    const std::string y_ = "3 1 0 0";
    const std::string X_from_course_ = "100 2 \
34.62365962451697 78.0246928153624   30.28671076822607 43.89499752400101 \
35.84740876993872 72.90219802708364  60.18259938620976 86.30855209546826 \
79.0327360507101 75.3443764369103    45.08327747668339 56.3163717815305  \
61.10666453684766 96.51142588489624  75.02474556738889 46.55401354116538 \
76.09878670226257 87.42056971926803  84.43281996120035 43.53339331072109 \
95.86155507093572 38.22527805795094  75.01365838958247 30.60326323428011 \
82.30705337399482 76.48196330235604  69.36458875970939 97.71869196188608 \
39.53833914367223 76.03681085115882  53.9710521485623 89.20735013750205  \
69.07014406283025 52.74046973016765  67.94685547711617 46.67857410673128 \
70.66150955499435 92.92713789364831  76.97878372747498 47.57596364975532 \
67.37202754570876 42.83843832029179  89.67677575072079 65.79936592745237 \
50.534788289883 48.85581152764205    34.21206097786789 44.20952859866288 \
77.9240914545704 68.9723599933059    62.27101367004632 69.95445795447587 \
80.1901807509566 44.82162893218353   93.114388797442 38.80067033713209   \
61.83020602312595 50.25610789244621  38.78580379679423 64.99568095539578 \
61.379289447425 72.80788731317097    85.40451939411645 57.05198397627122 \
52.10797973193984 63.12762376881715  52.04540476831827 69.43286012045222 \
40.23689373545111 71.16774802184875  54.63510555424817 52.21388588061123 \
33.91550010906887 98.86943574220611  64.17698887494485 80.90806058670817 \
74.78925295941542 41.57341522824434  34.1836400264419 75.2377203360134   \
83.90239366249155 56.30804621605327  51.54772026906181 46.85629026349976 \
94.44336776917852 65.56892160559052  82.36875375713919 40.61825515970618 \
51.04775177128865 45.82270145776001  62.22267576120188 52.06099194836679 \
77.19303492601364 70.45820000180959  97.77159928000232 86.7278223300282  \
62.07306379667647 96.76882412413983  91.56497449807442 88.69629254546599 \
79.94481794066932 74.16311935043758  99.2725269292572 60.99903099844988  \
90.54671411399852 43.39060180650027  34.52451385320009 60.39634245837173 \
50.2864961189907 49.80453881323059   49.58667721632031 59.80895099453265 \
97.64563396007767 68.86157272420604  32.57720016809309 95.59854761387875 \
74.24869136721598 69.82457122657193  71.79646205863379 78.45356224515052 \
75.3956114656803 85.75993667331619   35.28611281526193 47.02051394723416 \
56.25381749711624 39.26147251058019  30.05882244669796 49.59297386723685 \
44.66826172480893 66.45008614558913  66.56089447242954 41.09209807936973 \
40.45755098375164 97.53518548909936  49.07256321908844 51.88321182073966 \
80.27957401466998 92.11606081344084  66.74671856944039 60.99139402740988 \
32.72283304060323 43.30717306430063  64.0393204150601 78.03168802018232  \
72.34649422579923 96.22759296761404  60.45788573918959 73.09499809758037 \
58.84095621726802 75.85844831279042  99.82785779692128 72.36925193383885 \
47.26426910848174 88.47586499559782  50.45815980285988 75.80985952982456 \
60.45555629271532 42.50840943572217  82.22666157785568 42.71987853716458 \
88.9138964166533 69.80378889835472   94.83450672430196 45.69430680250754 \
67.31925746917527 66.58935317747915  57.23870631569862 59.51428198012956 \
80.36675600171273 90.96014789746954  68.46852178591112 85.59430710452014 \
42.0754545384731 78.84478600148043   75.47770200533905 90.42453899753964 \
78.63542434898018 96.64742716885644  52.34800398794107 60.76950525602592 \
94.09433112516793 77.15910509073893  90.44855097096364 87.50879176484702 \
55.48216114069585 35.57070347228866  74.49269241843041 84.84513684930135 \
89.84580670720979 45.35828361091658  83.48916274498238 48.38028579728175 \
42.2617008099817 87.10385094025457   99.31500880510394 68.77540947206617 \
55.34001756003703 64.9319380069486   74.77589300092767 89.52981289513276 \
";
    const std::string y_from_course_ = "100 \
0 0 0 1 1 0 1 1 1 1 0 0 1 1 0 1 1 0 1 1 0 1 0 0 1 1 1 0 0 0 1 1 0 1 0 0 0 1 0 0 1 0 1 0 0 0 1 1 1 1 \
1 1 1 0 0 0 1 0 1 1 1 0 0 0 0 0 1 0 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 0 1 1 0 1 1 1 1 1 1 1 \
";
};

TEST_F(LogisticCostFunctionTest, GetsCreated) {
    auto cost = Utilities::logisticCostFunctionFixture(X_, y_);
}

TEST_F(LogisticCostFunctionTest, Runs) {
    auto cost = Utilities::logisticCostFunctionFixture(X_, y_);
    cost.cost(Utilities::vectorFixture("2 0 0"));
}

TEST_F(LogisticCostFunctionTest, ReadsCourseData) {
    auto cost = Utilities::logisticCostFunctionFixture(X_from_course_, y_from_course_);
    ASSERT_EQ(100, cost.X().size1());
    ASSERT_EQ(2, cost.X().size2());
}

TEST_F(LogisticCostFunctionTest, Sigmoid) {
    auto cost = Utilities::logisticCostFunctionFixture("1 1 1", "1 1");
    ASSERT_FLOAT_EQ(1./(exp(1)+1), cost.sigmoid(Utilities::vectorFixture("1 -1"))(0));
    ASSERT_FLOAT_EQ(1./2., cost.sigmoid(Utilities::vectorFixture("1 0"))(0));
    ASSERT_FLOAT_EQ(1./(exp(-1)+1), cost.sigmoid(Utilities::vectorFixture("1 1"))(0));
}

float sigmoid(float x) {
    return (float) (1.f/(1.f+exp(-x)));
}

TEST_F(LogisticCostFunctionTest, SimpleData) {
    auto cost = Utilities::logisticCostFunctionFixture("1 1 1", "1 1");
    ASSERT_FLOAT_EQ(sigmoid(0), cost.h_theta(Utilities::vectorFixture("1 0"))(0));
    ASSERT_FLOAT_EQ(sigmoid(1), cost.h_theta(Utilities::vectorFixture("1 1"))(0));
    ASSERT_FLOAT_EQ(
            -log(sigmoid(0)),
            cost.cost(Utilities::vectorFixture("1 0"))
    );
    ASSERT_FLOAT_EQ(
            -log(sigmoid(1)),
            cost.cost(Utilities::vectorFixture("1 1"))
    );
}

TEST_F(LogisticCostFunctionTest, SimpleData2) {
    auto cost = Utilities::logisticCostFunctionFixture("1 1 1", "1 0");
    ASSERT_FLOAT_EQ(sigmoid(0), cost.h_theta(Utilities::vectorFixture("1 0"))(0));
    ASSERT_FLOAT_EQ(sigmoid(1), cost.h_theta(Utilities::vectorFixture("1 1"))(0));
    ASSERT_FLOAT_EQ(
            -log(1-sigmoid(0)),
            cost.cost(Utilities::vectorFixture("1 0"))
    );
    ASSERT_FLOAT_EQ(
            -log(1-sigmoid(1)),
            cost.cost(Utilities::vectorFixture("1 1"))
    );
}

TEST_F(LogisticCostFunctionTest, EvaluatesCourseData) {
    auto cost = Utilities::logisticCostFunctionFixture(X_from_course_, y_from_course_);
    ASSERT_FLOAT_EQ(0.693147, cost.cost(Utilities::vectorFixture("2 0 0")));
}

TEST_F(LogisticCostFunctionTest, GradientCourseData) {
    auto M = FileReader::add_bias_column<float>(Utilities::matrixFixture(X_from_course_));
    auto y = Utilities::vectorFixture(y_from_course_);
    auto cost = LogisticCostFunction<float>(M, y);
    auto grad = cost.gradient(Utilities::vectorFixture("3 0 0 0"));
    ASSERT_EQ(3, grad.size());
    ASSERT_FLOAT_EQ(-0.100000, grad(0));
    ASSERT_FLOAT_EQ(-12.009217, grad(1));
    ASSERT_FLOAT_EQ(-11.262842, grad(2));
}

void debugOptimization(const GradientDescent<float> &grad, const CostFunction<float> &cost) {
    std::cout << "theta min: " << grad.getMinimum() << " cost: " << cost.cost(grad.getMinimum())
              << " iterations: " << grad.getHistory().size() << ": "<< std::endl;
    for (int i = 0; i < std::min(int(grad.getHistory().size()), 10); ++i) std::cout << grad.getHistory()[i].first << ": " << grad.getHistory()[i].second << std::endl;
    std::cout << "   ..." << std::endl;
    for (int i = grad.getHistory().size()-10; i < grad.getHistory().size(); ++i) std::cout << grad.getHistory()[i].first << ": " << grad.getHistory()[i].second << std::endl;
}

TEST_F(LogisticCostFunctionTest, DISABLED_GradientDescentUnnormalized) {

    auto X = Utilities::matrixFixture(X_from_course_);
    auto M = FileReader::add_bias_column<float>(X);
    auto y = Utilities::vectorFixture(y_from_course_);

    auto cost = LogisticCostFunction<float>(M, y);
    auto grad = GradientDescent<float>(cost);
    grad.setLearningRate(0.001);
    grad.setMaxIter(10000);
    grad.setScaleStepUpFactor(1.005);
    grad.setScaleStepDownFactor(1.02);
    grad.setSkipConvergenceTest(true);
    grad.optimize(Utilities::vectorFixture("3 0 0 0"));

    std::cout << "********** UNNORMALIZED **********" << std::endl;
//    debugOptimization(grad, cost);
    auto probability = sigmoid(viennacl::linalg::inner_prod(grad.getMinimum(), Utilities::vectorFixture("3 1 45 85")));
    std::cout << "theta min: " << grad.getMinimum() << " cost: " << cost.cost(grad.getMinimum()) << " probability: " << probability << std::endl;

    ASSERT_NEAR(-25.16, grad.getMinimum()(0), 0.01);
    ASSERT_NEAR(  0.20, grad.getMinimum()(1), 0.01);
    ASSERT_NEAR(  0.20, grad.getMinimum()(2), 0.01);
    ASSERT_NEAR(  0.20, cost.cost(grad.getMinimum()), 0.01);

}

TEST_F(LogisticCostFunctionTest, DISABLED_GradientDescentNormalized) {
    auto X = Utilities::matrixFixture(X_from_course_);
    FeatureNormalize<float> normalize(X);
    auto Xnorm0 = normalize.normalize();
    viennacl::matrix<float> Xnorm(Xnorm0.size1(), Xnorm0.size2());
    copy(Xnorm0, Xnorm);
    auto M = FileReader::add_bias_column<float>(Xnorm);

    auto y = Utilities::vectorFixture(y_from_course_);
    auto cost = LogisticCostFunction<float>(M, y);
    auto grad = GradientDescent<float>(cost);
    grad.setLearningRate(1);
    grad.optimize(Utilities::vectorFixture("3 0 0 0"));
    std::cout << "********** NORMALIZED **********" << std::endl;
    debugOptimization(grad, cost);
}

TEST_F(LogisticCostFunctionTest, AdmissionProbability) {
    auto X = Utilities::matrixFixture(X_from_course_);
    auto M = FileReader::add_bias_column<float>(X);
    auto y = Utilities::vectorFixture(y_from_course_);

    auto cost = LogisticCostFunction<float>(M, y);
    auto grad = GradientDescent<float>(cost);
    grad.setLearningRate(0.001);
    grad.optimize(Utilities::vectorFixture("3 0 0 0"));

    auto theta = grad.getMinimum();
    auto probability = sigmoid(viennacl::linalg::inner_prod(theta, Utilities::vectorFixture("3 1 45 85")));
    std::cout << probability << std::endl;
}

TEST_F(LogisticCostFunctionTest, GradientDescentSimple1) {
    auto cost = Utilities::logisticCostFunctionFixture("1 1 1", "1 0");
    auto grad = GradientDescent<float>(cost);
    grad.optimize(Utilities::vectorFixture("1 0"));
    // optimal theta should be minus infinity, but due to the flat function let's say it's < -10.
    ASSERT_LT(grad.getMinimum()(0), -10);
    ASSERT_NEAR(cost.cost(grad.getMinimum()), 0, 1e-6);
}

TEST_F(LogisticCostFunctionTest, GradientDescentSimple2) {
    auto cost = Utilities::logisticCostFunctionFixture("1 1 1", "1 1");
    auto grad = GradientDescent<float>(cost);
    grad.optimize(Utilities::vectorFixture("1 0"));
    // optimal theta should be infinity, but due to the flat function let's say it's > 10.
    ASSERT_GT(grad.getMinimum()(0), 10);
    ASSERT_NEAR(cost.cost(grad.getMinimum()), 0, 1e-6);
}

TEST_F(LogisticCostFunctionTest, GradientDescentSimple3) {
    auto cost = Utilities::logisticCostFunctionFixture("2 1 0 1", "2 0 1");
    auto grad = GradientDescent<float>(cost);
    grad.optimize(Utilities::vectorFixture("1 0"));
//    debugOptimization(grad, cost);

    // optimal theta should be infinity, but due to the flat function let's say it's > 10.
    ASSERT_GT(grad.getMinimum()(0), 10);
//    ASSERT_NEAR(cost(grad.getMinimum()), 0, 1e-6);
}

TEST_F(LogisticCostFunctionTest, GradientDescentSimple4) {
    auto cost = Utilities::logisticCostFunctionFixture("2 1 0 1", "2 1 0");
    auto grad = GradientDescent<float>(cost);
    grad.optimize(Utilities::vectorFixture("1 0"));
//    debugOptimization(grad, cost);

    // optimal theta should be minus infinity, but due to the flat function let's say it's < -10.
    ASSERT_LT(grad.getMinimum()(0), -10);
//    ASSERT_NEAR(cost(grad.getMinimum()), 0, 1e-6);
}
