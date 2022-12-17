//
// Created by Erik Hyrkas on 10/25/2022.
//


#include <iostream>
#include <string>
#include "../types/tensor.hpp"
#include "../util/tensor_utils.hpp"
#include "../util/unit_test.hpp"
#include "../util/tensor_stats.hpp"
#include "../util/timers.hpp"

// Super slow on my machine, but needed to test everything. Probably not useful for day-to-day unit tests.
//#define FULL_TENSOR_TESTS

using namespace microml;
using namespace std;

void productTest() {
    auto matrixFunc = [](size_t row, size_t col, size_t channel) { return (float) (row + 1); };
    auto matrix = make_shared<TensorFromFunction>(matrixFunc, 3, 3, 1);
//    matrix->print();
//    1.000, 1.000, 1.000
//    2.000, 2.000, 2.000
//    3.000, 3.000, 3.000
    ASSERT_TRUE(216.0f == matrix->product());
}

void randomTest() {
    // This isn't a great test, since TensorFromRandomRepeatable is really only useful
    // in a single threaded situation where we want the results to be repeatable.
    // But the moment you start reading from it concurrently, it's no longer repeatable.
    //
    // The only reason I made this was that I may want to have the framework support running
    // in a repeatable mode (which would require running in a single thread.) This seems
    // like it would be terribly slow and make the framework near useless. The alternative
    // is to not use a view and persist all the random values on creation, which works
    // but would require gigabytes of memory for a matrix of any significant size.
    auto matrix1 = make_unique<TensorFromRandom>(2, 2, 1, -1000.0f, 1000.0f, 42);
//    matrix1->print();
    auto matrix2 = make_unique<TensorFromRandom>(2, 2, 1, -1000.0f, 1000.0f, 42);
//    matrix2->print();
    ASSERT_TRUE(matrix1->getValue(0, 0, 0) == matrix2->getValue(0, 0, 0));
    ASSERT_TRUE(matrix1->getValue(0, 1, 0) == matrix2->getValue(0, 1, 0));
    ASSERT_TRUE(matrix1->getValue(1, 0, 0) == matrix2->getValue(1, 0, 0));
    ASSERT_TRUE(matrix1->getValue(1, 1, 0) == matrix2->getValue(1, 1, 0));
    auto matrix3 = make_unique<TensorFromRandom>(2, 2, 1, -1000.0f, 1000.0f, 99);
    auto matrix4 = make_unique<TensorFromRandom>(2, 2, 1, -1000.0f, 1000.0f, 99);
    ASSERT_TRUE(matrix3->getValue(0, 0, 0) == matrix4->getValue(0, 0, 0));
    ASSERT_TRUE(matrix3->getValue(0, 1, 0) == matrix4->getValue(0, 1, 0));
    ASSERT_TRUE(matrix3->getValue(1, 0, 0) == matrix4->getValue(1, 0, 0));
    ASSERT_TRUE(matrix3->getValue(1, 1, 0) == matrix4->getValue(1, 1, 0));
    for (int i = 0; i < 200; i++) {
        // tested with much larger matrices, but it's too slow to leave in for day-to-day testing
        auto matrix5 = make_unique<TensorFromRandom>(100, 100, 1, -1000.0f, 1000.0f, i);
        float mean = std::abs(matrix5->arithmeticMean());
//        cout << "Seed: " << i << " Abs Mean: " << std::fixed << std::setprecision(38) << mean << endl;
        ASSERT_TRUE(mean < 1.0f);
    }
}

void sumTest() {
    auto matrixFunc = [](size_t row, size_t col, size_t channel) { return (float) (row + 1); };
    auto matrix = make_shared<TensorFromFunction>(matrixFunc, 3, 3, 1);
//    matrix->print();
//    1.000, 1.000, 1.000
//    2.000, 2.000, 2.000
//    3.000, 3.000, 3.000
    ASSERT_TRUE(18.0f == matrix->sum());
}

void assignSmallTest() {
    auto matrixRandom = make_shared<TensorFromRandom>(101, 103, 1, 4);
//    matrix_random->print();
    auto matrix = make_shared<QuarterTensor>(matrixRandom, 4);
//    matrix->print();
//    cout << "0, 0 original random: " << matrix_random->get_val(0, 0, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(0, 0, 0), 4, 0), 4, 0) << endl;
//    cout << "0, 0 new matrix: " << matrix->get_val(0, 0, 0) << endl;
//    cout << "5, 4 original random: " << matrix_random->get_val(5, 4, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(5, 4, 0), 4, 0), 4, 0) << endl;
//    cout << "5, 4 new matrix: " << matrix->get_val(5, 4, 0) << endl;
//    cout << "12, 10 original random: " << matrix_random->get_val(12, 10, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(12, 10, 0), 4, 0), 4, 0) << endl;
//    cout << "12, 10 new matrix: " << matrix->get_val(12, 10, 0) << endl;
//    cout << "50, 10 original random: " << matrix_random->get_val(50, 10, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(50, 10, 0), 4, 0), 4, 0) << endl;
//    cout << "50, 10 new matrix: " << matrix->get_val(50, 10, 0) << endl;
//    cout << "99, 99 original random: " << matrix_random->get_val(99, 99, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(99, 99, 0), 4, 0), 4, 0) << endl;
//    cout << "99, 99 new matrix: " << matrix->get_val(99, 99, 0) << endl;
//
//    cout << "99, 99 original random: " << matrix_random->get_val(100, 102, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(100, 102, 0), 4, 0), 4, 0) << endl;
//    cout << "99, 99 new matrix: " << matrix->get_val(100, 102, 0) << endl;

    ASSERT_TRUE(matrix->getValue(0, 0, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(0, 0, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(5, 4, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(5, 4, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(12, 10, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(12, 10, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(50, 10, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(50, 10, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(99, 99, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(99, 99, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(100, 102, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(100, 102, 0), 4), 4));
}


void assignMediumTest() {
    auto matrixRandom = make_shared<TensorFromRandom>(1001, 10003, 1, 4);
//    matrix_random->print();
    auto matrix = make_shared<QuarterTensor>(matrixRandom, 4);
//    matrix->print();
//    cout << "0, 0 original random: " << matrix_random->get_val(0, 0, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(0, 0, 0), 4, 0), 4, 0) << endl;
//    cout << "0, 0 new matrix: " << matrix->get_val(0, 0, 0) << endl;
//    cout << "5, 4 original random: " << matrix_random->get_val(5, 4, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(5, 4, 0), 4, 0), 4, 0) << endl;
//    cout << "5, 4 new matrix: " << matrix->get_val(5, 4, 0) << endl;
//    cout << "12, 10 original random: " << matrix_random->get_val(12, 10, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(12, 10, 0), 4, 0), 4, 0) << endl;
//    cout << "12, 10 new matrix: " << matrix->get_val(12, 10, 0) << endl;
//    cout << "50, 10 original random: " << matrix_random->get_val(50, 10, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(50, 10, 0), 4, 0), 4, 0) << endl;
//    cout << "50, 10 new matrix: " << matrix->get_val(50, 10, 0) << endl;
//    cout << "99, 99 original random: " << matrix_random->get_val(99, 99, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(99, 99, 0), 4, 0), 4, 0) << endl;
//    cout << "99, 99 new matrix: " << matrix->get_val(99, 99, 0) << endl;
//    cout << "1000, 10002 original random: " << matrix_random->get_val(1000, 10002, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(1000, 10002, 0), 4, 0), 4, 0) << endl;
//    cout << "1000, 10002 new matrix: " << matrix->get_val(1000, 10002, 0) << endl;
    ASSERT_TRUE(matrix->getValue(0, 0, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(0, 0, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(5, 4, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(5, 4, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(12, 10, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(12, 10, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(50, 10, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(50, 10, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(99, 99, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(99, 99, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(1000, 10002, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(1000, 10002, 0), 4), 4));
}

void assignLargeTest() {
    auto matrixRandom = make_shared<TensorFromRandom>(200000, 200001, 1, 4);
    auto matrix = make_shared<QuarterTensor>(matrixRandom, 4);
    cout << "0, 0 original random: " << matrixRandom->getValue(0, 0, 0) << " quantized: "
         << quarterToFloat(floatToQuarter(matrixRandom->getValue(0, 0, 0), 4), 4) << endl;
    cout << "0, 0 new matrix: " << matrix->getValue(0, 0, 0) << endl;
    cout << "199999, 199999 original random: " << matrixRandom->getValue(199999, 199999, 0) << " quantized: "
         << quarterToFloat(floatToQuarter(matrixRandom->getValue(199999, 199999, 0), 4), 4) << endl;
    cout << "199999, 199999 new matrix: " << matrix->getValue(199999, 199999, 0) << endl;
    ASSERT_TRUE(matrix->getValue(0, 0, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(0, 0, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(100, 50, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(100, 50, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(50, 10, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(50, 10, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(9000, 10000, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(9000, 10000, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(1, 183784, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(1, 183784, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(180034, 1, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(180034, 1, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(162341, 44228, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(162341, 44228, 0), 4), 4));
    ASSERT_TRUE(matrix->getValue(199999, 199999, 0) ==
                quarterToFloat(floatToQuarter(matrixRandom->getValue(199999, 199999, 0), 4), 4));
}

void reshapeTest() {
    auto matrixRandom = make_shared<TensorFromRandom>(1, 5, 1, 4);
//    matrix_random->print();
    auto matrix = make_shared<QuarterTensor>(matrixRandom, 4);
//    matrix->print();
    auto reshape = make_shared<TensorReshapeView>(matrixRandom, 5, 1);
    auto other = make_unique<QuarterTensor>(reshape, 4);
    ASSERT_TRUE(matrix->getValue(0, 0, 0) == other->getValue(0, 0, 0));
    ASSERT_TRUE(matrix->getValue(0, 1, 0) == other->getValue(1, 0, 0));
    ASSERT_TRUE(matrix->getValue(0, 2, 0) == other->getValue(2, 0, 0));
    ASSERT_TRUE(matrix->getValue(0, 3, 0) == other->getValue(3, 0, 0));
    ASSERT_TRUE(matrix->getValue(0, 4, 0) == other->getValue(4, 0, 0));
    auto otherView = make_unique<TensorReshapeView>(matrix, 5, 1);
    ASSERT_TRUE(otherView->getValue(0, 0, 0) == other->getValue(0, 0, 0));
    ASSERT_TRUE(otherView->getValue(1, 0, 0) == other->getValue(1, 0, 0));
    ASSERT_TRUE(otherView->getValue(2, 0, 0) == other->getValue(2, 0, 0));
    ASSERT_TRUE(otherView->getValue(3, 0, 0) == other->getValue(3, 0, 0));
    ASSERT_TRUE(otherView->getValue(4, 0, 0) == other->getValue(4, 0, 0));
}

void testCreate() {
    auto matrix = make_unique<QuarterTensor>(2, 2, 1, 4);
    ASSERT_TRUE(2 == matrix->rowCount());
    ASSERT_TRUE(2 == matrix->columnCount());
}

void testScalarMultiplication() {
    auto matrixFunc = [](size_t row, size_t col, size_t channel) { return (float) row; };
    auto matrix = make_shared<TensorFromFunction>(matrixFunc, 5, 5, 1);
//    matrix->print();
    auto scaledMatrix = make_unique<TensorMultiplyByScalarView>(matrix, 6.0f);
//    scaled_matrix->print();
    ASSERT_TRUE(0 == scaledMatrix->getValue(0, 0, 0));
    ASSERT_TRUE(0 == scaledMatrix->getValue(0, 2, 0));
    ASSERT_TRUE(0 == scaledMatrix->getValue(0, 4, 0));
    ASSERT_TRUE(12 == scaledMatrix->getValue(2, 0, 0));
    ASSERT_TRUE(12 == scaledMatrix->getValue(2, 2, 0));
    ASSERT_TRUE(12 == scaledMatrix->getValue(2, 4, 0));
    ASSERT_TRUE(24 == scaledMatrix->getValue(4, 0, 0));
    ASSERT_TRUE(24 == scaledMatrix->getValue(4, 2, 0));
    ASSERT_TRUE(24 == scaledMatrix->getValue(4, 4, 0));
}

void testStackingMultiplyViews() {
    auto matrix_func = [](size_t row, size_t col, size_t channel) {
        return (float) (((float) row * 10.f) + (float) col);
    };
    auto matrix = make_shared<TensorFromFunction>(matrix_func, 5, 5, 1);
    auto times2 = make_shared<TensorMultiplyByScalarView>(matrix, 2.f);
    auto timesHalf = make_shared<TensorMultiplyByScalarView>(times2, 0.5f);
    for (size_t r = 0; r < 5; r++) {
        for (size_t c = 0; c < 5; c++) {
//            cout << matrix->get_val(r, c, 0) << " == " << timesHalf->get_val(r, c, 0) << endl;
            ASSERT_TRUE(roughlyEqual(matrix->getValue(r, c, 0), timesHalf->getValue(r, c, 0)));
        }
    }
}

//void test_matrix_multiplication() {
// random(9)
//[
//-112, -32
//0.4375, -0.203125
//]
//random(62)
//[
//-104, 10
//-2.5, -7.5
//]
//}

void testArithmeticMean() {
    auto matrixFunc = [](size_t row, size_t col, size_t channel) { return (float) row; };
    auto matrix = make_unique<TensorFromFunction>(matrixFunc, 2, 2, 1);
//    matrix->print();
//    0.000, 0.000
//    1.000, 1.000
    ASSERT_TRUE(0.5 == matrix->arithmeticMean());
}

void testGeometricMean() {
    auto matrix = make_unique<TensorFromRandom>(2, 2, 1, -10.0f, 0.0f, 9);
//    matrix->print();
//    -5.236, -6.793
//    -1.519, -3.077
    ASSERT_TRUE(isnan(matrix->geometricMean()));
    auto matrix2 = make_unique<TensorFromRandom>(2, 2, 1, 1.0f, 10.0f, 36);
//    matrix2->print();
//    5.288, 3.886
//    8.004, 6.603
    float gm = matrix2->geometricMean();
//    cout << "GM: " << std::fixed << std::setprecision(38) << gm << endl;
    ASSERT_TRUE(roughlyEqual(5.7405242919921875f, gm));
}

void testBigStats() {
    // 200,000 x 200,000 = 40 billion elements = ~38gb matrix
    auto matrix = make_shared<TensorFromRandom>(200000, 200000, 1, 7);
//    matrix->print();
    auto stats = make_unique<TensorStats>(*matrix, FIT_BIAS_FOR_50);
    stats->print();
    ASSERT_TRUE(8 == stats->getRecommendedBias());
    ASSERT_TRUE(roughlyEqual(-120.0f, stats->getRecommendedOffset()));
}

void testMediumStats() {
    // 50,000 x 50,000 = 2.5 billion elements = ~2gb matrix
    auto matrix = make_shared<TensorFromRandom>(50000, 50000, 1, 14);
//    matrix->print();
    auto stats = make_unique<TensorStats>(*matrix, FIT_BIAS_FOR_50);
    stats->print();
    ASSERT_TRUE(14 == stats->getRecommendedBias());
    ASSERT_TRUE(roughlyEqual(0.0f, stats->getRecommendedOffset()));
}

void testSmallStats() {
    auto matrix = make_shared<TensorFromRandom>(50, 50, 1, 4);
//    matrix->print();
    auto stats = make_unique<TensorStats>(*matrix, FIT_BIAS_FOR_100);
//    stats->print();
    ASSERT_TRUE(4 == stats->getRecommendedBias());
    ASSERT_TRUE(roughlyEqual(-3.16455078125f, stats->getRecommendedOffset()));
}

// You'll notice that our distribution isn't even. The further we get from 0, the less granularity we have, so the
// more data that's grouped together in a bucket.
void testEvenDistributionQuarterMedium() {
    auto matrixFunc = [](size_t row, size_t col, size_t channel) { return ((float) (row + 1)) / 200.0; };
    auto matrix = make_unique<TensorFromFunction>(matrixFunc, 10000, 10000, 1);
//    matrix->print();
    auto stats = make_unique<TensorStats>(*matrix, FIT_BIAS_FOR_80);
//    stats->print();
    ASSERT_TRUE(8 == stats->getRecommendedBias());
    ASSERT_TRUE(roughlyEqual(0.0f, stats->getRecommendedOffset()));
}

void test_even_distribution_quarter_small() {
    auto matrixFunc = [](size_t row, size_t col, size_t channel) { return ((float) (row + 1)) / 200.0; };
    auto matrix = make_unique<TensorFromFunction>(matrixFunc, 350, 350, 1);
//    matrix->print();
    auto stats = make_unique<TensorStats>(*matrix, FIT_BIAS_FOR_80);
//    stats->print();
    ASSERT_TRUE(14 == stats->getRecommendedBias());
    const auto ro = stats->getRecommendedOffset();
//    cout << std::fixed << std::setprecision(38) << ro << endl;
    ASSERT_TRUE(roughlyEqual(0.8424999713897705078125f, ro));
}

void testEvenDistributionQuarterBig() {
    auto matrixFunc = [](size_t row, size_t col, size_t channel) { return ((float) (row + 1)) / 20000.0; };
    auto matrix = make_unique<TensorFromFunction>(matrixFunc, 100000000, 1, 1);
//    matrix->print();
    auto stats = make_unique<TensorStats>(*matrix, FIT_BIAS_FOR_80);
    stats->print();
    ASSERT_TRUE(1 == stats->getRecommendedBias());
    ASSERT_TRUE(roughlyEqual(2431.999756f, stats->getRecommendedOffset()));
}

void testEvenDistributionQuarterHugeNumbers() {
    auto matrixFunc = [](size_t row, size_t col, size_t channel) { return ((float) (row + 1)) * 500.0; };
    auto matrix = make_unique<TensorFromFunction>(matrixFunc, 1000, 1, 1);
//    matrix->print();
    auto stats = make_unique<TensorStats>(*matrix, FIT_BIAS_FOR_80);
//    stats->print();
    ASSERT_TRUE(-4 == stats->getRecommendedBias());
    ASSERT_TRUE(roughlyEqual(237500.0f, stats->getRecommendedOffset()));
}

void testConstant() {
    auto matrix = make_shared<UniformTensor>(10, 10, 1, 0.0f);
    for (size_t i = 0; i < matrix->rowCount(); i++) {
        for (size_t j = 0; j < matrix->columnCount(); j++) {
            ASSERT_TRUE(0.0f == matrix->getValue(i, j, 0));
        }
    }
//    matrix->print();
}


void testIdentity() {
    auto matrix = make_shared<IdentityTensor>(10, 10, 1);
    for (size_t i = 0; i < matrix->rowCount(); i++) {
        for (size_t j = 0; j < matrix->columnCount(); j++) {
            if (i == j) {
                ASSERT_TRUE(1.0f == matrix->getValue(i, j, 0));
            } else {
                ASSERT_TRUE(0.0f == matrix->getValue(i, j, 0));
            }
        }
    }
//    matrix->print();
}

void testDotProduct() {
    auto matrixFunc = [](size_t row, size_t col, size_t channel) { return ((float) (row + col)); };
    auto matrix1 = make_shared<TensorFromFunction>(matrixFunc, 2, 2, 1);
//    matrix_1->print();
    auto matrix2 = make_shared<TensorFromFunction>(matrixFunc, 2, 2, 1);
    auto dotProductView = make_shared<TensorDotTensorView>(matrix1, matrix2);
//    dot_product_view->print();
    ASSERT_TRUE(1.0f == dotProductView->getValue(0, 0, 0));
    ASSERT_TRUE(2.0f == dotProductView->getValue(0, 1, 0));
    ASSERT_TRUE(2.0f == dotProductView->getValue(1, 0, 0));
    ASSERT_TRUE(5.0f == dotProductView->getValue(1, 1, 0));
}

void testDotProduct2() {
    vector<vector<float>> a = {{4, 2},
                               {0, 3}};
    auto matrix1 = make_shared<QuarterTensor>(a, 8);
//    matrix_1->print();
    vector<vector<float>> b = {{4, 0},
                               {1, 4}};
    auto matrix2 = make_shared<QuarterTensor>(b, 8);
    auto dotProductView = make_shared<TensorDotTensorView>(matrix1, matrix2);
//    dot_product_view->print();
    ASSERT_TRUE(18.0f == dotProductView->getValue(0, 0, 0));
    ASSERT_TRUE(8.0f == dotProductView->getValue(0, 1, 0));
    ASSERT_TRUE(3.0f == dotProductView->getValue(1, 0, 0));
    ASSERT_TRUE(12.0f == dotProductView->getValue(1, 1, 0));
}

void testDotProduct3() {
    vector<vector<float>> a = {{2, 2},
                               {0, 3},
                               {0, 4}};
    auto matrix1 = make_shared<QuarterTensor>(a, 8);
//    matrix_1->print();
    vector<vector<float>> b = {{2, 1, 2},
                               {3, 2, 4}};
    auto matrix2 = make_shared<QuarterTensor>(b, 8);
    auto dotProductView = make_shared<TensorDotTensorView>(matrix1, matrix2);
//    dot_product_view->print();
    ASSERT_TRUE(10.0f == dotProductView->getValue(0, 0, 0));
    ASSERT_TRUE(6.0f == dotProductView->getValue(0, 1, 0));
    ASSERT_TRUE(12.0f == dotProductView->getValue(0, 2, 0));
    ASSERT_TRUE(9.0f == dotProductView->getValue(1, 0, 0));
    ASSERT_TRUE(6.0f == dotProductView->getValue(1, 1, 0));
    ASSERT_TRUE(12.0f == dotProductView->getValue(1, 2, 0));
    ASSERT_TRUE(12.0f == dotProductView->getValue(2, 0, 0));
    ASSERT_TRUE(8.0f == dotProductView->getValue(2, 1, 0));
    ASSERT_TRUE(16.0f == dotProductView->getValue(2, 2, 0));
}

void testDotProduct4() {
    vector<vector<float>> a = {{1, 2, 3},
                               {4, 5, 6}};
    auto matrix1 = make_shared<QuarterTensor>(a, 8);
//    matrix_1->print();
    vector<vector<float>> b = {{7,  8},
                               {9,  10},
                               {11, 12}};
    auto matrix2 = make_shared<QuarterTensor>(b, 8);
    auto dotProductView = make_shared<TensorDotTensorView>(matrix1, matrix2);
//    dot_product_view->print();
    ASSERT_TRUE(58.0f == dotProductView->getValue(0, 0, 0));
    ASSERT_TRUE(64.0f == dotProductView->getValue(0, 1, 0));
    ASSERT_TRUE(139.0f == dotProductView->getValue(1, 0, 0));
    ASSERT_TRUE(154.0f == dotProductView->getValue(1, 1, 0));
}

void testMatrixAddition() {
    vector<vector<float>> a = {{-1, 2,  3},
                               {2,  -3, 1},
                               {3,  1,  -2}};
    auto matrix1 = make_shared<QuarterTensor>(a, 8);
//    matrix_1->print();
    vector<vector<float>> b = {{3, -1, 2},
                               {1, 0,  3},
                               {2, -1, 0}};
    auto matrix2 = make_shared<QuarterTensor>(b, 8);
    auto addView = make_shared<TensorAddTensorView>(matrix1, matrix2);
    ASSERT_TRUE(2.0f == addView->getValue(0, 0, 0));
    ASSERT_TRUE(1.0f == addView->getValue(0, 1, 0));
    ASSERT_TRUE(5.0f == addView->getValue(0, 2, 0));
    ASSERT_TRUE(3.0f == addView->getValue(1, 0, 0));
    ASSERT_TRUE(-3.0f == addView->getValue(1, 1, 0));
    ASSERT_TRUE(4.0f == addView->getValue(1, 2, 0));
    ASSERT_TRUE(5.0f == addView->getValue(2, 0, 0));
    ASSERT_TRUE(0.0f == addView->getValue(2, 1, 0));
    ASSERT_TRUE(-2.0f == addView->getValue(2, 2, 0));
}

void testPixel() {
    auto matrix = make_shared<TensorFromRandom>(5, 5, 1, 0.f, 1.f, 42);
    auto pixel_test = make_shared<PixelTensor>(matrix);
    pixel_test->print();
    matrix->print();
    // todo: assert rather than print.
}

void testZeroPaddedView() {
    auto matrix = make_shared<UniformTensor>(1, 1, 1, 5.f);
    auto padded = make_shared<TensorZeroPaddedView>(matrix, 2, 2, 2, 2);
//    padded->print();
    ASSERT_TRUE(5 == padded->rowCount());
    ASSERT_TRUE(5 == padded->columnCount());
    ASSERT_TRUE(1 == padded->channelCount());
    ASSERT_TRUE(5 == padded->getValue(2, 2, 0));
    ASSERT_TRUE(0 == padded->getValue(0, 0, 0));
    ASSERT_TRUE(0 == padded->getValue(4, 4, 0));
}

void testZeroPaddedView2() {
    auto matrix = make_shared<UniformTensor>(2, 3, 1, 5.f);
    auto padded = make_shared<TensorZeroPaddedView>(matrix, 1, 1, 2, 2);
//    padded->print();
    ASSERT_TRUE(4 == padded->rowCount());
    ASSERT_TRUE(7 == padded->columnCount());
    ASSERT_TRUE(1 == padded->channelCount());
    ASSERT_TRUE(5 == padded->getValue(2, 2, 0));
    ASSERT_TRUE(0 == padded->getValue(0, 0, 0));
    ASSERT_TRUE(0 == padded->getValue(6, 6, 0));
}

void testFullConvolve2dView() {
    auto matrix1 = make_shared<UniformTensor>(3, 3, 1, 1.f);
    auto matrix2 = make_shared<UniformTensor>(3, 3, 1, 1.f);
    auto conv2d = make_shared<TensorFullConvolve2dView>(matrix1, matrix2);
    matrix1->print();
    conv2d->print();
    ASSERT_TRUE(conv2d->columnCount() == 5);
    ASSERT_TRUE(conv2d->rowCount() == 5);
    ASSERT_TRUE(conv2d->channelCount() == 1);
    //Expected
    //[[1 2 3 2 1]
    //[2 4 6 4 2]
    //[3 6 9 6 3]
    //[2 4 6 4 2]
    //[1 2 3 2 1]]
    vector<vector<vector<float>>> expectedVector = {{{1, 2, 3, 2, 1},
                                                     {2, 4, 6, 4, 2},
                                                     {3, 6, 9, 6, 3},
                                                     {2, 4, 6, 4, 2},
                                                     {1, 2, 3, 2, 1}}};
    auto expected = make_shared<FullTensor>(expectedVector);
    assertEqual(expected, conv2d);
}


void test2FullConvolve2dView() {

    vector<vector<vector<float>>> a = {{{1, 2, 3},
                                        {4, 5, 6},
                                        {7, 8, 9}}};
    auto matrix1 = make_shared<FullTensor>(a);
    vector<vector<vector<float>>> b = {{{10, 11, 12},
                                        {13, 14, 15},
                                        {16, 17, 18}}};
    auto matrix2 = make_shared<FullTensor>(b);
    auto conv2d = make_shared<TensorFullConvolve2dView>(matrix1, matrix2);
//    matrix1->print();
//    conv2d->print();
    ASSERT_TRUE(conv2d->columnCount() == 5);
    ASSERT_TRUE(conv2d->rowCount() == 5);
    ASSERT_TRUE(conv2d->channelCount() == 1);
    //Expected
    // [[ 10  31  64  57  36]
    //  [ 53 134 245 198 117]
    //  [138 327 570 441 252]
    //  [155 350 587 438 243]
    //  [112 247 406 297 162]]
    vector<vector<vector<float>>> expectedVector = {{{10, 31, 64, 57, 36},
                                                     {53, 134, 245, 198, 117},
                                                     {138, 327, 570, 441, 252},
                                                     {155, 350, 587, 438, 243},
                                                     {112, 247, 406, 297, 162}}};
    auto expected = make_shared<FullTensor>(expectedVector);
    assertEqual(expected, conv2d);
}

void testFullCrossCorrelation2d() {
    auto matrix1 = make_shared<UniformTensor>(3, 3, 1, 1.f);
    auto matrix2 = make_shared<UniformTensor>(3, 3, 1, 1.f);
    auto padding = (size_t) round(((double) matrix1->rowCount()) / 2.0);
    cout << "Padding: " << padding << " ... " << ((double) matrix1->rowCount() - 1) << " ... "
         << ((double) matrix1->rowCount()) / 2.0 << endl;
    ASSERT_TRUE(2 == padding);
    auto conv2d = make_shared<TensorFullCrossCorrelation2dView>(matrix1, matrix2);
    matrix1->print();
    conv2d->print();

    ASSERT_TRUE(conv2d->columnCount() == 5);
    ASSERT_TRUE(conv2d->rowCount() == 5);
    ASSERT_TRUE(conv2d->channelCount() == 1);
    //Expected
    //[[1 2 3 2 1]
    //[2 4 6 4 2]
    //[3 6 9 6 3]
    //[2 4 6 4 2]
    //[1 2 3 2 1]]
    vector<vector<vector<float>>> expectedVector = {{{1, 2, 3, 2, 1},
                                                     {2, 4, 6, 4, 2},
                                                     {3, 6, 9, 6, 3},
                                                     {2, 4, 6, 4, 2},
                                                     {1, 2, 3, 2, 1}}};
    auto expected = make_shared<FullTensor>(expectedVector);
    assertEqual(expected, conv2d);

}

void testValidCrossCorrelation2d() {
    auto matrix1 = make_shared<UniformTensor>(3, 3, 1, 1.f);
    auto matrix2 = make_shared<UniformTensor>(3, 3, 1, 1.f);
    auto validCrossCorrelation2DView = make_shared<TensorValidCrossCorrelation2dView>(matrix1, matrix2);
    matrix1->print();
    validCrossCorrelation2DView->print();
    //Expected
    //[[9]]
    ASSERT_TRUE(validCrossCorrelation2DView->columnCount() == 1);
    ASSERT_TRUE(validCrossCorrelation2DView->rowCount() == 1);
    ASSERT_TRUE(validCrossCorrelation2DView->channelCount() == 1);
    ASSERT_TRUE(validCrossCorrelation2DView->getValue(0, 0, 0) == 9);
}

void test2ValidCrossCorrelation2d() {
    vector<vector<vector<float>>> a = {{{-0.001, 0.364, 0.529, 0.303, -0.492, -0.367},
                                        {0.443, -0.117, -0.364, -0.280, 0.261, 0.604},
                                        {-0.367, 0.001, 0.366, 0.530, 0.305, -0.490},
                                        {0.533, 0.444, -0.115, -0.361, -0.277, 0.262},
                                        {-0.351, -0.364, 0.004, 0.367, 0.531, 0.307},
                                        {0.372, 0.534, 0.446, -0.113, -0.359, -0.275}}};
    auto matrix1 = make_shared<FullTensor>(a);
    vector<vector<vector<float>>> b = {{{-1.381, -1.227, 0.040, 0.752},
                                        {-0.780, -0.366, -0.907, -1.410},
                                        {-1.094, -1.010, -0.761, -1.337},
                                        {-1.720, -0.312, 0.060, -0.343}}};
    auto matrix2 = make_shared<FullTensor>(b);
    auto validCrossCorrelation2DView = make_shared<TensorValidCrossCorrelation2dView>(matrix1, matrix2);
    matrix1->print();
    validCrossCorrelation2DView->print();
    //Expected
    // [[-1.299014 -3.235515 -2.408694]
    // [-1.356435  0.487984  1.401795]
    // [ 0.470853 -1.322469 -3.257066]]
    ASSERT_TRUE(validCrossCorrelation2DView->columnCount() == 3);
    ASSERT_TRUE(validCrossCorrelation2DView->rowCount() == 3);
    ASSERT_TRUE(validCrossCorrelation2DView->channelCount() == 1);
    ASSERT_TRUE(roughlyEqual(validCrossCorrelation2DView->getValue(0, 0, 0), -1.299014f));
    ASSERT_TRUE(roughlyEqual(validCrossCorrelation2DView->getValue(0, 1, 0), -3.235515f));
    ASSERT_TRUE(roughlyEqual(validCrossCorrelation2DView->getValue(0, 2, 0), -2.408694f));
    ASSERT_TRUE(roughlyEqual(validCrossCorrelation2DView->getValue(1, 0, 0), -1.356435f));
    ASSERT_TRUE(roughlyEqual(validCrossCorrelation2DView->getValue(1, 1, 0), 0.487984f));
    ASSERT_TRUE(roughlyEqual(validCrossCorrelation2DView->getValue(1, 2, 0), 1.401795f));
    ASSERT_TRUE(roughlyEqual(validCrossCorrelation2DView->getValue(2, 0, 0), 0.470853f));
    ASSERT_TRUE(roughlyEqual(validCrossCorrelation2DView->getValue(2, 1, 0), -1.322469f));
    ASSERT_TRUE(roughlyEqual(validCrossCorrelation2DView->getValue(2, 2, 0), -3.257066f));
}

void testTensorRotate180() {
    vector<vector<vector<float>>> a = {{{1, 2, 3},
                                        {4, 5, 6},
                                        {7, 8, 9}}};
    auto matrix1 = make_shared<FullTensor>(a);
    auto rotated = make_shared<TensorRotate180View>(matrix1);
//    matrix1->print();
//    rotated->print();
    vector<vector<vector<float>>> b = {{{9, 8, 7},
                                        {6, 5, 4},
                                        {3, 2, 1}}};
    auto matrix2 = make_shared<FullTensor>(b);
    assertEqual(rotated, matrix2);
}

void testTensorRotate2() {
    vector<vector<vector<float>>> a = {{{10, 11, 12},
                                        {13, 14, 15},
                                        {16, 17, 18}}};
    auto matrix1 = make_shared<FullTensor>(a);
    auto rotated = make_shared<TensorRotate180View>(matrix1);
//    matrix1->print();
//    rotated->print();
    vector<vector<vector<float>>> b = {{{18, 17, 16},
                                        {15, 14, 13},
                                        {12, 11, 10}}};
    auto matrix2 = make_shared<FullTensor>(b);
    assertEqual(rotated, matrix2);
}

int main() {
    try {
        // TODO: a lot of these tests don't cover the situation where we have many channels
        // they often test the simple case of a single channel.
        EvenMoreSimpleTimer timer;
        testCreate();
        timer.printMilliseconds();
        sumTest();
        timer.printMilliseconds();
        productTest();
        timer.printMilliseconds();
        randomTest();
        timer.printMilliseconds();
        reshapeTest();
        timer.printMilliseconds();
        testScalarMultiplication();
        timer.printMilliseconds();
        testArithmeticMean();
        timer.printMilliseconds();
        testGeometricMean();
        timer.printMilliseconds();
        testSmallStats();
        timer.printMilliseconds();
        test_even_distribution_quarter_small();
        timer.printMilliseconds();
        testEvenDistributionQuarterHugeNumbers();
        timer.printMilliseconds();
        assignSmallTest();
        timer.printMilliseconds();
        assignMediumTest();
        timer.printMilliseconds();
        testConstant();
        timer.printMilliseconds();
        testIdentity();
        timer.printMilliseconds();
        testDotProduct();
        timer.printMilliseconds();
        testDotProduct2();
        timer.printMilliseconds();
        testDotProduct3();
        timer.printMilliseconds();
        testDotProduct4();
        timer.printMilliseconds();
        testMatrixAddition();
        timer.printMilliseconds();
        testStackingMultiplyViews();
        timer.printMilliseconds();
        testZeroPaddedView();
        timer.printMilliseconds();
        testZeroPaddedView2();
        timer.printMilliseconds();
        test2ValidCrossCorrelation2d();
        timer.printMilliseconds();
        testValidCrossCorrelation2d();
        timer.printMilliseconds();
        testFullCrossCorrelation2d();
        timer.printMilliseconds();
        testFullConvolve2dView();
        timer.printMilliseconds();
        testTensorRotate180();
        timer.printMilliseconds();
        testTensorRotate2();
        timer.printMilliseconds();

        test2FullConvolve2dView();
        timer.printMilliseconds();

        //test_pixel()
        //timer.print_milliseconds();
        // slow to test and not worth using day-to-day on my machine
#ifdef FULL_TENSOR_TESTS
        test_even_distribution_quarter_medium(); // roughly 2.5 seconds
        timer.print_milliseconds();
        test_even_distribution_quarter_big(); // 18 seconds async and some very long time single threaded (killed)
        timer.print_seconds();
        test_medium_stats(); // 49 seconds with async, 443 seconds single thread
        timer.print_seconds();
        test_big_stats(); // 752 seconds with async (~12.5 minutes), didn't try with single thread
        timer.print_seconds();
        assign_large_test(); // ~3.9 minutes (235 seconds) in parallel, ~40.5 minutes (2433 seconds) single thread
        timer.print_seconds();
#endif
    } catch (const exception &e) {
        cout << e.what() << endl;
    }

    return 0;
}




