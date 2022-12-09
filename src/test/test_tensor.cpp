//
// Created by Erik Hyrkas on 10/25/2022.
//


#include <iostream>
#include <string>
#include "../types/tensor.hpp"
#include "../util/unit_test.hpp"
#include "../util/tensor_stats.hpp"

// Super slow on my machine, but needed to test everything. Probably not useful for day-to-day unit tests.
//#define FULL_TENSOR_TESTS

using namespace microml;
using namespace std;

void product_test() {
    auto matrix_func = [](size_t row, size_t col, size_t channel) { return (float) (row + 1); };
    auto matrix = std::make_shared<TensorFromFunction>(matrix_func, 3, 3, 1);
//    matrix->print();
//    1.000, 1.000, 1.000
//    2.000, 2.000, 2.000
//    3.000, 3.000, 3.000
    ASSERT_TRUE(216.0f == matrix->product());
}

void random_test() {
    // This isn't a great test, since TensorFromRandomRepeatable is really only useful
    // in a single threaded situation where we want the results to be repeatable.
    // But the moment you start reading from it concurrently, it's no longer repeatable.
    //
    // The only reason I made this was that I may want to have the framework support running
    // in a repeatable mode (which would require running in a single thread.) This seems
    // like it would be terribly slow and make the framework near useless. The alternative
    // is to not use a view and persist all the random values on creation, which works
    // but would require gigabytes of memory for a matrix of any significant size.
    auto matrix1 = std::make_unique<TensorFromRandom>(2, 2, 1, -1000.0f, 1000.0f, 42);
//    matrix1->print();
    auto matrix2 = std::make_unique<TensorFromRandom>(2, 2, 1, -1000.0f, 1000.0f, 42);
//    matrix2->print();
    ASSERT_TRUE(matrix1->get_val(0, 0, 0) == matrix2->get_val(0, 0, 0));
    ASSERT_TRUE(matrix1->get_val(0, 1, 0) == matrix2->get_val(0, 1, 0));
    ASSERT_TRUE(matrix1->get_val(1, 0, 0) == matrix2->get_val(1, 0, 0));
    ASSERT_TRUE(matrix1->get_val(1, 1, 0) == matrix2->get_val(1, 1, 0));
    auto matrix3 = std::make_unique<TensorFromRandom>(2, 2, 1, -1000.0f, 1000.0f, 99);
    auto matrix4 = std::make_unique<TensorFromRandom>(2, 2, 1, -1000.0f, 1000.0f, 99);
    ASSERT_TRUE(matrix3->get_val(0, 0, 0) == matrix4->get_val(0, 0, 0));
    ASSERT_TRUE(matrix3->get_val(0, 1, 0) == matrix4->get_val(0, 1, 0));
    ASSERT_TRUE(matrix3->get_val(1, 0, 0) == matrix4->get_val(1, 0, 0));
    ASSERT_TRUE(matrix3->get_val(1, 1, 0) == matrix4->get_val(1, 1, 0));
    for (int i = 0; i < 200; i++) {
        // tested with much larger matrices, but it's too slow to leave in for day-to-day testing
        auto matrix5 = std::make_unique<TensorFromRandom>(100, 100, 1, -1000.0f, 1000.0f, i);
        float mean = std::abs(matrix5->arithmetic_mean());
//        std::cout << "Seed: " << i << " Abs Mean: " << std::fixed << std::setprecision(38) << mean << std::endl;
        ASSERT_TRUE(mean < 1.0f);
    }
}

void sum_test() {
    auto matrix_func = [](size_t row, size_t col, size_t channel) { return (float) (row + 1); };
    auto matrix = std::make_shared<TensorFromFunction>(matrix_func, 3, 3, 1);
//    matrix->print();
//    1.000, 1.000, 1.000
//    2.000, 2.000, 2.000
//    3.000, 3.000, 3.000
    ASSERT_TRUE(18.0f == matrix->sum());
}

void assign_small_test() {
    auto matrix_random = std::make_shared<TensorFromRandom>(101, 103, 1, 4);
//    matrix_random->print();
    auto matrix = std::make_shared<QuarterTensor>(matrix_random, 4);
//    matrix->print();
//    std::cout << "0, 0 original random: " << matrix_random->get_val(0, 0, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(0, 0, 0), 4, 0), 4, 0) << std::endl;
//    std::cout << "0, 0 new matrix: " << matrix->get_val(0, 0, 0) << std::endl;
//    std::cout << "5, 4 original random: " << matrix_random->get_val(5, 4, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(5, 4, 0), 4, 0), 4, 0) << std::endl;
//    std::cout << "5, 4 new matrix: " << matrix->get_val(5, 4, 0) << std::endl;
//    std::cout << "12, 10 original random: " << matrix_random->get_val(12, 10, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(12, 10, 0), 4, 0), 4, 0) << std::endl;
//    std::cout << "12, 10 new matrix: " << matrix->get_val(12, 10, 0) << std::endl;
//    std::cout << "50, 10 original random: " << matrix_random->get_val(50, 10, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(50, 10, 0), 4, 0), 4, 0) << std::endl;
//    std::cout << "50, 10 new matrix: " << matrix->get_val(50, 10, 0) << std::endl;
//    std::cout << "99, 99 original random: " << matrix_random->get_val(99, 99, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(99, 99, 0), 4, 0), 4, 0) << std::endl;
//    std::cout << "99, 99 new matrix: " << matrix->get_val(99, 99, 0) << std::endl;
//
//    std::cout << "99, 99 original random: " << matrix_random->get_val(100, 102, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(100, 102, 0), 4, 0), 4, 0) << std::endl;
//    std::cout << "99, 99 new matrix: " << matrix->get_val(100, 102, 0) << std::endl;

    ASSERT_TRUE(matrix->get_val(0, 0, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(0, 0, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(5, 4, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(5, 4, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(12, 10, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(12, 10, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(50, 10, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(50, 10, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(99, 99, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(99, 99, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(100, 102, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(100, 102, 0), 4), 4));
}


void assign_medium_test() {
    auto matrix_random = std::make_shared<TensorFromRandom>(1001, 10003, 1, 4);
//    matrix_random->print();
    auto matrix = std::make_shared<QuarterTensor>(matrix_random, 4);
//    matrix->print();
//    std::cout << "0, 0 original random: " << matrix_random->get_val(0, 0, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(0, 0, 0), 4, 0), 4, 0) << std::endl;
//    std::cout << "0, 0 new matrix: " << matrix->get_val(0, 0, 0) << std::endl;
//    std::cout << "5, 4 original random: " << matrix_random->get_val(5, 4, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(5, 4, 0), 4, 0), 4, 0) << std::endl;
//    std::cout << "5, 4 new matrix: " << matrix->get_val(5, 4, 0) << std::endl;
//    std::cout << "12, 10 original random: " << matrix_random->get_val(12, 10, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(12, 10, 0), 4, 0), 4, 0) << std::endl;
//    std::cout << "12, 10 new matrix: " << matrix->get_val(12, 10, 0) << std::endl;
//    std::cout << "50, 10 original random: " << matrix_random->get_val(50, 10, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(50, 10, 0), 4, 0), 4, 0) << std::endl;
//    std::cout << "50, 10 new matrix: " << matrix->get_val(50, 10, 0) << std::endl;
//    std::cout << "99, 99 original random: " << matrix_random->get_val(99, 99, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(99, 99, 0), 4, 0), 4, 0) << std::endl;
//    std::cout << "99, 99 new matrix: " << matrix->get_val(99, 99, 0) << std::endl;
//    std::cout << "1000, 10002 original random: " << matrix_random->get_val(1000, 10002, 0) << " quantized: " << quarter_to_float(float_to_quarter(matrix_random->get_val(1000, 10002, 0), 4, 0), 4, 0) << std::endl;
//    std::cout << "1000, 10002 new matrix: " << matrix->get_val(1000, 10002, 0) << std::endl;
    ASSERT_TRUE(matrix->get_val(0, 0, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(0, 0, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(5, 4, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(5, 4, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(12, 10, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(12, 10, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(50, 10, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(50, 10, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(99, 99, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(99, 99, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(1000, 10002, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(1000, 10002, 0), 4), 4));
}

void assign_large_test() {
    auto matrix_random = std::make_shared<TensorFromRandom>(200000, 200001, 1, 4);
    auto matrix = std::make_shared<QuarterTensor>(matrix_random, 4);
    std::cout << "0, 0 original random: " << matrix_random->get_val(0, 0, 0) << " quantized: "
              << quarter_to_float(float_to_quarter(matrix_random->get_val(0, 0, 0), 4), 4) << std::endl;
    std::cout << "0, 0 new matrix: " << matrix->get_val(0, 0, 0) << std::endl;
    std::cout << "199999, 199999 original random: " << matrix_random->get_val(199999, 199999, 0) << " quantized: "
              << quarter_to_float(float_to_quarter(matrix_random->get_val(199999, 199999, 0), 4), 4) << std::endl;
    std::cout << "199999, 199999 new matrix: " << matrix->get_val(199999, 199999, 0) << std::endl;
    ASSERT_TRUE(matrix->get_val(0, 0, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(0, 0, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(100, 50, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(100, 50, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(50, 10, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(50, 10, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(9000, 10000, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(9000, 10000, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(1, 183784, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(1, 183784, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(180034, 1, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(180034, 1, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(162341, 44228, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(162341, 44228, 0), 4), 4));
    ASSERT_TRUE(matrix->get_val(199999, 199999, 0) ==
                        quarter_to_float(float_to_quarter(matrix_random->get_val(199999, 199999, 0), 4), 4));
}

void reshape_test() {
    auto matrix_random = std::make_shared<TensorFromRandom>(1, 5, 1, 4);
//    matrix_random->print();
    auto matrix = std::make_shared<QuarterTensor>(matrix_random, 4);
//    matrix->print();
    auto reshape = std::make_shared<TensorReshapeView>(matrix_random, 5, 1);
    auto other = std::make_unique<QuarterTensor>(reshape, 4);
    ASSERT_TRUE(matrix->get_val(0, 0, 0) == other->get_val(0, 0, 0));
    ASSERT_TRUE(matrix->get_val(0, 1, 0) == other->get_val(1, 0, 0));
    ASSERT_TRUE(matrix->get_val(0, 2, 0) == other->get_val(2, 0, 0));
    ASSERT_TRUE(matrix->get_val(0, 3, 0) == other->get_val(3, 0, 0));
    ASSERT_TRUE(matrix->get_val(0, 4, 0) == other->get_val(4, 0, 0));
    auto otherView = std::make_unique<TensorReshapeView>(matrix, 5, 1);
    ASSERT_TRUE(otherView->get_val(0, 0, 0) == other->get_val(0, 0, 0));
    ASSERT_TRUE(otherView->get_val(1, 0, 0) == other->get_val(1, 0, 0));
    ASSERT_TRUE(otherView->get_val(2, 0, 0) == other->get_val(2, 0, 0));
    ASSERT_TRUE(otherView->get_val(3, 0, 0) == other->get_val(3, 0, 0));
    ASSERT_TRUE(otherView->get_val(4, 0, 0) == other->get_val(4, 0, 0));
}

void test_create() {
    auto matrix = std::make_unique<QuarterTensor>(2, 2, 1, 4);
    ASSERT_TRUE(2 == matrix->row_count());
    ASSERT_TRUE(2 == matrix->column_count());
}

void test_scalar_multiplication() {
    auto matrix_func = [](size_t row, size_t col, size_t channel) { return (float) row; };
    auto matrix = std::make_shared<TensorFromFunction>(matrix_func, 5, 5, 1);
//    matrix->print();
    auto scaled_matrix = std::make_unique<TensorMultiplyByScalarView>(matrix, 6.0f);
//    scaled_matrix->print();
    ASSERT_TRUE(0 == scaled_matrix->get_val(0, 0, 0));
    ASSERT_TRUE(0 == scaled_matrix->get_val(0, 2, 0));
    ASSERT_TRUE(0 == scaled_matrix->get_val(0, 4, 0));
    ASSERT_TRUE(12 == scaled_matrix->get_val(2, 0, 0));
    ASSERT_TRUE(12 == scaled_matrix->get_val(2, 2, 0));
    ASSERT_TRUE(12 == scaled_matrix->get_val(2, 4, 0));
    ASSERT_TRUE(24 == scaled_matrix->get_val(4, 0, 0));
    ASSERT_TRUE(24 == scaled_matrix->get_val(4, 2, 0));
    ASSERT_TRUE(24 == scaled_matrix->get_val(4, 4, 0));
}

void test_stacking_multiply_views() {
    auto matrix_func = [](size_t row, size_t col, size_t channel) { return (float) (((float)row*10.f) + (float)col); };
    auto matrix = std::make_shared<TensorFromFunction>(matrix_func, 5, 5, 1);
    auto times2 = make_shared<TensorMultiplyByScalarView>(matrix, 2.f);
    auto timesHalf = make_shared<TensorMultiplyByScalarView>(times2, 0.5f);
    for(size_t r = 0; r < 5; r++) {
        for(size_t c = 0; c < 5; c++) {
//            cout << matrix->get_val(r, c, 0) << " == " << timesHalf->get_val(r, c, 0) << endl;
            ASSERT_TRUE(roughly_equal(matrix->get_val(r, c, 0), timesHalf->get_val(r, c, 0)));
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

void test_arithmetic_mean() {
    auto matrix_func = [](size_t row, size_t col, size_t channel) { return (float) row; };
    auto matrix = std::make_unique<TensorFromFunction>(matrix_func, 2, 2, 1);
//    matrix->print();
//    0.000, 0.000
//    1.000, 1.000
    ASSERT_TRUE(0.5 == matrix->arithmetic_mean());
}

void test_geometric_mean() {
    auto matrix = std::make_unique<TensorFromRandom>(2, 2, 1, -10.0f, 0.0f, 9);
//    matrix->print();
//    -5.236, -6.793
//    -1.519, -3.077
    ASSERT_TRUE(isnan(matrix->geometric_mean()));
    auto matrix2 = std::make_unique<TensorFromRandom>(2, 2, 1, 1.0f, 10.0f, 36);
//    matrix2->print();
//    5.288, 3.886
//    8.004, 6.603
    float gm = matrix2->geometric_mean();
//    std::cout << "GM: " << std::fixed << std::setprecision(38) << gm << std::endl;
    ASSERT_TRUE(roughly_equal(5.7405242919921875f, gm));
}

void test_big_stats() {
    // 200,000 x 200,000 = 40 billion elements = ~38gb matrix
    auto matrix = std::make_shared<TensorFromRandom>(200000, 200000, 1, 7);
//    matrix->print();
    auto stats = std::make_unique<TensorStats>(*matrix, FIT_BIAS_FOR_50);
    stats->print();
    ASSERT_TRUE(8 == stats->get_recommended_bias());
    ASSERT_TRUE(roughly_equal(-120.0f, stats->get_recommended_offset()));
}

void test_medium_stats() {
    // 50,000 x 50,000 = 2.5 billion elements = ~2gb matrix
    auto matrix = std::make_shared<TensorFromRandom>(50000, 50000, 1, 14);
//    matrix->print();
    auto stats = std::make_unique<TensorStats>(*matrix, FIT_BIAS_FOR_50);
    stats->print();
    ASSERT_TRUE(14 == stats->get_recommended_bias());
    ASSERT_TRUE(roughly_equal(0.0f, stats->get_recommended_offset()));
}

void test_small_stats() {
    auto matrix = std::make_shared<TensorFromRandom>(50, 50, 1, 4);
//    matrix->print();
    auto stats = std::make_unique<TensorStats>(*matrix, FIT_BIAS_FOR_100);
//    stats->print();
    ASSERT_TRUE(4 == stats->get_recommended_bias());
    ASSERT_TRUE(roughly_equal(-3.16455078125f, stats->get_recommended_offset()));
}

// You'll notice that our distribution isn't even. The further we get from 0, the less granularity we have, so the
// more data that's grouped together in a bucket.
void test_even_distribution_quarter_medium() {
    auto matrix_func = [](size_t row, size_t col, size_t channel) { return ((float) (row + 1)) / 200.0; };
    auto matrix = std::make_unique<TensorFromFunction>(matrix_func, 10000, 10000, 1);
//    matrix->print();
    auto stats = std::make_unique<TensorStats>(*matrix, FIT_BIAS_FOR_80);
//    stats->print();
    ASSERT_TRUE(8 == stats->get_recommended_bias());
    ASSERT_TRUE(roughly_equal(0.0f, stats->get_recommended_offset()));
}

void test_even_distribution_quarter_small() {
    auto matrix_func = [](size_t row, size_t col, size_t channel) { return ((float) (row + 1)) / 200.0; };
    auto matrix = std::make_unique<TensorFromFunction>(matrix_func, 350, 350, 1);
//    matrix->print();
    auto stats = std::make_unique<TensorStats>(*matrix, FIT_BIAS_FOR_80);
//    stats->print();
    ASSERT_TRUE(14 == stats->get_recommended_bias());
    const auto ro = stats->get_recommended_offset();
//    std::cout << std::fixed << std::setprecision(38) << ro << std::endl;
    ASSERT_TRUE(roughly_equal(0.8424999713897705078125f, ro));
}

void test_even_distribution_quarter_big() {
    auto matrix_func = [](size_t row, size_t col, size_t channel) { return ((float) (row + 1)) / 20000.0; };
    auto matrix = std::make_unique<TensorFromFunction>(matrix_func, 100000000, 1, 1);
//    matrix->print();
    auto stats = std::make_unique<TensorStats>(*matrix, FIT_BIAS_FOR_80);
    stats->print();
    ASSERT_TRUE(1 == stats->get_recommended_bias());
    ASSERT_TRUE(roughly_equal(2431.999756f, stats->get_recommended_offset()));
}

void test_even_distribution_quarter_huge_numbers() {
    auto matrix_func = [](size_t row, size_t col, size_t channel) { return ((float) (row + 1)) * 500.0; };
    auto matrix = std::make_unique<TensorFromFunction>(matrix_func, 1000, 1, 1);
//    matrix->print();
    auto stats = std::make_unique<TensorStats>(*matrix, FIT_BIAS_FOR_80);
//    stats->print();
    ASSERT_TRUE(-4 == stats->get_recommended_bias());
    ASSERT_TRUE(roughly_equal(237500.0f, stats->get_recommended_offset()));
}

void test_constant() {
    auto matrix = std::make_shared<UniformTensor>(10, 10, 1, 0.0f);
    for (size_t i = 0; i < matrix->row_count(); i++) {
        for (size_t j = 0; j < matrix->column_count(); j++) {
            ASSERT_TRUE(0.0f == matrix->get_val(i, j, 0));
        }
    }
//    matrix->print();
}


void test_identity() {
    auto matrix = std::make_shared<IdentityTensor>(10, 10, 1);
    for (size_t i = 0; i < matrix->row_count(); i++) {
        for (size_t j = 0; j < matrix->column_count(); j++) {
            if (i == j) {
                ASSERT_TRUE(1.0f == matrix->get_val(i, j, 0));
            } else {
                ASSERT_TRUE(0.0f == matrix->get_val(i, j, 0));
            }
        }
    }
//    matrix->print();
}

void test_dot_product() {
    auto matrix_func = [](size_t row, size_t col, size_t channel) { return ((float) (row + col)); };
    auto matrix_1 = std::make_shared<TensorFromFunction>(matrix_func, 2, 2, 1);
//    matrix_1->print();
    auto matrix_2 = std::make_shared<TensorFromFunction>(matrix_func, 2, 2, 1);
    auto dot_product_view = std::make_shared<TensorDotTensorView>(matrix_1, matrix_2);
//    dot_product_view->print();
    ASSERT_TRUE(1.0f == dot_product_view->get_val(0, 0, 0));
    ASSERT_TRUE(2.0f == dot_product_view->get_val(0, 1, 0));
    ASSERT_TRUE(2.0f == dot_product_view->get_val(1, 0, 0));
    ASSERT_TRUE(5.0f == dot_product_view->get_val(1, 1, 0));
}

void test_dot_product_2() {
    std::vector<std::vector<float>> a = {{4, 2},
                                         {0, 3}};
    auto matrix_1 = std::make_shared<QuarterTensor>(a, 8);
//    matrix_1->print();
    std::vector<std::vector<float>> b = {{4, 0},
                                         {1, 4}};
    auto matrix_2 = std::make_shared<QuarterTensor>(b, 8);
    auto dot_product_view = std::make_shared<TensorDotTensorView>(matrix_1, matrix_2);
//    dot_product_view->print();
    ASSERT_TRUE(18.0f == dot_product_view->get_val(0, 0, 0));
    ASSERT_TRUE(8.0f == dot_product_view->get_val(0, 1, 0));
    ASSERT_TRUE(3.0f == dot_product_view->get_val(1, 0, 0));
    ASSERT_TRUE(12.0f == dot_product_view->get_val(1, 1, 0));
}

void test_dot_product_3() {
    std::vector<std::vector<float>> a = {{2, 2},
                                         {0, 3},
                                         {0, 4}};
    auto matrix_1 = std::make_shared<QuarterTensor>(a, 8);
//    matrix_1->print();
    std::vector<std::vector<float>> b = {{2, 1, 2},
                                         {3, 2, 4}};
    auto matrix_2 = std::make_shared<QuarterTensor>(b, 8);
    auto dot_product_view = std::make_shared<TensorDotTensorView>(matrix_1, matrix_2);
//    dot_product_view->print();
    ASSERT_TRUE(10.0f == dot_product_view->get_val(0, 0, 0));
    ASSERT_TRUE(6.0f == dot_product_view->get_val(0, 1, 0));
    ASSERT_TRUE(12.0f == dot_product_view->get_val(0, 2, 0));
    ASSERT_TRUE(9.0f == dot_product_view->get_val(1, 0, 0));
    ASSERT_TRUE(6.0f == dot_product_view->get_val(1, 1, 0));
    ASSERT_TRUE(12.0f == dot_product_view->get_val(1, 2, 0));
    ASSERT_TRUE(12.0f == dot_product_view->get_val(2, 0, 0));
    ASSERT_TRUE(8.0f == dot_product_view->get_val(2, 1, 0));
    ASSERT_TRUE(16.0f == dot_product_view->get_val(2, 2, 0));
}

void test_dot_product_4() {
    std::vector<std::vector<float>> a = {{1, 2, 3},
                                         {4, 5, 6}};
    auto matrix_1 = std::make_shared<QuarterTensor>(a, 8);
//    matrix_1->print();
    std::vector<std::vector<float>> b = {{7,  8},
                                         {9,  10},
                                         {11, 12}};
    auto matrix_2 = std::make_shared<QuarterTensor>(b, 8);
    auto dot_product_view = std::make_shared<TensorDotTensorView>(matrix_1, matrix_2);
//    dot_product_view->print();
    ASSERT_TRUE(58.0f == dot_product_view->get_val(0, 0, 0));
    ASSERT_TRUE(64.0f == dot_product_view->get_val(0, 1, 0));
    ASSERT_TRUE(139.0f == dot_product_view->get_val(1, 0, 0));
    ASSERT_TRUE(154.0f == dot_product_view->get_val(1, 1, 0));
}

void test_matrix_addition() {
    std::vector<std::vector<float>> a = {{-1, 2,  3},
                                         {2,  -3, 1},
                                         {3,  1,  -2}};
    auto matrix_1 = std::make_shared<QuarterTensor>(a, 8);
//    matrix_1->print();
    std::vector<std::vector<float>> b = {{3, -1, 2},
                                         {1, 0,  3},
                                         {2, -1, 0}};
    auto matrix_2 = std::make_shared<QuarterTensor>(b, 8);
    auto add_view = std::make_shared<TensorAddTensorView>(matrix_1, matrix_2);
    ASSERT_TRUE(2.0f == add_view->get_val(0, 0, 0));
    ASSERT_TRUE(1.0f == add_view->get_val(0, 1, 0));
    ASSERT_TRUE(5.0f == add_view->get_val(0, 2, 0));
    ASSERT_TRUE(3.0f == add_view->get_val(1, 0, 0));
    ASSERT_TRUE(-3.0f == add_view->get_val(1, 1, 0));
    ASSERT_TRUE(4.0f == add_view->get_val(1, 2, 0));
    ASSERT_TRUE(5.0f == add_view->get_val(2, 0, 0));
    ASSERT_TRUE(0.0f == add_view->get_val(2, 1, 0));
    ASSERT_TRUE(-2.0f == add_view->get_val(2, 2, 0));
}

void test_pixel() {
    auto matrix = std::make_shared<TensorFromRandom>(5, 5, 1, 0.f, 1.f, 42);
    auto pixel_test = make_shared<PixelTensor>(matrix);
    pixel_test->print();
    matrix->print();
    // todo: assert rather than print.
}

int main() {
    try {
        // TODO: a lot of these tests don't cover the situation where we have many channels
        // they often test the simple case of a single channel.
        EvenMoreSimpleTimer timer;
        test_create();
        timer.print_milliseconds();
        sum_test();
        timer.print_milliseconds();
        product_test();
        timer.print_milliseconds();
        random_test();
        timer.print_milliseconds();
        reshape_test();
        timer.print_milliseconds();
        test_scalar_multiplication();
        timer.print_milliseconds();
        test_arithmetic_mean();
        timer.print_milliseconds();
        test_geometric_mean();
        timer.print_milliseconds();
        test_small_stats();
        timer.print_milliseconds();
        test_even_distribution_quarter_small();
        timer.print_milliseconds();
        test_even_distribution_quarter_huge_numbers();
        timer.print_milliseconds();
        assign_small_test();
        timer.print_milliseconds();
        assign_medium_test();
        timer.print_milliseconds();
        test_constant();
        timer.print_milliseconds();
        test_identity();
        timer.print_milliseconds();
        test_dot_product();
        timer.print_milliseconds();
        test_dot_product_2();
        timer.print_milliseconds();
        test_dot_product_3();
        timer.print_milliseconds();
        test_dot_product_4();
        timer.print_milliseconds();
        test_matrix_addition();
        timer.print_milliseconds();
        test_stacking_multiply_views();
        timer.print_milliseconds();


        //auto weights = make_shared<TensorFromRandom>(5, 5, 1, 14);
//        for(int bias = 10; bias < 20; bias++) {
//            cout << bias << " MAX: " << quarter_to_float(QUARTER_MAX, bias) << " MIN: " << quarter_to_float(
//                    QUARTER_MAX, bias) << endl;
//        }

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
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}




