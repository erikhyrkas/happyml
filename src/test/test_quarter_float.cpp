#include <iostream>
#include <iomanip>
#include "../types/quarter_float.hpp"
#include "../util/unit_test.hpp"

using namespace microml;
using namespace std;

void print_conversion(int bias, float i, bool brief) {
    const quarter quarter_default = float_to_quarter(i, bias);
    const float float_default = quarter_to_float(quarter_default, bias);
//    const quarter quarter_round = float_to_quarter_round_nearest_v2(i, bias, 0, false);
//    const float float_round = quarter_to_float_v2(quarter_round, bias, 0);
//    const quarter quarter_round_avoid_zero = float_to_quarter_round_nearest_v2(i, bias, 0, true);
//    const float float_round_avoid_zero = quarter_to_float_v2(quarter_round_avoid_zero, bias, 0);
//    const quarter quarter_no_round = float_to_quarter_v1(i, bias, 0);
//    const float float_no_round = quarter_to_float_v2(quarter_no_round, bias, 0);
    if (brief) {
        std::cout << "bias " << bias << " value: " << std::setprecision(3) << i << std::fixed << std::setprecision(20)
                  << " default: " << float_default
//                  << " nearest: " << float_round
//                  << " nearest avoid zero: " << float_round_avoid_zero
//                  << " no round: " << float_no_round
                  << std::endl;
    } else {
        //<< std::fixed << std::setprecision(35)
        std::cout << std::endl << "Bias: " << bias << " Original value: " << std::setprecision(3) << i << std::endl;
        print_bits(i);
        std::cout << "quarter default: " << float_default << std::endl;
        print_bits(float_default);
        print_bits(quarter_default);
//        std::cout << "quarter round: " << float_round << std::endl;
//        print_bits(float_round);
//        print_bits(quarter_round);
//        std::cout << "quarter round avoid zero: " << float_round_avoid_zero << std::endl;
//        print_bits(float_round_avoid_zero);
//        print_bits(quarter_round_avoid_zero);
//        std::cout << "quarter no round: " << float_no_round << std::endl;
//        print_bits(float_no_round);
//        print_bits(quarter_no_round);
        std::cout << std::endl;
    }
}

void print_conversions_small_numbers(int bias, bool brief) {
    print_conversion(bias, 0, brief);
    for (int i = 1; i <= 10; i++) {
        float f = ((float) i) / 1000.0f;
        print_conversion(bias, f, brief);
    }
    for (int i = 1; i <= 10; i++) {
        float f = ((float) i) / 100.0f;
        print_conversion(bias, f, brief);
    }
    for (int i = 1; i <= 30; i++) {
        float f = 0.1f + (float) i / 10.0f;
        print_conversion(bias, f, brief);
    }
}

void print_conversions_big_numbers(int bias, bool brief) {
    print_conversion(bias, 0, brief);
    print_conversion(bias, 1, brief);
    for (int i = 1; i <= 10; i++) {
        print_conversion(bias, (float) i * 10.0f, brief);
    }
    for (int i = 1; i <= 10; i++) {
        print_conversion(bias, (float) i * 100.0f, brief);
    }
}

void test_add(float a, float b, float expected_result, int bias) {
    // yes this bounces back and forth between float and quarter a lot, but it's good exercise.
//    float real_result = a + b;
//    std::cout << "Real result: " << real_result << " "
//              << quarter_to_float(float_to_quarter_round_nearest_not_zero(real_result, bias, 0), bias, 0) << " vs "
//              << quarter_to_float(float_to_quarter_round_nearest_not_zero(expected_result, bias, 0), bias, 0) << std::endl;

    quarter expected_result_quarter = float_to_quarter(expected_result, bias);
    quarter first = float_to_quarter(a, bias);
    quarter second = float_to_quarter(b, bias);
    quarter add_result = quarter_add(first, bias, second, bias, bias);
    float add_result_float = quarter_to_float(add_result, bias);
    std::cout << std::endl << "Testing: " << bias << ": " << a << "(" << quarter_to_float(first, bias) << ")"
              << " + " << b << "(" << quarter_to_float(second, bias) << ")" << " = " << add_result_float << "("
              << quarter_to_float(expected_result_quarter, bias) << ")" << std::endl;
    print_bits(add_result);
    print_bits(expected_result_quarter);
    ASSERT_TRUE(roughly_equal(add_result, expected_result_quarter));
}

void test_subtract(float a, float b, float expected_result, int bias) {
    // yes this bounces back and forth between float and quarter a lot, but it's good exercise.
    quarter expected_result_quarter = float_to_quarter(expected_result, bias);
    quarter first = float_to_quarter(a, bias);
    quarter second = float_to_quarter(b, bias);
    quarter add_result = quarter_subtract(first, bias, second, bias, bias);
    float add_result_float = quarter_to_float(add_result, bias);
    std::cout << std::endl << "Testing: " << a << " - " << b << " = " << add_result_float << std::endl;
    ASSERT_TRUE(add_result == expected_result_quarter);
}

void test_multiply(float a, float b, float expected_result, int bias) {
    // yes this bounces back and forth between float and quarter a lot, but it's good exercise.
    quarter expected_result_quarter = float_to_quarter(expected_result, bias);
    quarter first = float_to_quarter(a, bias);
    quarter second = float_to_quarter(b, bias);
    quarter add_result = quarter_multiply(first, bias, second, bias, bias);
    float add_result_float = quarter_to_float(add_result, bias);
    std::cout << std::endl << "Testing: " << a << " * " << b << " = " << add_result_float << std::endl;
    ASSERT_TRUE(add_result == expected_result_quarter);
}

void test_divide(float a, float b, float expected_result, int bias) {
    // yes this bounces back and forth between float and quarter a lot, but it's good exercise.
    quarter expected_result_quarter = float_to_quarter(expected_result, bias);
    quarter first = float_to_quarter(a, bias);
    quarter second = float_to_quarter(b, bias);
    quarter add_result = quarter_divide(first, bias, second, bias, bias);
    float add_result_float = quarter_to_float(add_result, bias);
    std::cout << std::endl << "Testing: " << a << " / " << b << " = " << add_result_float << std::endl;
    ASSERT_TRUE(add_result == expected_result_quarter);
}

int test_one_quarter(float f, int quarter_bias) {
    std::cout << std::endl << "Testing: " << f << std::endl;
    quarter q = float_to_quarter(f, quarter_bias);
    float f2 = quarter_to_float(q, quarter_bias);
    print_bits(f);
    print_bits(q);
    print_bits(f2);
    std::cout << "Received: " << f2 << std::endl;
    if (isnan(f) && isnan(f2)) {
        return 1;
    }
    return f == f2;
}

int min_max_smallest_test(int bias) {
    std::cout << std::endl << bias << " bias:" << std::endl;
    int result = test_one_quarter(quarter_to_float(QUARTER_MAX, bias), bias);
    result &= test_one_quarter(quarter_to_float(QUARTER_MIN, bias), bias);
    result &= test_one_quarter(quarter_to_float(QUARTER_SMALLEST, bias), bias);
    return result;
}

void test_quarter() {
    ASSERT_TRUE(test_one_quarter(NAN, 4));
    ASSERT_TRUE(test_one_quarter(INFINITY, 4));
    ASSERT_TRUE(test_one_quarter(-INFINITY, 4));
    ASSERT_TRUE(test_one_quarter(1792, 4));
    ASSERT_TRUE(test_one_quarter(1, 4));
    ASSERT_TRUE(test_one_quarter(0.875, 4));
    ASSERT_TRUE(test_one_quarter(0.75, 4));
    ASSERT_TRUE(test_one_quarter(0.625, 4));
    ASSERT_TRUE(test_one_quarter(0.5, 4));
    ASSERT_TRUE(test_one_quarter(0.375, 4));
    ASSERT_TRUE(test_one_quarter(0.125, 4));
    ASSERT_TRUE(test_one_quarter(0, 4));
    ASSERT_TRUE(test_one_quarter(-0.125, 4));
    ASSERT_TRUE(test_one_quarter(-0.375, 4));
    ASSERT_TRUE(test_one_quarter(-0.875, 4));
    ASSERT_TRUE(test_one_quarter(-1, 4));
    ASSERT_TRUE(test_one_quarter(-6, 4));
    ASSERT_TRUE(test_one_quarter(-96, 4));
    ASSERT_TRUE(test_one_quarter(-1792, 4));
    ASSERT_TRUE(test_one_quarter(7680, 2));
    ASSERT_TRUE(test_one_quarter(7168, 2));
    ASSERT_TRUE(test_one_quarter(15360, 1));
    ASSERT_TRUE(test_one_quarter(14336, 1));
    ASSERT_TRUE(test_one_quarter(13312, 1));
    ASSERT_TRUE(test_one_quarter(8192, 1));
    ASSERT_TRUE(test_one_quarter(-14336, 1));
    ASSERT_TRUE(test_one_quarter(-15360, 0));
    for (int i = 0; i < 9; i++) {
        ASSERT_TRUE(min_max_smallest_test(i));
    }

    ASSERT_FALSE(test_one_quarter(0.00001, 0));
    ASSERT_FALSE(test_one_quarter(-0.2, 0));

    ASSERT_TRUE(test_one_quarter(2, 4));
    ASSERT_TRUE(test_one_quarter(1, 0));
    ASSERT_TRUE(test_one_quarter(-1, 0));

    ASSERT_TRUE(test_one_quarter(quarter_to_float(QUARTER_MIN, 0), 0));
    ASSERT_TRUE(test_one_quarter(quarter_to_float(QUARTER_SECOND_MIN, 0), 0));

    // Test that the second minimum value for bias 0 rounds to minimum value,
    // since the second minimum is used to represent 1.
    const uint32_t second_min_bits = 0b11000110111000000000000000000000;
    const float second_min = *(float *) &second_min_bits; // -28672
    ASSERT_TRUE(float_to_quarter(second_min, 0) == float_to_quarter(quarter_to_float(QUARTER_MIN, 0), 0));

    // Lots of rounding errors, but it is to be expected.
    test_add(1, 2, 3, 4);
    test_add(0.5, 10.3, 11, 4);
    test_add(0.1, 10.1, 10.2, 4);
    test_add(0.003, 0.003, 0.0087, 0);
    test_add(0.005, 0.005, 0.0097, 8);
    test_add(0.0012, 0.0012, 0.00195313, 8);
    test_subtract(0.0012, 0.0012, 0, 8);
    test_subtract(0.5, 0.1, 0.41, 8);
    test_multiply(1, 0.5, 0.5, 8);
    test_multiply(5, 5, 25, 8);
    test_divide(5, 5, 1, 8);
    test_divide(5, 0, INFINITY, 8);
    test_divide(0, 0, NAN, 8);

    test_add(0.003, 0.003, 0.00585938, 14);
    test_add(0.0012, 0.0012, 0.00244141, 14);
    test_subtract(0.0012, 0.0012, 0, 14);

}


int main() {
    try {
        test_quarter();

        print_conversions_small_numbers(0, true);
        print_conversions_big_numbers(0, true);

        print_conversions_small_numbers(4, true);
        print_conversions_big_numbers(4, true);

        print_conversions_small_numbers(8, true);
        print_conversions_big_numbers(8, true);

        print_conversions_small_numbers(14, true);
        print_conversions_big_numbers(14, true);
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
