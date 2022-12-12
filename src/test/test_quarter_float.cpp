#include <iostream>
#include <iomanip>
#include "../types/quarter_float.hpp"
#include "../util/unit_test.hpp"

using namespace microml;
using namespace std;

void printConversion(int bias, float i, bool brief) {
    const quarter quarter_default = floatToQuarter(i, bias);
    const float float_default = quarterToFloat(quarter_default, bias);
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
        printBits(i);
        std::cout << "quarter default: " << float_default << std::endl;
        printBits(float_default);
        printBits(quarter_default);
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

void printConversionsSmallNumbers(int bias, bool brief) {
    printConversion(bias, 0, brief);
    for (int i = 1; i <= 10; i++) {
        float f = ((float) i) / 1000.0f;
        printConversion(bias, f, brief);
    }
    for (int i = 1; i <= 10; i++) {
        float f = ((float) i) / 100.0f;
        printConversion(bias, f, brief);
    }
    for (int i = 1; i <= 30; i++) {
        float f = 0.1f + (float) i / 10.0f;
        printConversion(bias, f, brief);
    }
}

void printConversionsBigNumbers(int bias, bool brief) {
    printConversion(bias, 0, brief);
    printConversion(bias, 1, brief);
    for (int i = 1; i <= 10; i++) {
        printConversion(bias, (float) i * 10.0f, brief);
    }
    for (int i = 1; i <= 10; i++) {
        printConversion(bias, (float) i * 100.0f, brief);
    }
}

void testAdd(float a, float b, float expected_result, int bias) {
    // yes this bounces back and forth between float and quarter a lot, but it's good exercise.
//    float real_result = a + b;
//    std::cout << "Real result: " << real_result << " "
//              << quarter_to_float(float_to_quarter_round_nearest_not_zero(real_result, bias, 0), bias, 0) << " vs "
//              << quarter_to_float(float_to_quarter_round_nearest_not_zero(expected_result, bias, 0), bias, 0) << std::endl;

    quarter expected_result_quarter = floatToQuarter(expected_result, bias);
    quarter first = floatToQuarter(a, bias);
    quarter second = floatToQuarter(b, bias);
    quarter add_result = quarterAdd(first, bias, second, bias, bias);
    float add_result_float = quarterToFloat(add_result, bias);
    std::cout << std::endl << "Testing: " << bias << ": " << a << "(" << quarterToFloat(first, bias) << ")"
              << " + " << b << "(" << quarterToFloat(second, bias) << ")" << " = " << add_result_float << "("
              << quarterToFloat(expected_result_quarter, bias) << ")" << std::endl;
    printBits(add_result);
    printBits(expected_result_quarter);
    ASSERT_TRUE(roughlyEqual(add_result, expected_result_quarter));
}

void testSubtract(float a, float b, float expected_result, int bias) {
    // yes this bounces back and forth between float and quarter a lot, but it's good exercise.
    quarter expected_result_quarter = floatToQuarter(expected_result, bias);
    quarter first = floatToQuarter(a, bias);
    quarter second = floatToQuarter(b, bias);
    quarter add_result = quarterSubtract(first, bias, second, bias, bias);
    float add_result_float = quarterToFloat(add_result, bias);
    std::cout << std::endl << "Testing: " << a << " - " << b << " = " << add_result_float << std::endl;
    ASSERT_TRUE(add_result == expected_result_quarter);
}

void testMultiply(float a, float b, float expected_result, int bias) {
    // yes this bounces back and forth between float and quarter a lot, but it's good exercise.
    quarter expected_result_quarter = floatToQuarter(expected_result, bias);
    quarter first = floatToQuarter(a, bias);
    quarter second = floatToQuarter(b, bias);
    quarter add_result = quarterMultiply(first, bias, second, bias, bias);
    float add_result_float = quarterToFloat(add_result, bias);
    std::cout << std::endl << "Testing: " << a << " * " << b << " = " << add_result_float << std::endl;
    ASSERT_TRUE(add_result == expected_result_quarter);
}

void testDivide(float a, float b, float expected_result, int bias) {
    // yes this bounces back and forth between float and quarter a lot, but it's good exercise.
    quarter expected_result_quarter = floatToQuarter(expected_result, bias);
    quarter first = floatToQuarter(a, bias);
    quarter second = floatToQuarter(b, bias);
    quarter add_result = quarterDivide(first, bias, second, bias, bias);
    float add_result_float = quarterToFloat(add_result, bias);
    std::cout << std::endl << "Testing: " << a << " / " << b << " = " << add_result_float << std::endl;
    ASSERT_TRUE(add_result == expected_result_quarter);
}

int testOneQuarter(float f, int quarterBias) {
    std::cout << std::endl << "Testing: " << f << std::endl;
    quarter q = floatToQuarter(f, quarterBias);
    float f2 = quarterToFloat(q, quarterBias);
    printBits(f);
    printBits(q);
    printBits(f2);
    std::cout << "Received: " << f2 << std::endl;
    if (isnan(f) && isnan(f2)) {
        return 1;
    }
    return f == f2;
}

int minMaxSmallestTest(int bias) {
    std::cout << std::endl << bias << " bias:" << std::endl;
    int result = testOneQuarter(quarterToFloat(QUARTER_MAX, bias), bias);
    result &= testOneQuarter(quarterToFloat(QUARTER_MIN, bias), bias);
    result &= testOneQuarter(quarterToFloat(QUARTER_SMALLEST, bias), bias);
    return result;
}

void testQuarter() {
    ASSERT_TRUE(testOneQuarter(NAN, 4));
    ASSERT_TRUE(testOneQuarter(INFINITY, 4));
    ASSERT_TRUE(testOneQuarter(-INFINITY, 4));
    ASSERT_TRUE(testOneQuarter(1792, 4));
    ASSERT_TRUE(testOneQuarter(1, 4));
    ASSERT_TRUE(testOneQuarter(0.875, 4));
    ASSERT_TRUE(testOneQuarter(0.75, 4));
    ASSERT_TRUE(testOneQuarter(0.625, 4));
    ASSERT_TRUE(testOneQuarter(0.5, 4));
    ASSERT_TRUE(testOneQuarter(0.375, 4));
    ASSERT_TRUE(testOneQuarter(0.125, 4));
    ASSERT_TRUE(testOneQuarter(0, 4));
    ASSERT_TRUE(testOneQuarter(-0.125, 4));
    ASSERT_TRUE(testOneQuarter(-0.375, 4));
    ASSERT_TRUE(testOneQuarter(-0.875, 4));
    ASSERT_TRUE(testOneQuarter(-1, 4));
    ASSERT_TRUE(testOneQuarter(-6, 4));
    ASSERT_TRUE(testOneQuarter(-96, 4));
    ASSERT_TRUE(testOneQuarter(-1792, 4));
    ASSERT_TRUE(testOneQuarter(7680, 2));
    ASSERT_TRUE(testOneQuarter(7168, 2));
    ASSERT_TRUE(testOneQuarter(15360, 1));
    ASSERT_TRUE(testOneQuarter(14336, 1));
    ASSERT_TRUE(testOneQuarter(13312, 1));
    ASSERT_TRUE(testOneQuarter(8192, 1));
    ASSERT_TRUE(testOneQuarter(-14336, 1));
    ASSERT_TRUE(testOneQuarter(-15360, 0));
    for (int i = 0; i < 9; i++) {
        ASSERT_TRUE(minMaxSmallestTest(i));
    }

    ASSERT_FALSE(testOneQuarter(0.00001, 0));
    ASSERT_FALSE(testOneQuarter(-0.2, 0));

    ASSERT_TRUE(testOneQuarter(2, 4));
    ASSERT_TRUE(testOneQuarter(1, 0));
    ASSERT_TRUE(testOneQuarter(-1, 0));

    ASSERT_TRUE(testOneQuarter(quarterToFloat(QUARTER_MIN, 0), 0));
    ASSERT_TRUE(testOneQuarter(quarterToFloat(QUARTER_SECOND_MIN, 0), 0));

    // Test that the second minimum value for bias 0 rounds to minimum value,
    // since the second minimum is used to represent 1.
    const uint32_t second_min_bits = 0b11000110111000000000000000000000;
    const float second_min = *(float *) &second_min_bits; // -28672
    ASSERT_TRUE(floatToQuarter(second_min, 0) == floatToQuarter(quarterToFloat(QUARTER_MIN, 0), 0));

    // Lots of rounding errors, but it is to be expected.
    testAdd(1, 2, 3, 4);
    testAdd(0.5, 10.3, 11, 4);
    testAdd(0.1, 10.1, 10.2, 4);
    testAdd(0.003, 0.003, 0.0087, 0);
    testAdd(0.005, 0.005, 0.0097, 8);
    testAdd(0.0012, 0.0012, 0.00195313, 8);
    testSubtract(0.0012, 0.0012, 0, 8);
    testSubtract(0.5, 0.1, 0.41, 8);
    testMultiply(1, 0.5, 0.5, 8);
    testMultiply(5, 5, 25, 8);
    testDivide(5, 5, 1, 8);
    testDivide(5, 0, INFINITY, 8);
    testDivide(0, 0, NAN, 8);

    testAdd(0.003, 0.003, 0.00585938, 14);
    testAdd(0.0012, 0.0012, 0.00244141, 14);
    testSubtract(0.0012, 0.0012, 0, 14);

}


int main() {
    try {
        testQuarter();

        printConversionsSmallNumbers(0, true);
        printConversionsBigNumbers(0, true);

        printConversionsSmallNumbers(4, true);
        printConversionsBigNumbers(4, true);

        printConversionsSmallNumbers(8, true);
        printConversionsBigNumbers(8, true);

        printConversionsSmallNumbers(14, true);
        printConversionsBigNumbers(14, true);
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
