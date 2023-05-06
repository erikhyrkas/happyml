//
// Created by Erik Hyrkas on 4/17/2023.
// Copyright 2023. Usable under MIT license.
//

#include "../util/timers.hpp"
#include "../util/unit_test.hpp"
#include "../types/half_float.hpp"

using namespace std;
using namespace happyml;

void test_pos_infinity() {
    float i = INFINITY;
//    printBits(i);
    half h = floatToHalf(i);
//    printBits(h);
    float f = halfToFloat(h);
//    printBits(f);
    ASSERT_TRUE(isinf(f));
}


void test_neg_infinity() {
    float i = -INFINITY;
//    printBits(i);
    half h = floatToHalf(i);
//    printBits(h);
    float f = halfToFloat(h);
//    printBits(f);
    ASSERT_TRUE(isinf(f));
}

void test_not_pos_infinity() {
    uint32_t bits = 0b01111111100000000000000000000010;
    float i = *(float*)&bits;
//    cout << i << endl;
//    printBits(i);
    half h = floatToHalf(i);
//    printBits(h);
    float f = halfToFloat(h);
//    printBits(f);
//    cout << f << endl;
    ASSERT_TRUE(isnan(f));
}

void test_not_neg_infinity() {
    uint32_t bits = 0b11111111100000000000000000000010;
    float i = *(float*)&bits;
//    cout << i << endl;
//    printBits(i);
    half h = floatToHalf(i);
//    printBits(h);
    float f = halfToFloat(h);
//    printBits(f);
//    cout << f << endl;
    ASSERT_TRUE(isnan(f));
}


void test_pos_nan() {
    float i = NAN;
//    printBits(i);
    half h = floatToHalf(i);
//    printBits(h);
    float f = halfToFloat(h);
//    printBits(f);
    ASSERT_TRUE(isnan(f));
}


void test_neg_nan() {
    float i = -NAN;
//    printBits(i);
    half h = floatToHalf(i);
//    printBits(h);
    float f = halfToFloat(h);
//    printBits(f);
    ASSERT_TRUE(isnan(f));
}

int main() {
    try {
        EvenMoreSimpleTimer timer;
        test_neg_infinity();
        timer.printMicroseconds();
        test_pos_infinity();
        timer.printMicroseconds();
        test_not_neg_infinity();
        timer.printMicroseconds();
        test_not_pos_infinity();
        timer.printMicroseconds();
        test_pos_nan();
        timer.printMicroseconds();
        test_neg_nan();
        timer.printMicroseconds();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}