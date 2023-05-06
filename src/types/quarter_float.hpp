//
// Created by Erik Hyrkas on 10/23/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_QUARTER_FLOAT_HPP
#define HAPPYML_QUARTER_FLOAT_HPP

#include <cstdint>

#define FLOAT_BIAS 127
#define FLOAT_NEG_NAN 0b11111111110000000000000000000000
#define FLOAT_POS_NAN 0b01111111110000000000000000000000
#define FLOAT_INF 0b01111111100000000000000000000000
#define FLOAT_NEG_INF 0b11111111100000000000000000000000
#define QUARTER_NAN 0b11111111
#define QUARTER_POS_INFINITY 0b01111000
#define QUARTER_NEG_INFINITY 0b11111000
#define QUARTER_MAX_EXPONENT_AMOUNT 15
#define QUARTER_MAX 0b01110111
#define QUARTER_MIN 0b11110111
#define QUARTER_SMALLEST 0b00000001
#define QUARTER_SECOND_SMALLEST 0b00000010
#define QUARTER_SECOND_MIN 0b11110110

using namespace std;

namespace happyml {

    typedef unsigned char quarter;

    void printBits(const quarter x) {
        for (int i = 7; i >= 0; i--) {
            cout << ((x >> i) & 1);
            if (i == 7 || i == 3) { cout << " "; }
        }
        cout << endl;
    }

    void printBits(const uint32_t x) {
        for (int i = 31; i >= 0; i--) {
            cout << ((x >> i) & 1);
            if (i == 31 || i == 23) { cout << " "; }
        }
        cout << endl;
    }

    void printBits(const float x) {
        uint32_t b = *(uint32_t *) &x;
        printBits(b);
    }


// A bigger bias will let you represent more fine-grained numbers, but at the cost of a smaller max and min value.
// A bias of 8 is good for a granularity of about 0.005, where a bias of 4 only goes down to 0.125.
// Of course, a bias of 8 has a max value of 120, where a bias of 4 has a max value of 1792.
//
// We subtract the offset from the float before we convert it. This may allow us to have more granularity, but
// only in a specific range of the spectrum.
//
// Imagine if you have an array of a million numbers. If the average of those numbers is 5000, we'd need a big
// enough bias to hold it. However, since we have so few bits, our granularity would be very bad. If we know
// what our average is, we could use the offset to make our 8-bit float represent the distance away from
// that offset. This would potentially let us use a smaller bias to represent numbers in that range.
//
// So before we set our bias and offset, we have to understand what we want our range of numbers to be, with
// an understanding of roughly where the middle of that range will be (offset) so that we can use the best
// bias possible.
//
// With ML we use large data sets with many passes. We might not know the optimal offset or bias on the first
// pass, but we can slowly adjust with each pass to hone in on the correct offset.
//
// NOTE ON THIS IMPLEMENTATION:
// I've been through a few rounds of testing and this is where I'm at now.
// This version avoids rounding to 0 and will change bias 0 to bias 1, because bias 0 would need
// special handling to represent 1 and -1, which are important values in machine learning. You can look at the
// other versions of this function at the bottom of this file.
    quarter floatToQuarter(float original, int quarter_bias) {
        // This code looks crazy because I avoid branching (if statements) so the general case (happy path) is faster.
        // Convert the float to a datatype where we can easily do bit manipulation
        // avoid bias 0 because it would require special handling to represent 1 and -1
        const int bias = (quarter_bias == 0) + (quarter_bias != 0) * quarter_bias;
        const uint32_t encoded_value = (*(uint32_t *) &original);
        // grab the sign, we can use this as is
        const uint32_t sign = encoded_value >> 31;
        const uint32_t raw_exponent = (encoded_value & 0x7F800000) >> 23;
        const uint32_t raw_mantissa = (encoded_value & 0x00700000) >> 20;
        const uint32_t extra_mantissa = (encoded_value & 0x000FFFFF) >> 19;
        const uint32_t rounded_mantissa = raw_mantissa + (raw_mantissa < 0b111 && (extra_mantissa > 0)) * 0b001;
        const uint32_t float_bias_adjustment = FLOAT_BIAS - bias;
        const uint32_t quarter_max_exponent = float_bias_adjustment + QUARTER_MAX_EXPONENT_AMOUNT;
        // I'm pretty sure this can be optimized further
        const uint32_t clamp_top = raw_exponent >= quarter_max_exponent;
        const uint32_t clamp_bottom = raw_exponent < float_bias_adjustment;
        const uint32_t clamped = clamp_top || clamp_bottom;
        const int32_t adjusted_exponent = (int32_t) raw_exponent - (int32_t) float_bias_adjustment;
        const uint32_t exponent = (!clamped) * adjusted_exponent + (clamp_top * 0xE);
        const uint32_t clamped_exponent = (exponent < 0xF) * exponent + (exponent == 0xF) * 0xE;
        const uint32_t clamped_mantissa = ((!clamped) * rounded_mantissa) + (clamp_top * 0x7);
        const uint32_t prepped_bits = (sign << 7) | (clamped_exponent << 3) | (clamped_mantissa & 0x7);
        const uint32_t special_case_tiny =
                encoded_value && (!prepped_bits && adjusted_exponent >= (-12 + bias)); // -5 rounds nearest
        const uint32_t result_bits = (special_case_tiny) * 0b10000000 +
                                     (!special_case_tiny) * prepped_bits;

        return (encoded_value == FLOAT_NEG_NAN || encoded_value == FLOAT_POS_NAN) * QUARTER_NAN +
               (encoded_value == FLOAT_INF) * QUARTER_POS_INFINITY
               + (encoded_value == FLOAT_NEG_INF) * QUARTER_NEG_INFINITY +
               ((encoded_value != FLOAT_NEG_NAN && encoded_value != FLOAT_POS_NAN) && (encoded_value != FLOAT_INF) &&
                (encoded_value != FLOAT_NEG_INF)) *
               result_bits;
    }


    float quarterToFloat(quarter q, int quarter_bias) {
        // This code looks crazy because I avoid branching (if statements) so the general case (happy path) is faster.
        // avoid bias 0 because it would require special handling to represent 1 and -1
        const int bias = (quarter_bias == 0) + (quarter_bias != 0) * quarter_bias;
        const uint32_t special_case_tiny = 0b10000000 == q;
        const uint32_t sign = (!special_case_tiny) * (q >> 7);
        // 0b111111111 is 9 bits. that looks wrong but works. I can't remember why.
        const uint32_t raw_exponent = (!special_case_tiny) * ((q & 0x78) >> 3) + (special_case_tiny) * 0b111111111;
        const uint32_t mantissa = (!special_case_tiny) * (q & 0x7) + (special_case_tiny) * 0b1;
        const uint32_t mantissa_shift = (!special_case_tiny * 20) + (special_case_tiny) * 1;
        const uint32_t float_bias_adjustment = FLOAT_BIAS - bias;
        const uint32_t exponent = (raw_exponent > 0 || mantissa > 0) * (raw_exponent + float_bias_adjustment);
        const uint32_t result = ((sign << 31) | (exponent << 23) | (mantissa << mantissa_shift));
        const uint32_t no_branch_result = (q == QUARTER_NAN) * FLOAT_NEG_NAN +
                                          (q == QUARTER_POS_INFINITY) * FLOAT_INF +
                                          (q == QUARTER_NEG_INFINITY) * FLOAT_NEG_INF +
                                          (q != QUARTER_NAN && q != QUARTER_POS_INFINITY && q != QUARTER_NEG_INFINITY) *
                                          result;

        const float distance_from_offset = *(float *) &no_branch_result;
        return distance_from_offset;
    }

    quarter quarterMultiply(quarter a, int a_bias, quarter b, int b_bias, int result_bias) {
        const float af = quarterToFloat(a, a_bias);
        const float bf = quarterToFloat(b, b_bias);
        const float result_float = af * bf;
        return floatToQuarter(result_float, result_bias);
    }

    quarter quarterDivide(quarter a, int a_bias, quarter b, int b_bias, int result_bias) {
        const float af = quarterToFloat(a, a_bias);
        const float bf = quarterToFloat(b, b_bias);
        const float result_float = af / bf;
        return floatToQuarter(result_float, result_bias);
    }

    quarter quarterAdd(quarter a, int a_bias, quarter b, int b_bias, int result_bias) {
        const float af = quarterToFloat(a, a_bias);
        const float bf = quarterToFloat(b, b_bias);
        const float result_float = af + bf;
        return floatToQuarter(result_float, result_bias);
    }

    quarter quarterSubtract(quarter a, int a_bias, quarter b, int b_bias, int result_bias) {
        const float af = quarterToFloat(a, a_bias);
        const float bf = quarterToFloat(b, b_bias);
        const float result_float = af - bf;
        return floatToQuarter(result_float, result_bias);
    }

    float calculateBiasRange(int bias) {
        const float min_for_bias = quarterToFloat(QUARTER_MIN, bias);
        const float max_for_bias = quarterToFloat(QUARTER_MAX, bias);
        return std::abs(max_for_bias - min_for_bias);
    }

// we're very, very approximate in this comparison, taking 1000x epsilon.
    template<typename T>
    static bool roughlyEqual(T f1, T f2) {
        T abs_diff = std::fabs(f1 - f2);
        return (abs_diff <= std::numeric_limits<T>::epsilon() * 1000.0) || (abs_diff < std::numeric_limits<T>::min());
    }
}
#endif //HAPPYML_QUARTER_FLOAT_HPP