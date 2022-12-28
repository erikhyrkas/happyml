//
// Created by Erik Hyrkas on 10/23/2022.
// Copyright 2022. Usable under MIT license.
//

// A 32-bit float has the general structure of: sign bit + 8 exponent bits + 23 mantissa bits
//
// There is an extra hidden constant called bias that is 127, which we'll cover below.
//
// The sign bit is really straight forward:
// sign 0 = positive, sign 1 = negative
//
// The exponent bits have a small twist:
// The exponent bits can be converted to an unsigned int, but we have to subtract the bias to get the actual exponent.
// The bias for a standard float is 127. We need that bias so that we can represent negative exponents, which
// in turn lets us have fractions. In our future calculations, we are going to raise 2 to the power of this
// exponent.
//
// For our 8-bit float's exponent, we'll make the bias dynamic. A bias of 8 allows for reasonable accuracy between 0 and 1
// where a bias of 0 allows larger possible numbers, but with lower accuracy.
//
// The mantissa bits may require you to read this twice:
// You've probably seen scientific notation before. A number like 6.23e10. That's 6.23 x 2^10.
// We have en exponent already from the previous exponent bits, so we know what is going on
// the right side of the 'e', but that number on the left of the e? That's the mantissa.
//
// We don't have the mantissa itself yet, just some bits that we can use to calculate it.
//
// The mantissa bits don't just hold a number we use like the 6.23 in our example of 6.23e10. These bits
// hold a number written in base two that we have to decode and do math on to get to a number that you'd recognize.
//
// What's might throw you is that the bits are on the right side of a decimal point, so the exponents we use
// are going to take a moment to adjust to.
//
// This has to do with having the most significant bits (the values that have the greatest impact) be the closest
// to zero, which in our case is the left. We do this when we write 6.23. The two (2) is more significant than the 3.
// We're doing the same thing in base 2.
//
// Say we have a three bit mantissa, which we will.
//
// You might think that you know how to read a binary number:
// 110 = 2^2 + 2^1 + 0   = 4+2+0 = 6
// 100 = 2^2 +   0 + 0   = 4+0+0 = 4
// 010 =   0 + 2^1 + 0   = 0+2+0 = 2
// 001 =   0 +   0 + 2^0 = 0+0+1 = 1
// 111 = 2^2 + 2^1 + 2^0 = 4+2+1 = 7
//
// THIS IS WRONG! Unfortunately, the mantissa bits represent one plus the faction of a number.
//
// Let's start with 100. This is actually a mantissa of 1.5. So, how do we get there?
// The first thing to know is that the mantissa always has a constant of one added to it. The hidden bit.
// The other thing we do is read each bit from left to right, like we are looking in a mirror, using
// a negative number for the power.
//
// So, if we have the bits 100, we have a single bit to the right of our constant whole number 1 (we could have
// written the mantissa bits as 1.100) we do: 1 + 2^-1 = 1 + 0.5 = 1.5.
//
// How about 001? We write in our constant leading number, so we have 1.001, then we do the math:
// 1 + 0 + 0 + 2^-3 = 1 + 0.125 = 1.125.
//
// Let's do some more for practice:
// bits: base 2 conversion:       decimal addition:        result:
// 100 = 1 + 2^-1 +    0 +    0 = 1 + 0.5                = 1.5
// 010 = 1 +    0 + 2^-2 +    0 = 1       + 0.25         = 1.25
// 001 = 1 +    0 +    0 + 2^-3 = 1              + 0.125 = 1.125
// 110 = 1 + 2^-1 + 2^-2 +    0 = 1 + 0.5 + 0.25         = 1.75
// 111 = 1 + 2^-1 + 2^-2 + 2^-3 = 1 + 0.5 + 0.25 + 0.125 = 1.875
//
// Now that we have the mantissa, we can use it in a formula to build an actual number:
// (-1^sign) * (2^exponent) * (mantissa)
//
// My examples will focus on an 8-bit float, since that's easier to show the work for and
// what we are building. A 32-bit float works the same, with just a bigger bias and more
// bits.
//
// Bits:        Math:                                         Show work:       Result:
// 1 1111 111 = special case                                                     =  NAN
// 0 1111 000 = special case                                                     =  Positive Infinity
// 1 1111 000 = special case                                                     =  Negative Infinity
// 1 0000 000 = super special case see below
// For a bias of 0:
// 0 1110 111 = (-1^0) * (2^14) * (1 + 2^-1 + 2^-2 + 2^-3)  =  1 * 16384 * 1.875 =   30720
// 0 0001 001 = (-1^0) * (2^1)  * (1 + 2^-3)                =  1 * 2 * 1.125     =   2.25
// 0 0000 010 = (-1^0) * (2^0)  * (1 + 2^-2)                =  1 * 1 * 1.25      =   1.25
// 0 0000 001 = (-1^0) * (2^0)  * (1 + 2^-3)                =  1 * 1 * 1.125     =   1.125
// 0 0000 000 = (-1^0) * (2^0)  * 0                         =  1 * 1 * 0         =   0
// 1 1110 111 = (-1^1) * (2^14) * (1 + 2^-1 + 2^-2 + 2^-3)  = -1 * 16384 * 1.875 =  -30720
//
// Super special note on 10000000.
// When we try to represent numbers less than our bias can handle, something
// interesting happens: For bias 0, let's take the floating point 1.0 and convert it. 1.0f
// becomes 0 01111111 00000000000000000000000, well sometimes it becomes 0 01111111 00000000000000000000001, but
// you get the point. It's 2^0 power + ~0. At bias 0 though, we can only make 0 0111 000, which is 1.25. So,
// what happens if we round down? We get 0 0000 000, which is zero.
//
// By default, 10000000 represents -0. Negative zero is mathematically equivalent to positive zero. It feels
// terrible to waste this one permutation on something that is largely unused (there are some weird cases
// where people have used -0 for their own evil purposes in some libraries, but I'm not inclined to support it.)
//
// I'm inclined to use 10000000 to support something like 1/3 (0.33333...) or pi (3.14...) or euler's number (2.718...).
//
// I don't expect people to use the 8-bit numbers to hold constants. I expect them to hold the results of math done
// with 32-bit numbers. So, if they do 2*pi, I might need to hold that. Or if they raise e to the fourth power,
// I might need to represent that. This means that I'm not just picking between a couple different common constants
// but between values that I'll need to represent.
//
// NOTE: Coming back to this. Bias 0 has no representation of 1 or -1.
// SECOND NOTE: Only for Bias 0, I'm going to use 1 1110 110 to represent 1 and 1 0000 000 to represent -1.
// We lose our second most negative number, but we can represent 1 which is highly useful.
// THIRD NOTE: I've observed that representing the space between the smallest representable value and zero is
// larger than other gaps and is problematic. For bias other than 0, I'm going to use 10000000 to represent half
// of the current smallest value of that bias.
//
//
// Offset
// Let's talk for a moment about offset, which is my way of shifting the range of numbers that the 8-bit float will
// represent. Bias impacts our granularity at the cost of it's maximum range. If we want to represent big-ish numbers
// we can't have a high granularity (large bias) without doing something. By using an offset, we might be able
// to achieve a decently high bias to allow good granularity, but only around that offset.
//
// In general purpose computing, we may never know what a decent offset is, but in machine learning, we make many
// passes trying to find the optimal numbers, honing our numbers as we go. If we find all of our numbers are at the
// top end of our range that the bias we picked is using, by switching the offset on the next pass we can slowly
// hone in on the area that we want to actually look at.
//
// Clearly, this solution doesn't let us have our cake and eat it too. A high bias means a small range. We can
// never have a large range and great granularity with this solution. My hope instinct is that this should be
// fine for ML even if it isn't great for general computing.
//
//
// UPDATE: I've decided to remove offset. For ML, the bias within the last layer already deals with offset
// at a model level, and we don't need to account for it everywhere. It just makes things more complicated
// without making them truly better.

#ifndef HAPPYML_QUARTER_FLOAT_HPP
#define HAPPYML_QUARTER_FLOAT_HPP

//#include <array>
#include <cstdint>

#define FLOAT_BIAS 127
#define FLOAT_NAN 0b11111111110000000000000000000000
#define FLOAT_NAN2 0b01111111110000000000000000000000
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
            if (i == 7 || i == 3) cout << " ";
        }
        cout << endl;
    }

    void printBits(const uint32_t x) {
        for (int i = 31; i >= 0; i--) {
            cout << ((x >> i) & 1);
            if (i == 31 || i == 23) cout << " ";
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

        return (encoded_value == FLOAT_NAN || encoded_value == FLOAT_NAN2) * QUARTER_NAN + (encoded_value == FLOAT_INF) * QUARTER_POS_INFINITY
               + (encoded_value == FLOAT_NEG_INF) * QUARTER_NEG_INFINITY +
               ((encoded_value != FLOAT_NAN && encoded_value != FLOAT_NAN2) && (encoded_value != FLOAT_INF) && (encoded_value != FLOAT_NEG_INF)) *
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
        const uint32_t no_branch_result = (q == QUARTER_NAN) * FLOAT_NAN +
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