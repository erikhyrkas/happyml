//
// Created by Erik Hyrkas on 12/1/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_HALF_HPP
#define HAPPYML_HALF_HPP

#include "quarter_float.hpp"

#define HALF_POS_NAN 0b0111111110000001
#define HALF_NEG_NAN 0b1111111110000001
#define HALF_POS_INF 0b0111111110000000
#define HALF_NEG_INF 0b1111111110000000
namespace happyml {

    typedef uint16_t half;

    void printBits(const half x) {
        for (int i = 15; i >= 0; i--) {
            cout << ((x >> i) & 1);
            if (i == 15 || i == 7) cout << " ";
        }
        cout << endl;
    }

    half floatToHalf(float original) {
        const uint32_t encoded_value = (*(uint32_t *) &original);
        if (isnan(original)) {
            if (encoded_value == FLOAT_NEG_NAN) {
                return HALF_NEG_NAN;
            }
            return HALF_POS_NAN;
        } else if (isinf(original)) {
            if (encoded_value == FLOAT_NEG_INF) {
                return HALF_NEG_INF;
            }
            return HALF_POS_INF;
        }
        return (half) (encoded_value >> 16);
    }

    float halfToFloat(half h) {
        const uint32_t shifted_value = h << 16;
        const float decoded_value = *(float *) &shifted_value;
        return decoded_value;
    }
}
#endif //HAPPYML_HALF_HPP
