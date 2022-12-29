//
// Created by Erik Hyrkas on 12/1/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_HALF_HPP
#define HAPPYML_HALF_HPP

#include <cstdint>

namespace happyml {

    typedef uint16_t half;

    half floatToHalf(float original) {
        // TODO: handle infinity and NAN
        const uint32_t encoded_value = (*(uint32_t *) &original);
        return (half) (encoded_value >> 16);
    }

    float halfToFloat(half h) {
        // TODO: handle infinity and NAN
        const uint32_t shifted_value = h << 16;
        const float decoded_value = *(float *) &shifted_value;
        return decoded_value;
    }
}
#endif //HAPPYML_HALF_HPP
