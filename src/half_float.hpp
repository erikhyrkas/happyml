//
// Created by Erik Hyrkas on 12/1/2022.
//

#ifndef MICROML_HALF_HPP
#define MICROML_HALF_HPP

#include <cstdint>

namespace microml {

    typedef uint16_t half;

    half float_to_half(float original) {
        // TODO: handle infinity and NAN
        const uint32_t encoded_value = (*(uint32_t *) &original) ;
        return (half) (encoded_value >> 16);
    }

    float half_to_float(half h) {
        // TODO: handle infinity and NAN
        const uint32_t shifted_value = h << 16;
        const float decoded_value = *(float *) &shifted_value;
        return decoded_value;
    }
}
#endif //MICROML_HALF_HPP
