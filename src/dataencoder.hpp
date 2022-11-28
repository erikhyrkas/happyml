//
// Created by Erik Hyrkas on 11/27/2022.
//

#ifndef MICROML_DATAENCODER_HPP
#define MICROML_DATAENCODER_HPP

#include <string>
#include <charconv>
#include "tensor.hpp"

using namespace std;

namespace microml {

    class TrainingDataInputEncoder {
        virtual shared_ptr<BaseTensor> encode(const vector<string> &line, size_t rows, size_t columns, size_t channels) = 0;
    };


    class TextToPixelEncoder : public TrainingDataInputEncoder {

        shared_ptr<BaseTensor> encode(const vector<string> &line, size_t rows, size_t columns, size_t channels) override {
//            float value = 0.0;
//            auto [ptr, error_check] = std::from_chars(text.data(), text.data() + text.size(), value);
//            if (error_check != std::errc()) {
//                throw exception("Couldn't convert text to float");
//            }
            return nullptr;
        }
    };

    class TextToScalarEncoder : public TrainingDataInputEncoder {

        shared_ptr<BaseTensor> encode(const vector<string> &line, size_t rows, size_t columns, size_t channels) override {
//            float value = 0.0;
//            auto [ptr, error_check] = std::from_chars(text.data(), text.data() + text.size(), value);
//            if (error_check != std::errc()) {
//                throw exception("Couldn't convert text to float");
//            }
//            return column_vector({value});
            return nullptr;
        }
    };

}
#endif //MICROML_DATAENCODER_HPP
