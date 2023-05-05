//
// Created by Erik Hyrkas on 11/27/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_DATA_ENCODER_HPP
#define HAPPYML_DATA_ENCODER_HPP

#include <string>
#include <charconv>
#include <algorithm>
#include <cctype>
#include <locale>
#include <map>
#include <utility>
#include "../types/tensor.hpp"
#include "../util/tensor_utils.hpp"
#include "../ml/byte_pair_encoder.hpp"
#include "../ml/rotary_positional_embedding.hpp"
#include "../util/one_hot_encoder.hpp"

using namespace std;

namespace happyml {

    string strip(string text) {
        // we intentionally copy the string so we can trim it
        text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), text.end());
        return text;
    }

    string trimEnd(string text) {
        // we intentionally copy the string so we can trim it
        text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), text.end());
        return text;
    }

    float stringToFloat(const string &text) {
        float value = 0.0f;
        if (!text.empty()) {
            auto [ptr, error_check] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (error_check != std::errc()) {
                string message = "Couldn't convert text to float: " + text;
                throw exception(message.c_str());
            }
        }
        return value;
    }

    bool stringToFloat(const string &text, float &value) {
        auto [ptr, error_check] = std::from_chars(text.data(), text.data() + text.size(), value);
        return error_check == std::errc();
    }

    bool isFloat(const string &text) {
        float value = 0.0;
        auto [ptr, error_check] = std::from_chars(text.data(), text.data() + text.size(), value);
        return error_check == std::errc();
    }

    class DataEncoder {
    public:
        virtual shared_ptr<BaseTensor> encode(const vector<string> &words,
                                              size_t rows, size_t columns, size_t channels, bool trim) = 0;

        virtual vector<size_t> calculate_output_shape(size_t rows, size_t columns, size_t channels) = 0;
    };


    class TextToPixelEncoder : public DataEncoder {
    public:
        shared_ptr<BaseTensor> encode(const vector<string> &words,
                                      size_t rows, size_t columns, size_t channels, bool trim) override {
            // it is wasteful to allocate a huge vector only to copy it into a tensor
            // however, I wanted tensors to be immutable.
            // TODO: I think I could make a FullTensor constructor that steals the memory we are allocating here.
            // TODO: I could also make a PixelTensor that steals the memory of a vector using a uint_8.
            // Of course, then I'll need a string_to_uint_8() method.
            vector<vector<vector<float>>> result;
            result.resize(channels);
            size_t offset = 0;
            for (size_t channel = 0; channel < channels; channel++) {
                result[channel].resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    result[channel][row].resize(columns);
                    for (size_t column = 0; column < columns; column++) {
                        string word = trim ? strip(words[offset]) : words[offset];
                        // we store the value as percentage of 255.
                        result[channel][row][column] = stringToFloat(word) / 255.f;
                        offset++;
                    }
                }
            }
            return pixelTensor(result);
        }

        vector<size_t> calculate_output_shape(size_t rows, size_t columns, size_t channels) override {
            return {rows, columns, channels};
        }
    };

    class TextToScalarEncoder : public DataEncoder {
    public:
        shared_ptr<BaseTensor> encode(const vector<string> &words,
                                      size_t rows, size_t columns, size_t channels, bool trim) override {
            vector<vector<vector<float>>> result;
            result.resize(channels);
            size_t offset = 0;
            for (size_t channel = 0; channel < channels; channel++) {
                result[channel].resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    result[channel][row].resize(columns);
                    for (size_t column = 0; column < columns; column++) {
                        string word = trim ? strip(words[offset]) : words[offset];
                        result[channel][row][column] = stringToFloat(word);
                        offset++;
                    }
                }
            }
            return tensor(result);
        }

        vector<size_t> calculate_output_shape(size_t rows, size_t columns, size_t channels) override {
            return {rows, columns, channels};
        }
    };

    // See get_distinct_values() in dataset_utils.hpp for how to calculate the category labels.
    class TextToUniqueCategoryEncoder : public DataEncoder {
    public:
        explicit TextToUniqueCategoryEncoder(const vector<string> &categoryLabels) {
            for (size_t index = 0; index < categoryLabels.size(); index++) {
                categoryMapping[categoryLabels[index]] = index;
            }

        }

        explicit TextToUniqueCategoryEncoder(const map <string, size_t> &categoryMapping) {
            this->categoryMapping = categoryMapping;
        }

        shared_ptr<BaseTensor> encode(const vector<string> &words,
                                      size_t rows, size_t columns, size_t channels, bool trim) override {
            if (channels != 1) {
                throw exception("The result tensor must have exactly one channel.");
            }
            if (words.size() != rows) {
                string message = "The result tensor must have exactly the same number of rows as there are words to encode. Expected " + to_string(rows) + " but got " + to_string(words.size());
                throw exception(message.c_str());
            }
            if (columns != categoryMapping.size()) {
                string message = "The result tensor must have exactly the same number of columns as there are categories. Expected " + to_string(categoryMapping.size()) + " but got " + to_string(columns);
                throw exception(message.c_str());
            }
            vector<vector<vector<float>>> result;
            result.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                result[channel].resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    result[channel][row].resize(columns);
                }
            }

            size_t row_offset = 0;
            for (const auto &word: words) {
                size_t column_offset;
                if (trim) {
                    column_offset = categoryMapping.at(strip(word));
                } else {
                    column_offset = categoryMapping.at(word);
                }
                if (row_offset >= rows) {
                    throw exception("mapping returned an out of bounds index for rows.");
                }
                if (column_offset >= columns) {
                    throw exception("mapping returned an out of bounds index for columns.");
                }
                result[0][row_offset][column_offset] = 1.f;
                row_offset++;
            }

            return tensor(result);
        }

        vector<size_t> calculate_output_shape(size_t rows, size_t columns, size_t channels) override {
            return {rows, categoryMapping.size(), 1};
        }

    private:
        map <string, size_t> categoryMapping;
    };

    class TextToEmbeddedTokensEncoder : public DataEncoder {
    public:
        explicit TextToEmbeddedTokensEncoder(shared_ptr<BytePairEncoderModel> bytePairEncoderModel,
                                             shared_ptr<Embedder> embedder) :
                bytePairEncoderModel_(std::move(bytePairEncoderModel)),
                embedder_(std::move(embedder)) {}

        shared_ptr<BaseTensor> encode(const vector<string> &columnsOfText,
                                      size_t rows, size_t columns, size_t channels, bool trim) override {
            // I considered just doing character-level encoding using one_hot_encode_characters(), but
            // it would take a huge amount of memory and not produce good results. It's better to stop
            // the caller now and let them fix their code.
            if (bytePairEncoderModel_ == nullptr || embedder_ == nullptr) {
                throw exception("BytePairEncoderModel and Embedder must be set.");
            }
            size_t largest_bpe_code = bytePairEncoderModel_->getLargestCode();

            vector<vector<vector<float>>> result;
            result.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                result[channel].resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    result[channel][row].resize(columns);
                }
            }

            size_t channel_offset = 0;
            for (const auto &columnOfText: columnsOfText) {
                vector<string> tokens = trim ? string_to_tokens(strip(columnOfText)) : string_to_tokens(columnOfText);
                vector<u16string> bpe_encoded_tokens = bytePairEncoderModel_->encode(tokens);
                auto one_hot_encoded = one_hot_encode_bpe_tokens(bpe_encoded_tokens, largest_bpe_code);
                auto embedded_tokens = embedder_->embed_tokens(one_hot_encoded);
                result[channel_offset] = embedded_tokens;
                channel_offset++;
                if (channel_offset >= channels) {
                    break;
                }
            }

            return tensor(result);
        }

        vector<size_t> calculate_output_shape(size_t rows, size_t columns, size_t channels) override {
            auto embedded_shape = embedder_->calculate_result_shape();
            return {embedded_shape[0], embedded_shape[1], channels};
        }

    private:
        shared_ptr<BytePairEncoderModel> bytePairEncoderModel_;
        shared_ptr<Embedder> embedder_;
    };

}
#endif //HAPPYML_DATA_ENCODER_HPP
