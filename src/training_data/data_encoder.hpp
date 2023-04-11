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
#include "../types/tensor.hpp"
#include "../util/tensor_utils.hpp"

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
        float value = 0.0;
        auto [ptr, error_check] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error_check != std::errc()) {
            throw exception("Couldn't convert text to float");
        }
        return value;
    }

    class DataEncoder {
    public:
        virtual shared_ptr<BaseTensor> encode(const vector<string> &words,
                                              size_t rows, size_t columns, size_t channels, bool trim) = 0;
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
    };

    // TODO: we should be able to calculate category labels from a column
    class TextToUniqueCategoryEncoder : public DataEncoder {
    public:
        explicit TextToUniqueCategoryEncoder(const vector<string> &categoryLabels) {
            for (size_t index = 0; index < categoryLabels.size(); index++) {
                categoryMapping[categoryLabels[index]] = index;
            }

        }

        explicit TextToUniqueCategoryEncoder(const map<string, size_t> &categoryMapping) {
            this->categoryMapping = categoryMapping;
        }

        shared_ptr<BaseTensor> encode(const vector<string> &words,
                                      size_t rows, size_t columns, size_t channels, bool trim) override {
            if (words.size() != channels) {
                throw exception(
                        "The result tensor must have exactly the same number of channels as there are words to encode.");
            }
            vector<vector<vector<float>>> result;
            result.resize(channels);
            for (size_t channel = 0; channel < channels; channel++) {
                result[channel].resize(rows);
                for (size_t row = 0; row < rows; row++) {
                    result[channel][row].resize(columns);
                }
            }

            size_t channel_offset = 0;
            for (const auto &word: words) {
                size_t column_offset;
                if (trim) {
                    column_offset = categoryMapping.at(strip(word));
                } else {
                    column_offset = categoryMapping.at(word);
                }
                result[channel_offset][0][column_offset] = 1.f;
                channel_offset++;
            }

            return tensor(result);
        }

    private:
        map<string, size_t> categoryMapping;
    };


}
#endif //HAPPYML_DATA_ENCODER_HPP
