//
// Created by Erik Hyrkas on 1/13/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_DATA_DECODER_HPP
#define HAPPYML_DATA_DECODER_HPP

#include <string>
#include <locale>
#include "../types/tensor_views/unstandardize_tensor_view.hpp"
#include "../types/tensor_views/denormalize_tensor_view.hpp"

using namespace std;

namespace happyml {

    // I'm unhappy with the current decoder class hierarchy. Because we
    // sometimes need to return strings and sometimes return tensors,
    // it's really clunky. Eventually, we need images and simple numbers
    // as well.
    // This needs to be rethought. I think the core issue is that,
    // when I use it, I try to put all decoders in a single array, but
    // the decoders are then out-of-context of what they need to return.
    //
    // I think that it is generally true that all decoders are used to create
    // a text representation for the end user, but we'd need a way to specify
    // the format of the text, and in the case of tensors, I currently print
    // one row at a time. Maybe I need to make a "TextBlockResponse" class
    // that has the lines of the response, and then the decoder can return
    // that and the caller can decide how to print it.

//    template <typename T>
//    class DataDecoder {
//    public:
//        virtual T decode(shared_ptr<BaseTensor> tensor) = 0;
//    };

    // Noop decoder
    class RawDecoder {
    public:
        explicit RawDecoder(bool isNormalized = false,
                            bool isStandardized = false,
                            float minValue = 0.0f,
                            float maxValue = 0.0f,
                            float mean = 0.0f,
                            float standardDeviation = 0.0f) :
                is_normalized(isNormalized),
                is_standardized(isStandardized),
                min_value(minValue),
                max_value(maxValue),
                mean(mean),
                standard_deviation(standardDeviation) {
        }

        virtual shared_ptr<BaseTensor> decode(const shared_ptr<BaseTensor> &tensor) {
            shared_ptr<BaseTensor> result = tensor;
            if (is_normalized) {
                result = make_shared<DenormalizeTensorView>(result,
                                                            min_value,
                                                            max_value);
            }
            if (is_standardized) {
                result = make_shared<UnstandardizeTensorView>(result,
                                                              mean,
                                                              standard_deviation);
            }
            return result;
        }

        virtual string decodeBest(const shared_ptr<BaseTensor> &tensor) {
            stringstream ss;
            tensor->print(ss);
            return ss.str();
        }

        virtual vector<string> decodeTop(const shared_ptr<BaseTensor> &tensor, const size_t numberOfResults) {
            stringstream ss;
            tensor->print(ss);
            return {ss.str()};
        }

        virtual vector<string> decodeImage(const shared_ptr<BaseTensor> &tensor) {
            stringstream ss;
            tensor->print(ss);
            return {ss.str()};
        }

        // this is terrible design.
        virtual bool isText() {
            return false;
        }

        // this is terrible design.
        virtual bool isImage() {
            return false;
        }

    private:
        bool is_normalized;
        bool is_standardized;
        float min_value;
        float max_value;
        float mean;
        float standard_deviation;
    };

    // TODO: add a "minimum confidence" parameter, where it doesn't return values that are
    //  below a threshold.
    // TODO: could return the confidence with the text
    class BestTextCategoryDecoder : public RawDecoder {
    public:
        explicit BestTextCategoryDecoder(const vector<string> &categoryLabels)
                : categoryLabels_(categoryLabels) {
        }

        string decodeBest(const shared_ptr<BaseTensor> &tensor) override {
            const auto categoryIndex = maxIndex(tensor);
            if (categoryIndex >= categoryLabels_.size()) {
                string message = "Category index out of bounds: " + to_string(categoryIndex);
                throw runtime_error(message);
            }
            return categoryLabels_[categoryIndex];
        }

        vector<string> decodeTop(const shared_ptr<BaseTensor> &tensor, const size_t numberOfResults) override {
            const auto categoryIndex = tensor->topIndices(numberOfResults, 0, 0);
            vector<string> result;
            result.reserve(categoryIndex.size());
            for (const auto &index: categoryIndex) {
                result.push_back(categoryLabels_[index.getIndex()]);
            }
            return result;
        }

        bool isText() override {
            return true;
        }

    private:
        vector<string> categoryLabels_;
    };

    // The whole decoder class hierarchy needs to be rethought. I think that
    // this is the worst possible way to do it. I am just hacking this in
    // because I don't have time to rebuild this right now with my other priorities.
    // I think there probably should be one decoder class that can return
    // text, images, and tensors, and then the caller can decide what to do with it.
    class ImageDecoder : public RawDecoder {
    public:
        bool isImage() override {
            return true;
        }

        vector<string> decodeImage(const shared_ptr<BaseTensor> &tensor) override {
            vector<string> result;
            size_t rows = tensor->rowCount();
            size_t cols = tensor->columnCount();
            size_t channels = tensor->channelCount();
//            vector<char> ascii_chars = { ' ', '.', ':', '-', '=', '+', '*', '#', '%', '@' };
            vector<char> ascii_chars = {' ', (char) 176, (char) 177, (char) 178, (char) 219};

            size_t ascii_len = ascii_chars.size();

            // we're making ascii art and the characters are twice as tall as they are wide
            for (size_t row = 0; row < rows; row += 2) {
                stringstream ss;
                for (size_t col = 0; col < cols; col++) {
                    // build a gray-scale value for the combined channels
                    // we also have to combine the two rows, if there are two
                    // rows
                    float grayScale = 0.0f;
                    if (channels >= 3) {
                        float red = tensor->getValue(row, col, 0);
                        float green = tensor->getValue(row, col, 1);
                        float blue = tensor->getValue(row, col, 2);
                        grayScale = 0.299f * red + 0.587f * green + 0.114f * blue;
                        if (row + 1 < rows) {
                            red = tensor->getValue(row + 1, col, 0);
                            green = tensor->getValue(row + 1, col, 1);
                            blue = tensor->getValue(row + 1, col, 2);
                            grayScale += 0.299f * red + 0.587f * green + 0.114f * blue;
                            grayScale /= 2.0f;
                        }
                    } else if (channels == 1) {
                        grayScale = tensor->getValue(row, col, 0);
                        if (row + 1 < rows) {
                            grayScale += tensor->getValue(row + 1, col, 0);
                            grayScale /= 2.0f;
                        }
                    }
                    grayScale /= (float) (channels * (row + 1 < rows ? 2 : 1));
                    // Normalize the grayscale value to range [0, 1]
                    grayScale = (grayScale < 0.0f) ? 0.0f : ((grayScale > 1.0f) ? 1.0f : grayScale);

                    // Map the grayscale value to an ASCII character
                    auto ascii_index = (size_t) (grayScale * (float) (ascii_len - 1));
                    ss << ascii_chars[ascii_index];
                }
                result.push_back(ss.str());
            }
            return result;
        }
    };
}

#endif //HAPPYML_DATA_DECODER_HPP