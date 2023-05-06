//
// Created by Erik Hyrkas on 1/13/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_DATA_DECODER_HPP
#define HAPPYML_DATA_DECODER_HPP

#include <string>
#include <locale>
#include "../types/tensor_views/tensor_unstandardize_view.hpp"
#include "../types/tensor_views/tensor_denormalize_view.hpp"

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
                result = make_shared<TensorDenormalizeView>(result,
                                                            min_value,
                                                            max_value);
            }
            if (is_standardized) {
                result = make_shared<TensorUnstandardizeView>(result,
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

        virtual bool isText() {
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
}

#endif //HAPPYML_DATA_DECODER_HPP