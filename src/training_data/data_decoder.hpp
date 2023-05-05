//
// Created by Erik Hyrkas on 1/13/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_DATA_DECODER_HPP
#define HAPPYML_DATA_DECODER_HPP

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

        virtual shared_ptr <BaseTensor> decode(const shared_ptr <BaseTensor> &tensor) {
            shared_ptr < BaseTensor > result = tensor;
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

        virtual string decodeBest(const shared_ptr <BaseTensor> &tensor) {
            stringstream ss;
            tensor->print(ss);
            return ss.str();
        }

        virtual vector<string> decodeTop(const shared_ptr <BaseTensor> &tensor, const size_t numberOfResults) {
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
            if(categoryIndex >= categoryLabels_.size()) {
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