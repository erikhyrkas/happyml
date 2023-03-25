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
        shared_ptr<BaseTensor> decode(shared_ptr<BaseTensor> tensor) {
            return tensor;
        }
    };

    // TODO: add a "minimum confidence" parameter, where it doesn't return values that are
    //  below a threshold.
    // TODO: could return the confidence with the text
    class BestTextCategoryDecoder {
    public:
        BestTextCategoryDecoder(const vector<string> &categoryLabels) {
            this->categoryLabels = categoryLabels;
        }

        string decode(shared_ptr<BaseTensor> tensor) {
            const auto categoryIndex = maxIndex(tensor);
            return categoryLabels[categoryIndex];
        }

    private:
        vector<string> categoryLabels;
    };

    // If you want the top 5 or top 3 results, this would accomplish that by returning those categories
    // in the order from best to worst.
    // TODO: add a "minimum confidence" parameter, where it doesn't return values that are
    //  below a threshold.
    // TODO: could return the confidence with the text
    // TODO: test.
    class TopTextCategoryDecoder {
    public:
        explicit TopTextCategoryDecoder(const vector<string> &categoryLabels, const size_t numberOfResults)
                : categoryLabels(categoryLabels), numberOfResults(numberOfResults) {
        }

        vector<string> decode(shared_ptr<BaseTensor> tensor) {
            const auto categoryIndex = tensor->topIndices(numberOfResults, 0, 0);
            vector<string> result;
            result.reserve(categoryIndex.size());
            for (const auto &index: categoryIndex) {
                result.push_back(categoryLabels[index.getIndex()]);
            }
            return result;
        }

    private:
        const vector<string> categoryLabels;
        const size_t numberOfResults;
    };
}

#endif //HAPPYML_DATA_DECODER_HPP