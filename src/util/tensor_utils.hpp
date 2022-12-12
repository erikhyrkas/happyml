//
// Created by Erik Hyrkas on 12/9/2022.
//

#ifndef MICROML_TENSOR_UTILS_HPP
#define MICROML_TENSOR_UTILS_HPP

#include "../types/half_float.hpp"
#include "../types/quarter_float.hpp"
#include "../types/tensor.hpp"
#include "../types/tensor_views.hpp"
#include "../types/materialized_tensors.hpp"
#include <iomanip>
#include <vector>
#include <utility>
#include <iterator>
#include <future>
#include <execution>

namespace microml {

    std::shared_ptr<PixelTensor> pixelTensor(const std::vector<std::vector<std::vector<float>>> &t) {
        return make_shared<PixelTensor>(t);
    }

    std::shared_ptr<FullTensor> columnVector(const std::vector<float> &t) {
        return make_shared<FullTensor>(t);
    }

    std::shared_ptr<BaseTensor> randomTensor(size_t rows, size_t cols, size_t channels) {
        return make_shared<TensorFromRandom>(rows, cols, channels, 0.f, 1.f);
    }

    float scalar(const std::shared_ptr<BaseTensor> &tensor) {
        if (tensor->size() < 1) {
            return 0.f;
        }
        return tensor->getValue(0);
    }

    std::shared_ptr<BaseTensor> round(const std::shared_ptr<BaseTensor> &tensor) {
        return std::make_shared<TensorRoundedView>(tensor);
    }

    size_t maxIndex(const std::shared_ptr<BaseTensor> &tensor) {
        return tensor->maxIndex(0, 0);
    }

    int estimateBias(int estimate_min, int estimate_max, const float adj_min, const float adj_max) {
        int quarter_bias = estimate_min;
        for (int proposed_quarter_bias = estimate_max; proposed_quarter_bias >= estimate_min; proposed_quarter_bias--) {
            const float bias_max = quarterToFloat(QUARTER_MAX, proposed_quarter_bias);
            const float bias_min = -bias_max;
            if (adj_min > bias_min && adj_max < bias_max) {
                quarter_bias = proposed_quarter_bias;
                break;
            }
        }
        return quarter_bias;
    }

    std::shared_ptr<BaseTensor> materializeTensor(const std::shared_ptr<BaseTensor> &tensor, uint8_t bits) {
        if (bits == 32) {
            if (tensor->isMaterialized()) {
                // there is no advantage to materializing an already materialized tensor to 32 bits.
                // whether other bit options may reduce memory footprint.
                return tensor;
            }
            return std::make_shared<FullTensor>(tensor);
        } else if (bits == 16) {
            return std::make_shared<HalfTensor>(tensor);
        }
        auto minMax = tensor->range();
        int quarterBias = estimateBias(4, 15, minMax.first, minMax.second);
        return std::make_shared<QuarterTensor>(tensor, quarterBias);
    }

// channels, rows, columns
    shared_ptr<BaseTensor> materializeTensor(const shared_ptr<BaseTensor> &other) {
        if (other->isMaterialized()) {
            return other;
        }
        return make_shared<FullTensor>(other);
    }

    shared_ptr<FullTensor> tensor(const vector<vector<vector<float>>> &t) {
        return make_shared<FullTensor>(t);
    }
}

#endif //MICROML_TENSOR_UTILS_HPP
