//
// Created by Erik Hyrkas on 4/1/2023.
//

#ifndef HAPPYML_ROTARY_POSITIONAL_EMBEDDING_HPP
#define HAPPYML_ROTARY_POSITIONAL_EMBEDDING_HPP

#include <cmath>
#include <vector>
#include <iostream>
#include <cassert>
#include "../types/tensor.hpp"
#include "../util/tensor_utils.hpp"

using namespace std;

namespace happyml {

    /*
     * I first read about Rotary Positional Embeddings here:
     * https://blog.eleuther.ai/rotary-embeddings/
     *
     * RotaryPositionalEmbedding class computes rotary positional encodings
     * for the given sequence length and dimensions.
     */

    class RotaryPositionalEmbedding {
    public:
        RotaryPositionalEmbedding(unsigned int seq_length, unsigned int dim)
                : seq_length(seq_length), dim(dim) {
            assert(dim % 2 == 0);
            init_rotary_constants();
        }

        void init_rotary_constants() {
            rotary_constants.reserve(dim);
            float log_base = 1 / log(10000.0);
            for (unsigned int i = 0; i < dim; i += 2) {
                float div_term = exp(i * log_base);
                rotary_constants.push_back(sin(1.0 / div_term));
                rotary_constants.push_back(cos(1.0 / div_term));
            }
        }

        shared_ptr<BaseTensor> get_positional_encoding() const {
            vector<vector<float>> pos_enc(seq_length, vector<float>(dim));
            for (unsigned int pos = 0; pos < seq_length; ++pos) {
                for (unsigned int i = 0; i < dim; i += 2) {
                    pos_enc[pos][i] = sin(static_cast<float>(pos) * rotary_constants[i]);
                    pos_enc[pos][i + 1] = cos(static_cast<float>(pos) * rotary_constants[i + 1]);
                }
            }

            return tensor({pos_enc});
        }

        vector<float> get_unknown_token_embedding() const {
            return vector<float>(dim, 0.0f);
        }

        vector<float> get_padding_token_embedding() const {
            return vector<float>(dim, 0.0f);
        }

    private:
        unsigned int seq_length;
        unsigned int dim;
        vector<float> rotary_constants;
    };
}
#endif //HAPPYML_ROTARY_POSITIONAL_EMBEDDING_HPP
