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


/*
 * I first read about Rotary Positional Embeddings here:
 * https://blog.eleuther.ai/rotary-embeddings/
 *
 * RotaryPositionalEmbedding class computes rotary positional encodings
 * for the given sequence length and dimensionality.
 */

namespace happyml {
    /*
     * The seq_length and dim parameters directly impact the capabilities, quality, and resource intensiveness of your model.
     *
     * sequence_length is the maximum number of tokens our model supports at time. Longer sequence lengths means considerably more memory use.
     * dimensionality is the size of the embedding, which is used to capture relationships between the positions in the token sequence.
     *
     * Based on this article: https://dugas.ch/artificial_curiosity/GPT_architecture.html
     * It looks to me like gpt-3 has a sequence_length of 2048 and a dimensionality of 12288
     */
    class Embedder {
    public:
        explicit Embedder(size_t sequence_length, size_t dimensionality = 512)
                : sequence_length_(sequence_length), dimensionality_(dimensionality) {
        }

        [[nodiscard]] size_t getSequenceLength() const {
            return sequence_length_;
        }

        [[nodiscard]] size_t getDimensionality() const {
            return dimensionality_;
        }

        [[nodiscard]] virtual vector<float> get_unknown_token_embedding() const {
            return vector<float> (dimensionality_, 0.0f);
        }

        [[nodiscard]] virtual vector<float> get_padding_token_embedding() const {
            return vector<float> (dimensionality_, 0.0f);
        }

        vector<vector<float>> embed_tokens(const vector<vector<float>> &one_hot_encoded_tokens) {
            vector<vector<float>> embedded_tokens;
            embedded_tokens.reserve(one_hot_encoded_tokens.size());
            size_t position = 0;
            for (const auto &one_hot_encoded_token : one_hot_encoded_tokens) {
                embedded_tokens.push_back(embed_token(one_hot_encoded_token, position));
                ++position;
                if( position >= sequence_length_) {
                    break;
                }
            }
            return embedded_tokens;
        }

        virtual vector<float> embed_token(const vector<float> &one_hot_encoded_token, size_t position) = 0;

    protected:
        size_t sequence_length_;
        size_t dimensionality_;
    };


    class RotaryPositionalEmbedder : public Embedder {
    public:

        explicit RotaryPositionalEmbedder(size_t sequence_length, size_t dimensionality = 512)
                : Embedder(sequence_length, dimensionality) {
            size_t even_dimensionality = dimensionality_ + (dimensionality_ % 2);

            // create rotary_constants
            vector<float> rotary_constants;
            rotary_constants.reserve(even_dimensionality);
            double const log_base = 1 / log(10000.0);
            for (unsigned int i = 0; i < even_dimensionality; i += 2) {
                double const div_term = exp((double) i * log_base);
                rotary_constants.push_back((float) sin(1.0 / div_term));
                rotary_constants.push_back((float) cos(1.0 / div_term));
            }
            // create positional encoding tensor
            vector<vector<float>> pos_enc(sequence_length_, vector<float> (even_dimensionality));
            for (unsigned int pos = 0; pos < sequence_length_; ++pos) {
                for (unsigned int i = 0; i < even_dimensionality; i += 2) {
                    pos_enc[pos][i] = sin(static_cast<float>(pos) * rotary_constants[i]);
                    pos_enc[pos][i + 1] = cos(static_cast<float>(pos) * rotary_constants[i + 1]);
                }
            }
            positional_encoding = std::move(pos_enc);
        }


        [[nodiscard]] vector<vector<float>> get_positional_encoding() const {
            return positional_encoding;
        }

        vector<float> embed_token(const vector<float> &one_hot_encoded_token, const size_t position) override {
            if(position >= sequence_length_) {
                throw exception("Rotary Positional Encoding cannot embed a position beyond it's configured sequence length.");
            }
            if( one_hot_encoded_token.size() > dimensionality_) {
                throw exception("The embedding dimension must match the one-hot encoding length.");
            }

            vector<float> embedded_token(dimensionality_, 0.0f);
            auto one_hot_encoded_token_size = one_hot_encoded_token.size();
            for (size_t i = 0; i < one_hot_encoded_token_size; ++i) {
                embedded_token[i] = one_hot_encoded_token[i] + positional_encoding[position][i];
            }
            return embedded_token;
        }

    private:
        vector<vector<float>> positional_encoding;
    };


//            positional_encoding = tensor({pos_enc});

    //        [[nodiscard]] shared_ptr<BaseTensor> get_positional_encoding() const {
    //            return positional_encoding;
    //        }
    //shared_ptr<BaseTensor> positional_encoding;

}
#endif //HAPPYML_ROTARY_POSITIONAL_EMBEDDING_HPP
