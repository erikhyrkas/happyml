//
// Created by Erik Hyrkas on 4/1/2023.
//

#ifndef HAPPYML_TEXT_ENCODER_HPP
#define HAPPYML_TEXT_ENCODER_HPP

#include "one_hot_encoder.hpp"
#include "../ml/byte_pair_encoder.hpp"
#include "../ml/rotary_positional_embedding.hpp"

using namespace std;

namespace happyml {

    vector<vector<float>> text_to_tensor_bpe_rotary(const string &text,
                                                    const shared_ptr<BytePairEncoderModel> &bpe_encoder,
                                                    const shared_ptr<RotaryPositionalEmbedder> &embedder) {

        // Tokenize the input text into words and symbols
        vector<string> tokens = string_to_tokens(text);

        // Encode tokens using BytePairEncoderModel
        vector<u16string> bpe_encoded_tokens = bpe_encoder->encode(tokens);

        // One-hot encode the bpe encoded tokens.
        vector<vector<float>> one_hot_encoded_tokens = one_hot_encode_bpe_tokens(bpe_encoded_tokens,
                                                                                 bpe_encoder->getLargestCode());

        // Embed the one-hot encoded vectors
        vector<vector<float>> embedded_tokens = embedder->embed_tokens(one_hot_encoded_tokens);

        return std::move(embedded_tokens);
    }


}

#endif //HAPPYML_TEXT_ENCODER_HPP
