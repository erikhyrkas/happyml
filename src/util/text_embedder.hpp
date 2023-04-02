//
// Created by Erik Hyrkas on 4/1/2023.
//

#ifndef HAPPYML_TEXT_ENCODER_HPP
#define HAPPYML_TEXT_ENCODER_HPP

#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include "tensor_utils.hpp"
#include "data_util.hpp"
#include "../ml/byte_pair_encoder.hpp"

using namespace std;

namespace happyml {

    shared_ptr<BaseTensor> text_to_tensor(const string &text,
                                         shared_ptr<BytePairEncoderModel> &encoder,
                                         int number_of_supported_tokens,
                                         const shared_ptr<BaseTensor> &embedding_matrix,
                                         const vector<float> &padding_token,
                                         const vector<float> &unknown_token_embedding) {

        if(padding_token.size() != embedding_matrix->columnCount()) {
            throw runtime_error("padding token length must match embedding matrix");
        }
        if(unknown_token_embedding.size() != embedding_matrix->columnCount()) {
            throw runtime_error("unknown token length must match embedding matrix");
        }

        // Tokenize the input text
        vector<string> tokens = string_to_tokens(text);

        // Encode tokens using BytePairEncoderModel
        vector<u16string> encoded_tokens;
        for (const string &token: tokens) {
            encoded_tokens.push_back(encoder->encode(token));
        }

        auto embeddingRows = embedding_matrix->rowCount();

        // Convert encoded tokens to embeddings using the embedding matrix
        vector<vector<float>> result;
        for (const u16string &encoded_token: encoded_tokens) {
            // Use the first character of the encoded token as the index for the embedding matrix
            auto rowNum = static_cast<size_t>(encoded_token[0]);
            if (rowNum < embeddingRows) {
                result.push_back(embedding_matrix->getRowValues(rowNum));
            } else {
                result.push_back(unknown_token_embedding);
            }
        }

        // Pad with padding_token if necessary
        while (result.size() < number_of_supported_tokens) {
            result.push_back(padding_token);
        }

        // Truncate the tensor if it's longer than the specified length
        if (result.size() > number_of_supported_tokens) {
            result.resize(number_of_supported_tokens);
        }

        return tensor({result});
    }


}

#endif //HAPPYML_TEXT_ENCODER_HPP
