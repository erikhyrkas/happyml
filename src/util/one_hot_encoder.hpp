//
// Created by Erik Hyrkas on 4/3/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_ONE_HOT_ENCODER_HPP
#define HAPPYML_ONE_HOT_ENCODER_HPP

#include <vector>
#include <string>
#include <unordered_map>


using namespace std;

namespace happyml {

    void pad_or_truncate_tokens(vector<vector<float>> &tokens, size_t target_length, size_t token_size) {
        // Truncate tokens if it's longer than target_length
        if (tokens.size() > target_length) {
            tokens.resize(target_length);
        } else if (tokens.size() < target_length) {
            // Pad tokens if it's shorter than target_length
            vector<float> padding_token(token_size, 0.0);
            // Add padding tokens until tokens reaches target_length
            size_t num_padding_tokens = target_length - tokens.size();
            for (size_t i = 0; i < num_padding_tokens; ++i) {
                tokens.push_back(padding_token);
            }
        }
    }

    vector<vector<float>> one_hot_encode_bpe_tokens(vector<u16string> tokens, size_t largest_bpe_code) {
        // For the purposes of one-hot encoding, each character of each incoming token needs
        // to be encoded into its own vector because each of those characters represents a common
        // sub-word.
        // By default, return a vector of all zeroes for unknown tokens.
        vector<vector<float>> encoded_tokens;
        for (const auto &token: tokens) {
            for (const auto &c: token) {
                vector<float> encoded_chars(largest_bpe_code, 0.0);
                if (c <= largest_bpe_code) {
                    encoded_chars[c] = 1.0;
                }
                encoded_tokens.push_back(encoded_chars);
            }
        }
        return encoded_tokens;
    }

    vector<vector<float>> one_hot_encode_characters(vector<string> tokens) {
        // If you have a model that wants to predict the next character of a string
        // then you want to one-hot encode each character. Not as useful for building
        // a model for predicting words or sub-words.
        vector<vector<float>> encoded_tokens;
        for (const auto &token: tokens) {
            for (const auto &c: token) {
                vector<float> encoded_chars(255, 0.0);
                encoded_chars[c] = 1.0;
                encoded_tokens.push_back(encoded_chars);
            }
        }
        return encoded_tokens;
    }

    vector<vector<float>> one_hot_encode_words(vector<string> tokens, unordered_map<string, int> token_to_index) {
        // If you have a model that wants to predict the next whole word, this is the encoder for you.
        // It will produce one one-hot encoded vector per input token.
        // It requires a vocabulary map of tokens to an index, where the id is a unique representation of the
        // token. The size of the vocabulary map should be 1 greater than the number of tokens in it, since
        // the first token should have an index of 0 and the last token added to the token_to_index map
        // should have an index of (size - 1).
        // By default, return a vector of all zeroes for unknown tokens.
        vector<vector<float>> encoded_tokens;
        int vocab_size = token_to_index.size();
        for (const auto &token: tokens) {
            vector<float> encoded_words(vocab_size, 0.0);
            if (token_to_index.count(token)) {
                encoded_words[token_to_index[token]] = 1.0;
            }
            encoded_tokens.push_back(encoded_words);
        }
        return encoded_tokens;
    }
}
#endif //HAPPYML_ONE_HOT_ENCODER_HPP
