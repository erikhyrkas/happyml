//
// Created by Erik Hyrkas on 4/3/2023.
// Copyright 2023. Usable under MIT license.
//
#include <iostream>
#include "../util/one_hot_encoder.hpp"
#include "../util/unit_test.hpp"
#include "../util/timers.hpp"
#include "../util/text_embedder.hpp"

using namespace std;
using namespace happyml;

void test_one_hot1() {
    vector<vector<float>> tokens = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    pad_or_truncate_tokens(tokens, 2, 2);
    vector<vector<float>> expected_tokens = {{1.0, 2.0}, {3.0, 4.0}};
    ASSERT_TRUE(are_vector_of_vectors_equal(tokens, expected_tokens));
}

void test_one_hot2() {
    vector<vector<float>> tokens = {{1.0, 2.0}};
    pad_or_truncate_tokens(tokens, 2, 2);
    vector<vector<float>> expected_tokens = {{1.0, 2.0}, {0.0, 0.0}};
    ASSERT_TRUE(are_vector_of_vectors_equal(tokens, expected_tokens));
}

void test_one_hot3() {
    vector<u16string> tokens = {u"hell", u"o"};
    size_t largest_bpe_code = 256;
    vector<vector<float>> encoded_tokens = one_hot_encode_bpe_tokens(tokens, largest_bpe_code);
    vector<vector<float>> expected_encoded_tokens(5, vector<float>(256, 0));
    expected_encoded_tokens[0]['h'] = 1.0;
    expected_encoded_tokens[1]['e'] = 1.0;
    expected_encoded_tokens[2]['l'] = 1.0;
    expected_encoded_tokens[3]['l'] = 1.0;
    expected_encoded_tokens[4]['o'] = 1.0;

    ASSERT_TRUE(are_vector_of_vectors_equal(encoded_tokens, expected_encoded_tokens));
}

void test_one_hot4() {
    vector<string> tokens = {"hello", "world"};
    vector<vector<float>> encoded_tokens = one_hot_encode_characters(tokens);
    vector<vector<float>> expected_encoded_tokens(10, vector<float>(255, 0));
    expected_encoded_tokens[0]['h'] = 1.0;
    expected_encoded_tokens[1]['e'] = 1.0;
    expected_encoded_tokens[2]['l'] = 1.0;
    expected_encoded_tokens[3]['l'] = 1.0;
    expected_encoded_tokens[4]['o'] = 1.0;
    expected_encoded_tokens[5]['w'] = 1.0;
    expected_encoded_tokens[6]['o'] = 1.0;
    expected_encoded_tokens[7]['r'] = 1.0;
    expected_encoded_tokens[8]['l'] = 1.0;
    expected_encoded_tokens[9]['d'] = 1.0;
    ASSERT_TRUE(are_vector_of_vectors_equal(encoded_tokens, expected_encoded_tokens));
}

void test_one_hot5() {
    vector<string> tokens = {"hello", "world", "world"};
    unordered_map<string, int> token_to_index = {
            {"<unk>", 0},
            {"hello", 1},
            {"world", 2},
    };
    vector<vector<float>> encoded_tokens = one_hot_encode_words(tokens, token_to_index);
    vector<vector<float>> expected_encoded_tokens = {
            {0.0, 1.0, 0.0},  // encoding for "hello"
            {0.0, 0.0, 1.0},  // encoding for "world"
            {0.0, 0.0, 1.0},  // encoding for "world"
    };
    ASSERT_TRUE(are_vector_of_vectors_equal(encoded_tokens, expected_encoded_tokens));
}
int main() {
    try {
        EvenMoreSimpleTimer timer;
        test_one_hot1();
        test_one_hot2();
        test_one_hot3();
        test_one_hot4();
        test_one_hot5();
        timer.printMilliseconds();


    } catch (const exception &e) {
        cout << e.what() << endl;
    }
    return 0;
}