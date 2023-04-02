//
// Created by Erik Hyrkas on 4/1/2023.
//

#include <iostream>
#include "../util/unit_test.hpp"
#include "../util/timers.hpp"
#include "../util/text_embedder.hpp"
#include "../ml/rotary_positional_embedding.hpp"

using namespace std;
using namespace happyml;


void quickTest() {
    unsigned int seq_length = 10;
    unsigned int dim = 64;
    RotaryPositionalEmbedding rotary_positional_embedding(seq_length, dim);
    auto positional_encoding = rotary_positional_embedding.get_positional_encoding();
    positional_encoding->print();
}

void test_text_embedding() {
    unsigned int seq_length = 10;
    unsigned int dim = 4096;
    auto bpe = make_shared<BytePairEncoderModel>();
    RotaryPositionalEmbedding rotary_positional_embedding(seq_length, dim);
    auto embeddingMatrix = rotary_positional_embedding.get_positional_encoding();

    auto tensor = text_to_tensor("some random text", bpe, 100,
                                 embeddingMatrix,
                                 rotary_positional_embedding.get_padding_token_embedding(),
                                 rotary_positional_embedding.get_unknown_token_embedding());
    tensor->print();

}

int main() {
    try {
        EvenMoreSimpleTimer timer;
        test_text_embedding();
        timer.printMilliseconds();

//        quickTest();
//        timer.printMilliseconds();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
    return 0;
}