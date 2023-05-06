//
// Created by Erik Hyrkas on 4/1/2023.
// Copyright 2023. Usable under MIT license.
//

#include <iostream>
#include "../util/timers.hpp"
#include "../util/text_embedder.hpp"

using namespace std;
using namespace happyml;


void test_text_embedding() {
    size_t model_max_tokens = 10;
    auto bpe = make_shared<BytePairEncoderModel>();
    auto rpe = make_shared<RotaryPositionalEmbedder>(model_max_tokens, bpe->getLargestCode());
    auto tensor = text_to_tensor_bpe_rotary("some random text", bpe, rpe);

    cout << fixed << setprecision(2);
    for (const auto &token: tensor) {
        for (const auto &val: token) {
            cout << val << " ";
        }
        cout << endl << endl;
    }
}

int main() {
    try {
        EvenMoreSimpleTimer timer;
        test_text_embedding();
        timer.printMilliseconds();

    } catch (const exception &e) {
        cout << e.what() << endl;
    }
    return 0;
}