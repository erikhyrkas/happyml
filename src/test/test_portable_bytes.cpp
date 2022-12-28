//
// Created by Erik Hyrkas on 12/19/2022.
// Copyright 2022. Usable under MIT license.
//

#include <iostream>
#include <string>
#include "../types/tensor.hpp"
#include "../util/unit_test.hpp"
#include "../util/timers.hpp"

using namespace happyml;
using namespace std;

void testPortableBytes16() {
    uint16_t test = 276;
//    cout << portableBytes(portableBytes(test)) << " vs " << portableBytes(test) << endl;
    ASSERT_TRUE(5121 == portableBytes(test));
    ASSERT_TRUE(test == portableBytes(portableBytes(test)));
}

void testPortableBytes32() {
    uint32_t test = 3346354676;
//    cout << portableBytes(portableBytes(test)) << " vs " << portableBytes(test) << endl;
    ASSERT_TRUE(4098979271 == portableBytes(test));
    ASSERT_TRUE(test == portableBytes(portableBytes(test)));
}

void testPortableBytes64() {
    uint64_t test = 3346354676524356676;
    cout << portableBytes(portableBytes(test)) << " vs " << portableBytes(test) << endl;
    ASSERT_TRUE(4941771756473118766 == portableBytes(test));
    ASSERT_TRUE(test == portableBytes(portableBytes(test)));
}

int main() {
    try {
        EvenMoreSimpleTimer timer;
        testPortableBytes16();
        timer.printMilliseconds();
        testPortableBytes32();
        timer.printMilliseconds();
        testPortableBytes64();
        timer.printMilliseconds();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }

    return 0;
}