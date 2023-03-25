//
// Created by Erik Hyrkas on 3/23/2023.
//

#include <iostream>
#include "../util/unit_test.hpp"
#include "../util/timers.hpp"
#include "../lang/happyml_script_init.hpp"

using namespace std;
using namespace happyml;

void testParser1() {
    auto parser = initializeHappymlParser();
    string val = "create dataset test from file://blah with format x with expected image at 0 with given float at 1 through 21";
    auto result = parser->parse(val);
    ASSERT_TRUE(result->isSuccessful());
}

void testParser2() {
    auto parser = initializeHappymlParser();
    string val = "create dataset";
    auto result = parser->parse(val);
    ASSERT_FALSE(result->isSuccessful());
}

void testParser3() {
    auto parser = initializeHappymlParser();
    string val = "create model";
    auto result = parser->parse(val);
    ASSERT_FALSE(result->isSuccessful());
}

void testParser4() {
    auto parser = initializeHappymlParser();
    string val = "create dataset x";
    ASSERT_TRUE(parser->parse(val)); // TODO: this should fail, more error handling needed.
}

void testParser5() {
    auto parser = initializeHappymlParser();
    string val = "create dataset x from file://path/parts";
    auto result = parser->parse(val);
    ASSERT_TRUE(result->isSuccessful());
}

int main() {
    try {
        EvenMoreSimpleTimer timer;
        testParser1();
        timer.printMilliseconds();
        testParser2();
        timer.printMilliseconds();
        testParser3();
        timer.printMilliseconds();
        testParser4();
        timer.printMilliseconds();
        testParser5();
        timer.printMilliseconds();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}