//
// Created by Erik Hyrkas on 3/23/2023.
//

#include <iostream>
#include "../lang/simple_interpreter.hpp"
#include "../util/unit_test.hpp"
#include "../util/timers.hpp"

using namespace std;
using namespace happyml;


void testInterpreter() {
    string val = "create dataset test from blah with format x with expected image at 0 with given float at 1 through 21";
    ASSERT_TRUE(simple_interpret(val));
}

void testInterpreter2() {
    string val = "create dataset";
    ASSERT_TRUE(simple_interpret(val)); // TODO: this should fail, more error handling needed.
}

void testInterpreter3() {
    string val = "create model";
    ASSERT_FALSE(simple_interpret(val));
}

void testInterpreter4() {
    string val = "create dataset x";
    ASSERT_TRUE(simple_interpret(val)); // TODO: this should fail, more error handling needed.
}

void testInterpreter5() {
    string val = "create dataset x from path/parts";
    ASSERT_TRUE(simple_interpret(val));
}
int main() {
    try {
        EvenMoreSimpleTimer timer;
        testInterpreter();
        timer.printMilliseconds();
        testInterpreter2();
        timer.printMilliseconds();
        testInterpreter3();
        timer.printMilliseconds();
        testInterpreter4();
        timer.printMilliseconds();
        testInterpreter5();
        timer.printMilliseconds();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}