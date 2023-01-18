//
// Created by Erik Hyrkas on 10/23/2022.
// Copyright 2022. Usable under MIT license.
//
#include <iostream>
#include "lang/interpreter.hpp"


using namespace happyml;

int main(int argc, char *argv[]) {
    try {
        auto parser = initializeHappymlParser();
        auto interpreterSession = make_shared<InterpreterSession>(parser);
        if (argc < 2) {
            // enter interactive session
            interpreterSession->interactiveInterpret();
        } else {
            for (int argIndex = 1; argIndex < argc; argIndex++) {
                cout << argIndex << " " << argv[argIndex] << endl;
                bool done = interpreterSession->interpretFile(argv[argIndex]);
                if (done) {
                    break;
                }
            }
        }
    } catch (const exception &e) {
        cout << e.what() << endl;
        return 1;
    }
    return 0;
}
