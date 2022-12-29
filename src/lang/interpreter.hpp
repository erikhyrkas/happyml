//
// Created by Erik Hyrkas on 12/29/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_INTERPRETER_HPP
#define HAPPYML_INTERPRETER_HPP

#include <fstream>
#include <sstream>
#include "parser.hpp"

// syntax might be something like:
// set output <human/machine>

// create <data set type> dataset <name> from <location> [with <format>]
// add rows to dataset <name> using delimited data:
// 1, 2, 3, 4
// <empty line to denote end of data>

// create [<adjective>*] <model type> model <model name> [<model version>] using <data set name>
// tune <model name> [<model version>] [as [<model name>] [<model version>]] using <data set name>
// retrain <model name> [<model version>] [as [<model name>] [<model version>]] using <data set name>

// predict using <model name> [<model version] given <input>
//               or
// infer using <model name> [<model version] given <input>

class InterpreterSession {
public:
    bool interpretCommands(const string &text) {
        bool done = false;
        cout << "interpreting [" << text << "]..." << endl;
        // temporary exit check
        if (text == "!exit") {
            done = true;
        } else {

        }
        return done;
    }

    bool interpretFile(const string &filePath) {
        bool done;
        try {
            ifstream stream;
            stream.open(filePath, ifstream::in);
            stringstream fullText;
            fullText << stream.rdbuf();
            done = interpretCommands(fullText.str());
            stream.close();
        } catch (ofstream::failure &e) {
            cerr << "Failed to load: " << filePath << endl << e.what() << endl;
            throw e;
        }
        return done;
    }

    void interactiveInterpret() {
        // interpret commandline until done
        std::string nextLine;
        while (std::getline(std::cin, nextLine)) {
            bool done = interpretCommands(nextLine);
            if (done) {
                break;
            }
        }
    }
};

#endif //HAPPYML_INTERPRETER_HPP
