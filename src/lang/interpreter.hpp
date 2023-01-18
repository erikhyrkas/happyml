//
// Created by Erik Hyrkas on 12/29/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_INTERPRETER_HPP
#define HAPPYML_INTERPRETER_HPP

#include <fstream>
#include <sstream>
#include "happml_script_init.hpp"

using namespace std;

namespace happyml {
    class InterpreterSession {
    public:
        InterpreterSession(const shared_ptr<Parser> &parser) {
            this->parser = parser;
            sessionState = make_shared<SessionState>();
        }

        bool interpretCommands(const string &text) {
            cout << "interpreting [" << text << "]..." << endl;
            // todo: can cache the compiled executable scripts, since they are stateless.
            auto executable = parser->parse(text);
            // in terms of caching the output execution, that would rely
            // on the internal workings of what is being executed to decide
            // if there is an opportunity to optimize. For example, if a
            // model wants to cache a prediction, it is welcome to do so,
            // but we won't do it here because we don't know if the session
            // state would impact how it made its prediction.
            return executable->execute(sessionState);
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
                const bool done = interpretCommands(nextLine);
                if (done) {
                    break;
                }
            }
        }

    private:
        shared_ptr<SessionState> sessionState;
        shared_ptr<Parser> parser;
    };
}


#endif //HAPPYML_INTERPRETER_HPP
