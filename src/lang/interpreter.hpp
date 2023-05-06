//
// Created by Erik Hyrkas on 12/29/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_INTERPRETER_HPP
#define HAPPYML_INTERPRETER_HPP

#include <fstream>
#include <sstream>
#include "happyml_script_init.hpp"
#include "execution_context.hpp"

using namespace std;

namespace happyml {
    class InterpreterSession {
    public:
        explicit InterpreterSession(const shared_ptr<Parser> &parser) {
            this->parser = parser;
            executionContext = make_shared<ExecutionContext>();
        }

        bool interpretCommands(const string &text, const string &source = "unknown") {
//            cout << "interpreting [" << text << "]..." << endl;
            // todo: can cache the compiled executable scripts, since they are stateless.
            auto parseResult = parser->parse(text, source);
            if (!parseResult->isSuccessful()) {
                cerr << parseResult->getMessage() << endl;
                return false;
            }
            // in terms of caching the output execution, that would rely
            // on the internal workings of what is being executed to decide
            // if there is an opportunity to optimize. For example, if a
            // model wants to cache a prediction, it is welcome to do so,
            // but we won't do it here because we don't know if the session
            // state would impact how it made its prediction.
            auto executable = parseResult->getExecutable();
            auto result = executable->execute(executionContext);
            // TODO: handle errors
            if (!result->isSuccessful()) {
                // print error
                cerr << result->getMessage() << endl;
            }
            return result->exitRequested();
        }

        bool interpretFile(const string &filePath) {
            bool done;
            try {
                ifstream stream;
                stream.open(filePath, ifstream::in);
                stringstream fullText;
                fullText << stream.rdbuf();
                try {
                    done = interpretCommands(fullText.str(), filePath);
                } catch (const exception &e) {
                    cerr << "Failed to interpret: " << filePath << endl << e.what() << endl;
                    throw e;
                }
                stream.close();
            } catch (ofstream::failure &e) {
                cerr << "Failed to load: " << filePath << endl << e.what() << endl;
                throw e;
            }
            return done;
        }

        void interactiveInterpret() {
            cout << "happyml v0.0.1 interpreter." << endl;
            cout << "For a list of commands use the command: help" << endl;
            cout << "READY" << endl;
            // interpret commandline until done
            std::string nextLine, fullLine;
            cout << "> ";
            while (std::getline(std::cin, nextLine)) {
                nextLine = trimEnd(nextLine);
                if (nextLine.back() == '\\') {
                    // Remove the backslash and add the line to the buffer
                    nextLine.pop_back();
                    fullLine += nextLine + '\n';
                } else {
                    // Add the line to the buffer and interpret the command
                    fullLine += nextLine;
                    const bool done = interpretCommands(fullLine, "cli");
                    if (done) {
                        break;
                    }
                    // Reset the buffer for the next command
                    fullLine.clear();
                    cout << "> ";
                }
            }
        }

    private:
        shared_ptr<ExecutionContext> executionContext;
        shared_ptr<Parser> parser;
    };
}


#endif //HAPPYML_INTERPRETER_HPP
