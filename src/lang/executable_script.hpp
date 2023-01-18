//
// Created by Erik Hyrkas on 12/30/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_EXECUTABLE_SCRIPT_HPP
#define HAPPYML_EXECUTABLE_SCRIPT_HPP

#include <memory>
#include "session_state.hpp"

using namespace std;

namespace happyml {
    class ExecutableScript {
    public:
        virtual bool execute(const shared_ptr<SessionState> &sessionState) = 0;
    };

}
#endif //HAPPYML_EXECUTABLE_SCRIPT_HPP
