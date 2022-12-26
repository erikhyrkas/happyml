//
// Created by Erik Hyrkas on 12/2/2022.
//

#ifndef HAPPYML_BASIC_PROFILER_HPP
#define HAPPYML_BASIC_PROFILER_HPP
#include <chrono>
#include <iostream>
#include <map>
#include <sstream>

using namespace std;

namespace happyml {
    class ProfileBlock {
    public:
        ProfileBlock(const char *file, const string &func, const int line) {
            stringstream ss;
            ss << file  << ":" << func << ":" << line;
            this->label = ss.str();
            blockEntered();
        }
        ~ProfileBlock() {
            blockLeft();
        }

        void blockEntered() {
            startTime = chrono::high_resolution_clock::now();
            cout << "||start " << label << "||" << endl;
        }

        void blockLeft() {
            thread_local map<string,chrono::microseconds> duration{};
            auto stopTime = chrono::high_resolution_clock::now();
            auto elapsed = chrono::duration_cast<chrono::microseconds>(stopTime - startTime);
            duration[label] += elapsed;
            cout << "||end " << label << " " << elapsed.count() << " ms (" << duration[label].count() << " ms)||" << endl;
        }

    private:
        chrono::time_point<chrono::high_resolution_clock> startTime{};
        string label;
    };



//#define MICRO_ML_PROFILE
#ifdef MICRO_ML_PROFILE
#define PROFILE_BLOCK(x) ProfileBlock x(__FILE__, __func__, __LINE__)
#else
#define PROFILE_BLOCK(x) ((void)0)
#endif
}
#endif //HAPPYML_BASIC_PROFILER_HPP
