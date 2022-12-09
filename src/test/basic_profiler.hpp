//
// Created by Erik Hyrkas on 12/2/2022.
//

#ifndef MICROML_BASIC_PROFILER_HPP
#define MICROML_BASIC_PROFILER_HPP
#include <chrono>
#include <iostream>
#include <map>
#include <sstream>

using namespace std;

namespace microml {
    class ProfileBlock {
    public:
        ProfileBlock(const char *file, const string &func, const int line) {
            stringstream ss;
            ss << file  << ":" << func << ":" << line;
            this->label = ss.str();
            block_entered();
        }
        ~ProfileBlock() {
            block_left();
        }

        void block_entered() {
            start_time = chrono::high_resolution_clock::now();
            cout << "||start " << label << "||" << endl;
        }

        void block_left() {
            thread_local map<string,chrono::microseconds> duration{};
            auto stop_time = chrono::high_resolution_clock::now();
            auto elapsed = chrono::duration_cast<chrono::microseconds>(stop_time - start_time);
            duration[label] += elapsed;
            cout << "||end " << label << " " << elapsed.count() << " ms (" << duration[label].count() << " ms)||" << endl;
        }

    private:
        chrono::time_point<chrono::high_resolution_clock> start_time{};
        string label;
    };



//#define MICRO_ML_PROFILE
#ifdef MICRO_ML_PROFILE
#define PROFILE_BLOCK(x) ProfileBlock x(__FILE__, __func__, __LINE__)
#else
#define PROFILE_BLOCK(x) ((void)0)
#endif
}
#endif //MICROML_BASIC_PROFILER_HPP
