//
// Created by Erik Hyrkas on 12/2/2022.
// Copyright 2022. Usable under MIT license.
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
            ss << file << ":" << func << ":" << line;
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
            thread_local map<string, chrono::microseconds> duration{};
            thread_local map<string, double> count{};
            count[label]++;
            auto stopTime = chrono::high_resolution_clock::now();
            auto elapsed = chrono::duration_cast<chrono::microseconds>(stopTime - startTime);
            duration[label] += elapsed;
            auto average = (double) duration[label].count() / count[label];
            cout << "||end " << label << " Calls: " << count[label] << " Last: " << elapsed.count() << " ms Total: " << duration[label].count() << " ms Average: " << average << " ms||"
                 << endl;
        }

    private:
        chrono::time_point<chrono::high_resolution_clock> startTime{};
        string label;
    };

    class SummaryProfileBlock {
    public:
        SummaryProfileBlock(const char *file, const string &func, const int line) {
            stringstream ss;
            ss << file << ":" << func << ":" << line;
            this->label = ss.str();
            blockEntered();
        }

        ~SummaryProfileBlock() {
            blockLeft();
        }

        void blockEntered() {
            startTime = chrono::high_resolution_clock::now();
        }

        void blockLeft() {
            thread_local map<string, chrono::microseconds> duration{};
            thread_local map<string, double> count{};
            thread_local map<string, int> count_from_last_print{};
            auto stopTime = chrono::high_resolution_clock::now();
            auto elapsed = chrono::duration_cast<chrono::microseconds>(stopTime - startTime);
            duration[label] += elapsed;
            count[label] += 1;
            if (count_from_last_print[label] > 2000) {
                auto average = (double) duration[label].count() / count[label];
                cout << "||" << label << " Calls: " << count[label] << " Last: " << elapsed.count() << " ms Total: " << duration[label].count() << " ms Average: " << average << " ms)||"
                     << endl;
                count_from_last_print[label] = 0;
            } else {
                count_from_last_print[label] += 1;
            }
        }

    private:
        chrono::time_point<chrono::high_resolution_clock> startTime{};
        string label;
    };

//#define HAPPYML_ML_PROFILE_DETAILS
//#define HAPPYML_ML_PROFILE_SUMMARY

#ifdef HAPPYML_ML_PROFILE_DETAILS
#define PROFILE_BLOCK(x) ProfileBlock x(__FILE__, __func__, __LINE__)
#else
#ifdef HAPPYML_ML_PROFILE_SUMMARY
#define PROFILE_BLOCK(x) SummaryProfileBlock x(__FILE__, __func__, __LINE__)
#else
#define PROFILE_BLOCK(x) ((void)0)
#endif // HAPPYML_ML_PROFILE_SUMMARY
#endif //HAPPYML_ML_PROFILE_DETAILS
}
#endif //HAPPYML_BASIC_PROFILER_HPP
