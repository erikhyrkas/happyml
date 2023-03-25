//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TIMERS_HPP
#define HAPPYML_TIMERS_HPP

#include <iostream>
#include <chrono>

using namespace std;

namespace happyml {

    class ElapsedTimer {
    public:
        ElapsedTimer() {
            startTime = chrono::high_resolution_clock::now();
        }

        int64_t getMicroseconds() {
            auto stopTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(stopTime - startTime);
            startTime = chrono::high_resolution_clock::now();
            return duration.count();
        }

        int64_t peekMicroseconds() {
            auto stopTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(stopTime - startTime);
            return duration.count();
        }

        int64_t getMilliseconds() {
            auto stopTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(stopTime - startTime);
            startTime = chrono::high_resolution_clock::now();
            return duration.count();
        }

        int64_t peekMilliseconds() {
            auto stopTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(stopTime - startTime);
            return duration.count();
        }

        int64_t getSeconds() {
            auto stopTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(stopTime - startTime);
            startTime = chrono::high_resolution_clock::now();
            return duration.count();
        }

        int64_t peekSeconds() {
            auto stopTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(stopTime - startTime);
            startTime = chrono::high_resolution_clock::now();
            return duration.count();
        }

    private:
        chrono::time_point <chrono::high_resolution_clock> startTime;
    };

    class SimpleTimer {
    public:
        SimpleTimer() = default;

        void start() {
            startTime = chrono::high_resolution_clock::now();
        }

        void stop() {
            stopTime = chrono::high_resolution_clock::now();
        }

        void printMicroseconds() {
            auto duration = chrono::duration_cast<chrono::microseconds>(stopTime - startTime);
            cout << "Elapsed Time: " << duration.count() << " microseconds" << endl;
        }

        void printMilliseconds() {
            auto duration = chrono::duration_cast<chrono::milliseconds>(stopTime - startTime);
            cout << "Elapsed Time: " << duration.count() << " milliseconds" << endl;
        }

        void printSeconds() {
            auto duration = chrono::duration_cast<chrono::seconds>(stopTime - startTime);
            cout << "Elapsed Time: " << duration.count() << " seconds" << endl;
        }

    private:
        chrono::time_point <chrono::high_resolution_clock> startTime;
        chrono::time_point <chrono::high_resolution_clock> stopTime;
    };

    class EvenMoreSimpleTimer {
    public:
        EvenMoreSimpleTimer() {
            startTime = chrono::high_resolution_clock::now();
        }

        void printMicroseconds() {
            auto stopTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(stopTime - startTime);
            cout << "Elapsed Time: " << duration.count() << " microseconds" << endl;
            startTime = chrono::high_resolution_clock::now();
        }

        void printMilliseconds() {
            auto stopTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(stopTime - startTime);
            cout << "Elapsed Time: " << duration.count() << " milliseconds" << endl;
            startTime = chrono::high_resolution_clock::now();
        }

        void printSeconds() {
            auto stopTime = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(stopTime - startTime);
            cout << "Elapsed Time: " << duration.count() << " seconds" << endl;
            startTime = chrono::high_resolution_clock::now();
        }

    private:
        chrono::time_point <chrono::high_resolution_clock> startTime;
    };

}
#endif //HAPPYML_TIMERS_HPP
