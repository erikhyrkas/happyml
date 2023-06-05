//
// Created by Erik Hyrkas on 12/26/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_EXIT_STRATEGY_HPP
#define HAPPYML_EXIT_STRATEGY_HPP

#include <iostream>

#define FIFTEEN_SECONDS_MS  15000
#define THIRTY_SECONDS_MS   30000
#define MINUTE_MS           60000
#define FIVE_MINUTES_MS    300000
#define FIFTEEN_MINUTES_MS 900000
#define HALF_HOUR_MS      1800000
#define HOUR_MS           3600000
#define EIGHT_HOURS      28800000
#define DAY_MS           86400000
#define NINETY_DAYS_MS 7776000000

using namespace std;

namespace happyml {

    class ExitStrategy {
    public:
        virtual bool isDone(size_t currentEpoch, float loss, int64_t trainingElapsedTimeInMilliseconds) = 0;

        virtual string whyDone(size_t currentEpoch, float loss, int64_t trainingElapsedTimeInMilliseconds) = 0;
    };

    class DefaultExitStrategy : public ExitStrategy {
    public:
        explicit DefaultExitStrategy(const size_t patience,
                                     const int64_t maxElapsedTime,
                                     const size_t maxEpochs,
                                     const float zeroPrecisionTolerance,
                                     const float improvementTolerance,
                                     const size_t minEpochs,
                                     const float maxDegradationTolerance) {
            this->patience = patience;
            this->maxEpochs = maxEpochs;
            this->maxElapsedTime = maxElapsedTime;
            // I could have called zeroPrecisionTolerance "epsilon",
            // but I wanted to use a term that everybody could understand.
            // We want to stop training when we reach zero, but it's unlikely
            // that we'd ever hit perfectly zero, so what is close enough?
            // That "close enough" is zeroPrecisionTolerance.
            this->zeroPrecisionTolerance = zeroPrecisionTolerance;

            // Could probably also be called "epsilon", this is the minimum amount of
            // improvement we need to show in an epoch before giving up.
            this->improvementTolerance = improvementTolerance;
            this->minEpochs = minEpochs;
            this->maxDegradationTolerance = maxDegradationTolerance;

            lowestLossEpoch = 0;
            lowestLoss = INFINITY;
        }

        bool isDone(size_t currentEpoch, float loss, int64_t trainingElapsedTimeInMilliseconds) override {
            if (loss + improvementTolerance <= lowestLoss) {
                lowestLoss = min(loss, lowestLoss);
                lowestLossEpoch = currentEpoch;
            } else if (lowestLossEpoch > currentEpoch) {
                throw runtime_error("IMPOSSIBLE: lowestLossEpoch > currentEpoch. Did you reuse an exit strategy?");
            }
            // If we're not improving, we're degrading. If we're degrading too fast, we're done.
            const auto degradation = (loss - lowestLoss) / lowestLoss;
            const auto elapsedEpochsSinceLowestEpoch = currentEpoch - lowestLossEpoch;
            const auto done = (currentEpoch >= minEpochs) &&
                              (currentEpoch >= maxEpochs ||
                               degradation >= maxDegradationTolerance ||
                               trainingElapsedTimeInMilliseconds >= maxElapsedTime ||
                               elapsedEpochsSinceLowestEpoch >= patience ||
                               loss <= zeroPrecisionTolerance);
            return done;
        }

        string whyDone(size_t currentEpoch, float loss, int64_t trainingElapsedTimeInMilliseconds) override {
            if (currentEpoch < minEpochs) {
                stringstream ss;
                ss << "Should not be done yet: Current Epoch (" << currentEpoch << ") < Minimum Epochs (" << minEpochs << ")";
                return ss.str();
            }
            if (currentEpoch >= maxEpochs) {
                stringstream ss;
                ss << "Current Epoch (" << currentEpoch << ") >= Maximum Epochs (" << maxEpochs << ")";
                return ss.str();
            }
            const auto degradation = (loss - lowestLoss) / lowestLoss;
            if (degradation >= maxDegradationTolerance) {
                stringstream ss;
                ss << std::fixed << std::setprecision(15);
                ss << "Degradation (" << degradation << ") >= Maximum Degradation Tolerance (" << maxDegradationTolerance << ")";
                return ss.str();
            }
            if (trainingElapsedTimeInMilliseconds >= maxElapsedTime) {
                stringstream ss;
                ss << "Training Elapsed Time In Milliseconds (" << trainingElapsedTimeInMilliseconds << ") >= Maximum Elapsed Time (" << maxElapsedTime << ")";
                return ss.str();
            }
            const auto elapsedEpochsSinceLowestEpoch = currentEpoch - lowestLossEpoch;
            if (elapsedEpochsSinceLowestEpoch >= patience) {
                stringstream ss;
                ss << std::fixed << std::setprecision(15);
                ss << "Elapsed Epochs Since Lowest Epoch (" << currentEpoch << "-" << lowestLossEpoch
                   << "=" << elapsedEpochsSinceLowestEpoch << ") >= Patience (" << patience << "); Lowest Loss is " << lowestLoss
                   << " and Improvement Tolerance is " << improvementTolerance;
                return ss.str();
            }
            if (loss <= zeroPrecisionTolerance) {
                stringstream ss;
                ss << "Loss (" << loss << ") <= Zero Precision Tolerance (" << zeroPrecisionTolerance << ")";
                return ss.str();
            }
            return "Unknown";
        }

    private:
        size_t patience;
        size_t maxEpochs;
        int64_t maxElapsedTime;
        float zeroPrecisionTolerance;
        float improvementTolerance;
        float lowestLoss;
        size_t lowestLossEpoch;
        size_t minEpochs;
        float maxDegradationTolerance;
    };
}
#endif // HAPPYML_EXIT_STRATEGY_HPP