//
// Created by Erik Hyrkas on 10/30/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_TENSOR_STATS_HPP
#define HAPPYML_TENSOR_STATS_HPP

#include <execution>
#include <future>
#include <array>
#include <iostream>
#include <iomanip>

namespace happyml {

#define FIT_BIAS_FOR_100 0
#define FIT_BIAS_FOR_80 1
#define FIT_BIAS_FOR_50 2

    class TensorStats {
    public:
        explicit TensorStats(BaseTensor &source) : TensorStats(source, FIT_BIAS_FOR_80, true) {}

        explicit TensorStats(BaseTensor &source, int bias_fit) : TensorStats(source, bias_fit, true) {}

        // See FIT_BIAS_FOR_100, FIT_BIAS_FOR_90, FIT_BIAS_FOR_50
        explicit TensorStats(BaseTensor &source, int bias_fit, bool require_0_for_fit) {
            // The source could be a tensor or a view. Remember, if we are calling a view,
            // as we iterate, we could be touching many underlying records.
            // Why not turn the views into a matrix first to avoid complicated calculations and cpu consumption?
            // Well, our tensor might have billions of entries and iterating over it twice
            // may make it possible to do something we couldn't otherwise do with the same accuracy.
            // A standard float is 4 bytes. Even 20 billion entries would be 80,000,000,000 bytes or
            // ~75 gigabytes. Even by using 8-bit floats, that same matrix would hold 18.6 gigabytes.
            // At no point, are we going to hold a 32-bit floating point representation of all of those 8-bit
            // numbers, so we have to get clever.
            // I'm using quarter to hash a float. This is imperfect,
            // since the bias and offset may lead to all numbers being jammed to one
            // end of the spectrum. However, this means that we'll only have 256 entries in our bag.
            // In an effort to only iterate over the matrix one time (which might have billions of elements),
            // I'll capture the information at more than one granularity and then calculate what's the
            // best granularity to capture most of the rows with the greatest accuracy.
            // While I'm spending kilobytes of memory on this compared to only doing this once with a low bias,
            // creating a more accurate representation for a matrix is our overall goal. Spend a little compute
            // and track groups of quarters to make them the best representations we can manage.
            this->elementCount = 0;
            this->biasFit = bias_fit;
            this->minValue = 0;
            this->maxValue = 0;
            this->require0ForFit = require_0_for_fit;
            const size_t rows = source.rowCount();
            const size_t cols = source.columnCount();
            const size_t channels = source.channelCount();
            auto bagCounts = make_shared<BagCounts>();
            // TODO: We can improve the conditions in which we are single threaded vs concurrent. This
            // works on my machine, but it isn't a general solution.
//        cout << "elements_per_channel: " << source.elementsPerChannel() << endl;
            if (source.elementsPerChannel() < 100000000) {
//            cout << "single thread" << endl;
                for (size_t channel = 0; channel < channels; channel++) {
                    for (size_t row = 0; row < rows; row++) {
                        for (size_t col = 0; col < cols; col++) {
                            populateBags(&source, row, col, channel, bagCounts);
                        }
                    }
                }
            } else {
                queue<future<void>> futures;
                // this is an imperfect rule. We want to have adequate work for each thread to
                // do that it makes up for the overhead of the thread itself.
                if (cols >= rows) {
//                cout << "by col " << rows << ", " << cols << endl;
                    // TODO: this isn't a general solution. Works on my machine.
                    const size_t wait_amount = 8096;
                    for (size_t channel = 0; channel < channels; channel++) {
                        for (size_t row = 0; row < rows; row++) {
                            auto next_async = std::async(std::launch::async, populateBagsByCol, &source, row, cols,
                                                         channel,
                                                         bagCounts);
                            futures.push(std::move(next_async));
                            if (futures.size() >= wait_amount) {
                                wait(futures);
                            }
                        }
                    }
                } else {
//                cout << "by row " << rows << ", " << cols << endl;
                    // TODO: this isn't a general solution. Works on my machine.
                    const size_t wait_amount = 8096;
                    for (size_t channel = 0; channel < channels; channel++) {
                        for (size_t col = 0; col < cols; col++) {
                            auto next_async = std::async(std::launch::async, populateBagsByRow, &source, rows, col,
                                                         channel,
                                                         bagCounts);
                            futures.push(std::move(next_async));
                            if (futures.size() >= wait_amount) {
                                wait(futures);
                            }
                        }
                    }
                }
                wait(futures);
            }

            // counts should be same for all bags, so we'll just count one
            countElementsAndFindMinMax(bagCounts->bagCounts14);


            // This could be more efficient. We calculate for bias 0, then discard after we find the target range
            if (quarterToFloat(QUARTER_MIN, 14) <= minValue &&
                quarterToFloat(QUARTER_MAX, 14) >= maxValue) {
//            cout << "min and max fit in quarter 14: " << min_value << " -> " << max_value << " must fit in "
//                      << quarter_to_float(QUARTER_MIN, 14, 0) << " -> " << quarter_to_float(QUARTER_MAX, 14, 0)
//                      << endl;
                bag(bagCounts->bagCounts14);
            } else if (quarterToFloat(QUARTER_MIN, 8) <= minValue &&
                       quarterToFloat(QUARTER_MAX, 8) >= maxValue) {
//            cout << "min and max fit in quarter 8: " << min_value << " -> " << max_value << " must fit in "
//                      << quarter_to_float(QUARTER_MIN, 8, 0) << " -> " << quarter_to_float(QUARTER_MAX, 8, 0)
//                      << endl;
                bag(bagCounts->bagCounts8);
            } else if (quarterToFloat(QUARTER_MIN, 4) <= minValue &&
                       quarterToFloat(QUARTER_MAX, 4) >= maxValue) {
//            cout << "min and max fit in quarter 4: " << min_value << " -> " << max_value << " must fit in "
//                      << quarter_to_float(QUARTER_MIN, 4, 0) << " -> " << quarter_to_float(QUARTER_MAX, 4, 0)
//                      << endl;
                bag(bagCounts->bagCounts4);
            } else if (quarterToFloat(QUARTER_MIN, 1) <= minValue &&
                       quarterToFloat(QUARTER_MAX, 1) >= maxValue) {
//            cout << "min and max fit in quarter 1: " << min_value << " -> " << max_value << " must fit in "
//                      << quarter_to_float(QUARTER_MIN, 1, 0) << " -> " << quarter_to_float(QUARTER_MAX, 1, 0)
//                      << endl;
                bag(bagCounts->bagCounts1);
            } else {
//            cout << "min and max forced to fit in quarter -4: " << min_value << " -> " << max_value
//                      << " must fit in " << quarter_to_float(QUARTER_MIN, -4, 0) << " -> "
//                      << quarter_to_float(QUARTER_MAX, -4, 0) << endl;
                bag(bagCounts->bagCountsNegative4);
            }
            double wide_target_range;
            if (bias_fit == FIT_BIAS_FOR_80) {
                wide_target_range = tenTo90Range();
            } else if (bias_fit == FIT_BIAS_FOR_50) {
                wide_target_range = q2ToQ3Range();
            } else {
                wide_target_range = fullRange();
            }

//        cout << "wide range: " << wide_target_range << endl;
            if (bagAndCheckRangeForBiasGoal(bagCounts->bagCounts14, 14, wide_target_range)) {
                recommendedBias = 14;
            } else if (bagAndCheckRangeForBiasGoal(bagCounts->bagCounts8, 8, wide_target_range)) {
                recommendedBias = 8;
            } else if (bagAndCheckRangeForBiasGoal(bagCounts->bagCounts4, 4, wide_target_range)) {
                recommendedBias = 4;
            } else if (bagAndCheckRangeForBiasGoal(bagCounts->bagCounts1, 1, wide_target_range)) {
                recommendedBias = 1;
            } else {
                // we tried to fit, but we're left with the default
                bag(bagCounts->bagCountsNegative4);
                recommendedBias = -4;
            }

            const double half_range = wide_target_range / 2;
            if (bias_fit == FIT_BIAS_FOR_80) {
                if (require_0_for_fit) {
                    auto low = std::min(0.0f, eightyValues.at(1));
                    recommendedOffset = (float) (low + half_range);
                } else {
                    recommendedOffset = eightyValues.at(2);
                }
            } else if (bias_fit == FIT_BIAS_FOR_50) {
                if (require_0_for_fit) {
                    auto low = std::min(0.0f, quarterValues.at(1));
                    recommendedOffset = (float) (low + half_range);
                } else {
                    recommendedOffset = quarterValues.at(2);
                }
            } else {
                if (require_0_for_fit) {
                    auto low = std::min(0.0f, eightyValues.at(0));
                    recommendedOffset = (float) (low + half_range);
                } else {
                    recommendedOffset = (float) (eightyValues.at(0) + half_range);
                }
            }
//        cout << "finished constructor" << endl;
        }

        void print() {
//        cout << "Double max: " << std::fixed << std::numeric_limits<double>::max() << endl;
            cout << "Bag contents(" << elementCount << "/" << bagElements.size() << "): [" << endl;
            for (auto it = bagElements.begin(); it != bagElements.end(); ++it) {
                const auto val = (float) it->at(0);
                const auto count = (unsigned long) it->at(1);
                cout << std::fixed << "\t" << val << "\t" << std::setw(10) << count << "\t";
                const auto dots_len = 100 * ((double) count / (double) elementCount);
                for (uint32_t i = 0; i < dots_len; i++) {
                    cout << ".";
                }
                cout << endl;
            }
            cout << "]" << endl << "Quartile parts: ";
            string delim;
            for (auto it = quarterValues.begin(); it != quarterValues.end(); ++it) {
                cout << delim << std::fixed << *it;
                delim = ", ";
            }
            cout << endl << "80% parts: ";
            delim = "";
            for (auto it = eightyValues.begin(); it != eightyValues.end(); ++it) {
                cout << delim << std::fixed << *it;
                delim = ", ";
            }
            cout << endl << "recommended bias: " << recommendedBias << endl;
            cout << "recommended offset: " << std::fixed << std::setprecision(15) << recommendedOffset
                 << endl;
            cout << "min: " << std::fixed << minValue << endl;
            cout << "max: " << std::fixed << maxValue << endl;
            cout << "range: " << std::fixed << (maxValue - minValue) << endl;
            cout << "Zero required for fit: " << (require0ForFit ? "true" : "false") << endl;
        }

        [[nodiscard]] int getRecommendedBias() const {
            return recommendedBias;
        }

        [[nodiscard]] float getRecommendedOffset() const {
            return recommendedOffset;
        }

        // See FIT_BIAS_FOR_100, FIT_BIAS_FOR_90, FIT_BIAS_FOR_50
        [[nodiscard]] bool targetBiasFit() const {
            return biasFit;
        }

    private:
        int biasFit; // See FIT_BIAS_FOR_100, FIT_BIAS_FOR_90, FIT_BIAS_FOR_50
        unsigned long elementCount;
        vector<array<double, 2>> bagElements;
        vector<float> quarterValues; // five values from: 0%, 25%, 50%, 75%, 100%
        vector<float> eightyValues; // five values from: 0%, 10%, 50%, 90%, 100%
        int recommendedBias;
        float recommendedOffset;
        double minValue;
        double maxValue;
        bool require0ForFit;

        static bool bagEntryCompare(array<double, 2> a, array<double, 2> b) {
            return a.at(0) < b.at(0);
        }

        struct BagCounts {
            BagCounts() : bagCounts14{}, bagCounts8{}, bagCounts4{}, bagCounts1{}, bagCountsNegative4{} {}

            mutex bagMutex;
            array<array<double, 2>, 256> bagCounts14;
            array<array<double, 2>, 256> bagCounts8;
            array<array<double, 2>, 256> bagCounts4;
            array<array<double, 2>, 256> bagCounts1;
            array<array<double, 2>, 256> bagCountsNegative4;
        };

        static void addToBag(array<array<double, 2>, 256> &bagCounts, const float f, const int bias) {
            quarter q = floatToQuarter(f, bias);
            // gravitate to numbers that are furthest from zero, unless zero (this may not be needed
            // since it should initialize to zero.)
            // is branching too expensive here? maybe we always assign, accepting slightly worse
            // results for better performance?
            const auto old_val = bagCounts[q][0];
            if ((f > 0 && f > old_val) || (f < 0 && f < old_val) || (f == 0)) {
                bagCounts[q][0] = f;
            }
            bagCounts[q][1] += 1.0;
        }

        static void populateBags(BaseTensor *source,
                                 const size_t row,
                                 const size_t col,
                                 const size_t channel,
                                 const shared_ptr<BagCounts> &bagCounts) {
            float f = source->getValue(row, col, channel);
            if (isinf(f) || isnan(f)) {
                return;
            }
            addToBag(bagCounts->bagCounts14, f, 14);
            addToBag(bagCounts->bagCounts8, f, 8);
            addToBag(bagCounts->bagCounts4, f, 4);
            addToBag(bagCounts->bagCounts1, f, 1);
            addToBag(bagCounts->bagCountsNegative4, f, -4);
        }

        static void populateBagsByCol(BaseTensor *source,
                                      size_t row,
                                      size_t maxCols,
                                      size_t channel,
                                      const shared_ptr<BagCounts> &bagCounts) {
            auto local = make_shared<BagCounts>();
            for (size_t col = 0; col < maxCols; col++) {
                populateBags(source, row, col, channel, local);
            }
            const lock_guard<mutex> lock(bagCounts->bagMutex);
            size_t index = 0;
            for (const auto &[val, count]: local->bagCounts14) {
                const double original = bagCounts->bagCounts14[index][0];
                if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                    bagCounts->bagCounts14[index][0] = val;
                }
                bagCounts->bagCounts14[index][1] += count;
                index++;
            }
            index = 0;
            for (const auto &[val, count]: local->bagCounts8) {
                const double original = bagCounts->bagCounts8[index][0];
                if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                    bagCounts->bagCounts8[index][0] = val;
                }
                bagCounts->bagCounts8[index][1] += count;
                index++;
            }
            index = 0;
            for (const auto &[val, count]: local->bagCounts4) {
                const double original = bagCounts->bagCounts4[index][0];
                if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                    bagCounts->bagCounts4[index][0] = val;
                }
                bagCounts->bagCounts4[index][1] += count;
                index++;
            }
            index = 0;
            for (const auto &[val, count]: local->bagCounts1) {
                const double original = bagCounts->bagCounts1[index][0];
                if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                    bagCounts->bagCounts1[index][0] = val;
                }
                bagCounts->bagCounts1[index][1] += count;
                index++;
            }
            index = 0;
            for (const auto &[val, count]: local->bagCountsNegative4) {
                const double original = bagCounts->bagCountsNegative4[index][0];
                if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                    bagCounts->bagCountsNegative4[index][0] = val;
                }
                bagCounts->bagCountsNegative4[index][1] += count;
                index++;
            }
        }

        static void populateBagsByRow(BaseTensor *source,
                                      size_t maxRows,
                                      size_t col,
                                      size_t channel,
                                      const shared_ptr<BagCounts> &bagCounts) {
            auto local = make_shared<BagCounts>();
            for (size_t row = 0; row < maxRows; row++) {
                populateBags(source, row, col, channel, local);
            }
            const lock_guard<mutex> lock(bagCounts->bagMutex);
            size_t index = 0;
            for (const auto &[val, count]: local->bagCounts14) {
                const double original = bagCounts->bagCounts14[index][0];
                if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                    bagCounts->bagCounts14[index][0] = val;
                }
                bagCounts->bagCounts14[index][1] += count;
                index++;
            }
            index = 0;
            for (const auto &[val, count]: local->bagCounts8) {
                const double original = bagCounts->bagCounts8[index][0];
                if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                    bagCounts->bagCounts8[index][0] = val;
                }
                bagCounts->bagCounts8[index][1] += count;
                index++;
            }
            index = 0;
            for (const auto &[val, count]: local->bagCounts4) {
                const double original = bagCounts->bagCounts4[index][0];
                if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                    bagCounts->bagCounts4[index][0] = val;
                }
                bagCounts->bagCounts4[index][1] += count;
                index++;
            }
            index = 0;
            for (const auto &[val, count]: local->bagCounts1) {
                const double original = bagCounts->bagCounts1[index][0];
                if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                    bagCounts->bagCounts1[index][0] = val;
                }
                bagCounts->bagCounts1[index][1] += count;
                index++;
            }
            index = 0;
            for (const auto &[val, count]: local->bagCountsNegative4) {
                const double original = bagCounts->bagCountsNegative4[index][0];
                if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                    bagCounts->bagCountsNegative4[index][0] = val;
                }
                bagCounts->bagCountsNegative4[index][1] += count;
                index++;
            }
        }

        static void wait(queue<future<void>> &futures) {
            while (!futures.empty()) {
                futures.front().wait();
                futures.pop();
            }
        }


        inline void countElementsAndFindMinMax(array<array<double, 2>, 256> &bagCounts) {
            minValue = HUGE_VAL;
            maxValue = -HUGE_VAL;
            for (auto &it: bagCounts) {
                double val = it.at(0);
                if (val < minValue) {
                    minValue = val;
                }
                if (val > maxValue) {
                    maxValue = val;
                }
                elementCount += (unsigned long) it.at(1);
            }
        }

        double q2ToQ3Range() {
            if (quarterValues.size() != 5) {
                cout << "Mid 50 quarter values size: " << quarterValues.size() << endl;
                throw runtime_error("Mid 50 range calculation only works after quarter_values are populated.");
            }
            if (require0ForFit) {
                return std::abs(std::max(0.0f, quarterValues.at(3)) - std::min(0.0f, quarterValues.at(1)));
            }
            return std::abs(quarterValues.at(3) - quarterValues.at(1));
        }

        double tenTo90Range() {
            if (eightyValues.size() != 5) {
                cout << "Mid 80 values size: " << eightyValues.size() << endl;
                throw runtime_error("Mid 80 range calculation only works after eighty_values are populated.");
            }
            if (require0ForFit) {
                return std::abs(std::max(0.0f, eightyValues.at(3)) - std::min(0.0f, eightyValues.at(1)));
            }
            return std::abs(eightyValues.at(3) - eightyValues.at(1));
        }

        double fullRange() {
            if (quarterValues.size() != 5) {
                cout << "Full range quarter values size: " << quarterValues.size() << endl;
                throw runtime_error("full range calculation only works after quarter_values are populated.");
            }
            if (require0ForFit) {
                return std::abs(std::max(0.0f, quarterValues.at(4)) - std::min(0.0f, quarterValues.at(0)));
            }
            return std::abs(quarterValues.at(4) - quarterValues.at(0));
        }

        void bag(array<array<double, 2>, 256> &bag_counts) {
            buildBagFromCounts(bag_counts);
            calculateQuarters();
            calculateEightyPercent();
        }

        void calculateEightyPercent() {
            if (elementCount == 0 || bagElements.empty()) {
                return;
            }
            const unsigned long ten_percent = elementCount / 10;
            const auto fifty_percent = std::min((unsigned long) (elementCount * 0.5),
                                                5 * ten_percent); //round down for small values
            const auto ninety_percent = std::min((unsigned long) (elementCount * 0.9),
                                                 9 * ten_percent); //round down for small values
//        cout << "Ten Percent Size: " << ten_percent << endl;
//        cout << "Fifty Percent Boundary: " << fifty_percent << endl;
//        cout << "Ninety Percent Boundary: " << ninety_percent << endl;

            eightyValues.clear();
            eightyValues.push_back((float) bagElements.front().at(0));
            unsigned long current_element = 0;
            for (auto it = bagElements.begin(); it != bagElements.end(); ++it) {
                current_element += (unsigned long) it->at(1);
                if (eightyValues.size() == 3 && current_element >= ninety_percent) {
                    eightyValues.push_back((float) it->at(0));
                    break;
                }
                if (eightyValues.size() == 2 && current_element >= fifty_percent) {
                    eightyValues.push_back((float) it->at(0));
                }
                if (eightyValues.size() == 1 && current_element > ten_percent) {
                    eightyValues.push_back((float) it->at(0));
                }
            }
            while (eightyValues.size() < 5) {
                eightyValues.push_back((float) bagElements.back().at(0));
            }
        }

        void calculateQuarters() {
            const unsigned long quarter_size = elementCount / 4;
//        cout << "Quarter Size: " << quarter_size << endl;
            unsigned long next_quarter = quarter_size;
            unsigned long current_element = 0;
            quarterValues.clear();
            quarterValues.push_back((float) bagElements.front().at(0));
            for (auto it = bagElements.begin(); it != bagElements.end(); ++it) {
//            cout << "[" << it->at(0) << ", " <<  (unsigned long)it->at(1) << "]" << endl;
                current_element += (unsigned long) it->at(1);
//            cout << "current_element: " << current_element << endl;
                while (current_element >= next_quarter) {
//                cout << "adding quarter current_element: " << current_element
//                          << " with value: " << it->at(0) << endl;
                    quarterValues.push_back((float) it->at(0));
                    if (quarterValues.size() == 4) {
                        break;
                    }
                    next_quarter += quarter_size;
                }
                if (quarterValues.size() == 4) {
                    break;
                }
            }
            quarterValues.push_back((float) bagElements.back().at(0));
//        cout << "Done calculate_quarters" << endl;
        }

        void buildBagFromCounts(array<array<double, 2>, 256> &bagCounts) {
            bagElements.clear();
            for (auto &bagCount: bagCounts) {
                if (bagCount.at(1) > 0) {
                    bagElements.push_back(bagCount);
                }
            }
            std::sort(bagElements.begin(), bagElements.end(), bagEntryCompare);
        }

        bool bagAndCheckRangeForBiasGoal(array<array<double, 2>, 256> &bagCounts, int bias,
                                         double wideTargetRange) {
            double biasRange = calculateBiasRange(bias);
            if (biasRange < wideTargetRange) {
                return false;
            }
            bag(bagCounts);
            return (biasFit == FIT_BIAS_FOR_100 && fullRange() <= biasRange) ||
                   (biasFit == FIT_BIAS_FOR_80 && tenTo90Range() <= biasRange) ||
                   (biasFit == FIT_BIAS_FOR_50 && q2ToQ3Range() <= biasRange);
        }
    };
}
#endif //HAPPYML_TENSOR_STATS_HPP
