//
// Created by Erik Hyrkas on 10/30/2022.
//

#ifndef MICROML_TENSOR_STATS_HPP
#define MICROML_TENSOR_STATS_HPP

#include <execution>
#include <future>
#include <iterator>
#include <array>
#include <iostream>
#include <format>
#include <iomanip>
#include "tensor.hpp"

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
        this->element_count = 0;
        this->bias_fit = bias_fit;
        this->min_value = 0;
        this->max_value = 0;
        this->require_0_for_fit = require_0_for_fit;
        const size_t rows = source.row_count();
        const size_t cols = source.column_count();
        const size_t channels = source.channel_count();
        auto bagCounts = std::make_shared<BagCounts>();
        // TODO: We can improve the conditions in which we are single threaded vs concurrent. This
        // works on my machine, but it isn't a general solution.
//        std::cout << "elements_per_channel: " << source.elements_per_channel() << std::endl;
        if (source.elements_per_channel() < 100000000) {
//            std::cout << "single thread" << std::endl;
            for (size_t channel = 0; channel < channels; channel++) {
                for (size_t row = 0; row < rows; row++) {
                    for (size_t col = 0; col < cols; col++) {
                        populate_bags(&source, row, col, channel, bagCounts);
                    }
                }
            }
        } else {
            std::queue<std::future<void>> futures;
            // this is an imperfect rule. We want to have adequate work for each thread to
            // do that it makes up for the overhead of the thread itself.
            if (cols >= rows) {
//                std::cout << "by col " << rows << ", " << cols << std::endl;
                // TODO: this isn't a general solution. Works on my machine.
                const size_t wait_amount = 8096;
                for (size_t channel = 0; channel < channels; channel++) {
                    for (size_t row = 0; row < rows; row++) {
                        auto next_async = std::async(std::launch::async, populate_bags_by_col, &source, row, cols,
                                                     channel,
                                                     bagCounts);
                        futures.push(std::move(next_async));
                        if (futures.size() >= wait_amount) {
                            wait(futures);
                        }
                    }
                }
            } else {
//                std::cout << "by row " << rows << ", " << cols << std::endl;
                // TODO: this isn't a general solution. Works on my machine.
                const size_t wait_amount = 8096;
                for (size_t channel = 0; channel < channels; channel++) {
                    for (size_t col = 0; col < cols; col++) {
                        auto next_async = std::async(std::launch::async, populate_bags_by_row, &source, rows, col,
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
        count_elements_and_find_min_max(bagCounts->bag_counts_14);


        // This could be more efficient. We calculate for bias 0, then discard after we find the target range
        if (quarter_to_float(QUARTER_MIN, 14, 0) <= min_value && quarter_to_float(QUARTER_MAX, 14, 0) >= max_value) {
//            std::cout << "min and max fit in quarter 14: " << min_value << " -> " << max_value << " must fit in "
//                      << quarter_to_float(QUARTER_MIN, 14, 0) << " -> " << quarter_to_float(QUARTER_MAX, 14, 0)
//                      << std::endl;
            bag(bagCounts->bag_counts_14);
        } else if (quarter_to_float(QUARTER_MIN, 8, 0) <= min_value &&
                   quarter_to_float(QUARTER_MAX, 8, 0) >= max_value) {
//            std::cout << "min and max fit in quarter 8: " << min_value << " -> " << max_value << " must fit in "
//                      << quarter_to_float(QUARTER_MIN, 8, 0) << " -> " << quarter_to_float(QUARTER_MAX, 8, 0)
//                      << std::endl;
            bag(bagCounts->bag_counts_8);
        } else if (quarter_to_float(QUARTER_MIN, 4, 0) <= min_value &&
                   quarter_to_float(QUARTER_MAX, 4, 0) >= max_value) {
//            std::cout << "min and max fit in quarter 4: " << min_value << " -> " << max_value << " must fit in "
//                      << quarter_to_float(QUARTER_MIN, 4, 0) << " -> " << quarter_to_float(QUARTER_MAX, 4, 0)
//                      << std::endl;
            bag(bagCounts->bag_counts_4);
        } else if (quarter_to_float(QUARTER_MIN, 1, 0) <= min_value &&
                   quarter_to_float(QUARTER_MAX, 1, 0) >= max_value) {
//            std::cout << "min and max fit in quarter 1: " << min_value << " -> " << max_value << " must fit in "
//                      << quarter_to_float(QUARTER_MIN, 1, 0) << " -> " << quarter_to_float(QUARTER_MAX, 1, 0)
//                      << std::endl;
            bag(bagCounts->bag_counts_1);
        } else {
//            std::cout << "min and max forced to fit in quarter -4: " << min_value << " -> " << max_value
//                      << " must fit in " << quarter_to_float(QUARTER_MIN, -4, 0) << " -> "
//                      << quarter_to_float(QUARTER_MAX, -4, 0) << std::endl;
            bag(bagCounts->bag_counts_negative_4);
        }
        double wide_target_range;
        if (bias_fit == FIT_BIAS_FOR_80) {
            wide_target_range = ten_to_90_range();
        } else if (bias_fit == FIT_BIAS_FOR_50) {
            wide_target_range = q2_to_q3_range();
        } else {
            wide_target_range = full_range();
        }

//        std::cout << "wide range: " << wide_target_range << std::endl;
        if (bag_and_check_range_for_bias_goal(bagCounts->bag_counts_14, 14, wide_target_range)) {
            recommended_bias = 14;
        } else if (bag_and_check_range_for_bias_goal(bagCounts->bag_counts_8, 8, wide_target_range)) {
            recommended_bias = 8;
        } else if (bag_and_check_range_for_bias_goal(bagCounts->bag_counts_4, 4, wide_target_range)) {
            recommended_bias = 4;
        } else if (bag_and_check_range_for_bias_goal(bagCounts->bag_counts_1, 1, wide_target_range)) {
            recommended_bias = 1;
        } else {
            // we tried to fit, but we're left with the default
            bag(bagCounts->bag_counts_negative_4);
            recommended_bias = -4;
        }

        const double half_range = wide_target_range / 2;
        if (bias_fit == FIT_BIAS_FOR_80) {
            if (require_0_for_fit) {
                auto low = std::min(0.0f, eighty_values.at(1));
                recommended_offset = (float)(low + half_range);
            } else {
                recommended_offset = eighty_values.at(2);
            }
        } else if (bias_fit == FIT_BIAS_FOR_50) {
            if (require_0_for_fit) {
                auto low = std::min(0.0f, quarter_values.at(1));
                recommended_offset = (float)(low + half_range);
            } else {
                recommended_offset = quarter_values.at(2);
            }
        } else {
            if (require_0_for_fit) {
                auto low = std::min(0.0f, eighty_values.at(0));
                recommended_offset = (float)(low + half_range);
            } else {
                recommended_offset = (float)(eighty_values.at(0) + half_range);
            }
        }
//        std::cout << "finished constructor" << std::endl;
    }

    void print() {
//        std::cout << "Double max: " << std::fixed << std::numeric_limits<double>::max() << std::endl;
        std::cout << "Bag contents(" << element_count << "/" << bag_elements.size() << "): [" << std::endl;
        for (auto it = bag_elements.begin(); it != bag_elements.end(); ++it) {
            const auto val = (float) it->at(0);
            const auto count = (unsigned long) it->at(1);
            std::cout << std::fixed << "\t" << val << "\t" << std::setw(10) << count << "\t";
            const auto dots_len = 100 * ((double) count / (double) element_count);
            for (uint32_t i = 0; i < dots_len; i++) {
                std::cout << ".";
            }
            std::cout << std::endl;
        }
        std::cout << "]" << std::endl << "Quartile parts: ";
        std::string delim;
        for (auto it = quarter_values.begin(); it != quarter_values.end(); ++it) {
            std::cout << delim << std::fixed << *it;
            delim = ", ";
        }
        std::cout << std::endl << "80% parts: ";
        delim = "";
        for (auto it = eighty_values.begin(); it != eighty_values.end(); ++it) {
            std::cout << delim << std::fixed << *it;
            delim = ", ";
        }
        std::cout << std::endl << "recommended bias: " << recommended_bias << std::endl;
        std::cout << "recommended offset: " << std::fixed << std::setprecision(15) << recommended_offset << std::endl;
        std::cout << "min: " << std::fixed << min_value << std::endl;
        std::cout << "max: " << std::fixed << max_value << std::endl;
        std::cout << "range: " << std::fixed << (max_value - min_value) << std::endl;
        std::cout << "Zero required for fit: " << (require_0_for_fit ? "true" : "false") << std::endl;
    }

    [[nodiscard]] int get_recommended_bias() const {
        return recommended_bias;
    }

    [[nodiscard]] float get_recommended_offset() const {
        return recommended_offset;
    }

    // See FIT_BIAS_FOR_100, FIT_BIAS_FOR_90, FIT_BIAS_FOR_50
    [[nodiscard]] bool target_bias_fit() const {
        return bias_fit;
    }

private:
    int bias_fit; // See FIT_BIAS_FOR_100, FIT_BIAS_FOR_90, FIT_BIAS_FOR_50
    unsigned long element_count;
    std::vector<std::array<double, 2>> bag_elements;
    std::vector<float> quarter_values; // five values from: 0%, 25%, 50%, 75%, 100%
    std::vector<float> eighty_values; // five values from: 0%, 10%, 50%, 90%, 100%
    int recommended_bias;
    float recommended_offset;
    double min_value;
    double max_value;
    bool require_0_for_fit;

    static bool bag_entry_compare(std::array<double, 2> a, std::array<double, 2> b) {
        return a.at(0) < b.at(0);
    }

    struct BagCounts {
        BagCounts() : bag_counts_14{}, bag_counts_8{}, bag_counts_4{}, bag_counts_1{}, bag_counts_negative_4{} {}

        std::mutex bag_mutex;
        std::array<std::array<double, 2>, 256> bag_counts_14;
        std::array<std::array<double, 2>, 256> bag_counts_8;
        std::array<std::array<double, 2>, 256> bag_counts_4;
        std::array<std::array<double, 2>, 256> bag_counts_1;
        std::array<std::array<double, 2>, 256> bag_counts_negative_4;
    };

    static void add_to_bag(std::array<std::array<double, 2>, 256> &bag_counts, const float f, const int bias) {
        quarter q = float_to_quarter(f, bias, 0);
        // gravitate to numbers that are furthest from zero, unless zero (this may not be needed
        // since it should initialize to zero.)
        // is branching too expensive here? maybe we always assign, accepting slightly worse
        // results for better performance?
        const auto old_val = bag_counts[q][0];
        if ((f > 0 && f > old_val) || (f < 0 && f < old_val) || (f == 0)) {
            bag_counts[q][0] = f;
        }
        bag_counts[q][1] += 1.0;
    }

    static void populate_bags(BaseTensor *source,
                              const size_t row,
                              const size_t col,
                              const size_t channel,
                              const std::shared_ptr<BagCounts> &bagCounts) {
        float f = source->get_val(row, col, channel);
        if (isinf(f) || isnan(f)) {
            return;
        }
        add_to_bag(bagCounts->bag_counts_14, f, 14);
        add_to_bag(bagCounts->bag_counts_8, f, 8);
        add_to_bag(bagCounts->bag_counts_4, f, 4);
        add_to_bag(bagCounts->bag_counts_1, f, 1);
        add_to_bag(bagCounts->bag_counts_negative_4, f, -4);
    }

    static void populate_bags_by_col(BaseTensor *source,
                                     size_t row,
                                     size_t max_cols,
                                     size_t channel,
                                     const std::shared_ptr<BagCounts> &bagCounts) {
        auto local = std::make_shared<BagCounts>();
        for (size_t col = 0; col < max_cols; col++) {
            populate_bags(source, row, col, channel, local);
        }
        const std::lock_guard<std::mutex> lock(bagCounts->bag_mutex);
        size_t index = 0;
        for (const auto &[val, count]: local->bag_counts_14) {
            const double original = bagCounts->bag_counts_14[index][0];
            if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                bagCounts->bag_counts_14[index][0] = val;
            }
            bagCounts->bag_counts_14[index][1] += count;
            index++;
        }
        index = 0;
        for (const auto &[val, count]: local->bag_counts_8) {
            const double original = bagCounts->bag_counts_8[index][0];
            if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                bagCounts->bag_counts_8[index][0] = val;
            }
            bagCounts->bag_counts_8[index][1] += count;
            index++;
        }
        index = 0;
        for (const auto &[val, count]: local->bag_counts_4) {
            const double original = bagCounts->bag_counts_4[index][0];
            if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                bagCounts->bag_counts_4[index][0] = val;
            }
            bagCounts->bag_counts_4[index][1] += count;
            index++;
        }
        index = 0;
        for (const auto &[val, count]: local->bag_counts_1) {
            const double original = bagCounts->bag_counts_1[index][0];
            if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                bagCounts->bag_counts_1[index][0] = val;
            }
            bagCounts->bag_counts_1[index][1] += count;
            index++;
        }
        index = 0;
        for (const auto &[val, count]: local->bag_counts_negative_4) {
            const double original = bagCounts->bag_counts_negative_4[index][0];
            if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                bagCounts->bag_counts_negative_4[index][0] = val;
            }
            bagCounts->bag_counts_negative_4[index][1] += count;
            index++;
        }
    }

    static void populate_bags_by_row(BaseTensor *source,
                                     size_t max_rows,
                                     size_t col,
                                     size_t channel,
                                     const std::shared_ptr<BagCounts> &bagCounts) {
        auto local = std::make_shared<BagCounts>();
        for (size_t row = 0; row < max_rows; row++) {
            populate_bags(source, row, col, channel, local);
        }
        const std::lock_guard<std::mutex> lock(bagCounts->bag_mutex);
        size_t index = 0;
        for (const auto &[val, count]: local->bag_counts_14) {
            const double original = bagCounts->bag_counts_14[index][0];
            if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                bagCounts->bag_counts_14[index][0] = val;
            }
            bagCounts->bag_counts_14[index][1] += count;
            index++;
        }
        index = 0;
        for (const auto &[val, count]: local->bag_counts_8) {
            const double original = bagCounts->bag_counts_8[index][0];
            if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                bagCounts->bag_counts_8[index][0] = val;
            }
            bagCounts->bag_counts_8[index][1] += count;
            index++;
        }
        index = 0;
        for (const auto &[val, count]: local->bag_counts_4) {
            const double original = bagCounts->bag_counts_4[index][0];
            if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                bagCounts->bag_counts_4[index][0] = val;
            }
            bagCounts->bag_counts_4[index][1] += count;
            index++;
        }
        index = 0;
        for (const auto &[val, count]: local->bag_counts_1) {
            const double original = bagCounts->bag_counts_1[index][0];
            if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                bagCounts->bag_counts_1[index][0] = val;
            }
            bagCounts->bag_counts_1[index][1] += count;
            index++;
        }
        index = 0;
        for (const auto &[val, count]: local->bag_counts_negative_4) {
            const double original = bagCounts->bag_counts_negative_4[index][0];
            if ((val == 0) || (val < 0 && val < original) || (val > 0 && val > original)) {
                bagCounts->bag_counts_negative_4[index][0] = val;
            }
            bagCounts->bag_counts_negative_4[index][1] += count;
            index++;
        }
    }

    static void wait(std::queue<std::future<void>> &futures) {
        while (!futures.empty()) {
            futures.front().wait();
            futures.pop();
        }
    }


    inline void count_elements_and_find_min_max(std::array<std::array<double, 2>, 256> &bag_counts) {
        min_value = HUGE_VAL;
        max_value = -HUGE_VAL;
        for (auto &it: bag_counts) {
            double val = it.at(0);
            if (val < min_value) {
                min_value = val;
            }
            if (val > max_value) {
                max_value = val;
            }
            element_count += (unsigned long) it.at(1);
        }
    }

    double q2_to_q3_range() {
        if (quarter_values.size() != 5) {
            std::cout << "Mid 50 quarter values size: " << quarter_values.size() << std::endl;
            throw std::exception("Mid 50 range calculation only works after quarter_values are populated.");
        }
        if (require_0_for_fit) {
            return std::abs(std::max(0.0f, quarter_values.at(3)) - std::min(0.0f, quarter_values.at(1)));
        }
        return std::abs(quarter_values.at(3) - quarter_values.at(1));
    }

    double ten_to_90_range() {
        if (eighty_values.size() != 5) {
            std::cout << "Mid 80 values size: " << eighty_values.size() << std::endl;
            throw std::exception("Mid 80 range calculation only works after eighty_values are populated.");
        }
        if (require_0_for_fit) {
            return std::abs(std::max(0.0f, eighty_values.at(3)) - std::min(0.0f, eighty_values.at(1)));
        }
        return std::abs(eighty_values.at(3) - eighty_values.at(1));
    }

    double full_range() {
        if (quarter_values.size() != 5) {
            std::cout << "Full range quarter values size: " << quarter_values.size() << std::endl;
            throw std::exception("full range calculation only works after quarter_values are populated.");
        }
        if (require_0_for_fit) {
            return std::abs(std::max(0.0f, quarter_values.at(4)) - std::min(0.0f, quarter_values.at(0)));
        }
        return std::abs(quarter_values.at(4) - quarter_values.at(0));
    }

    void bag(std::array<std::array<double, 2>, 256> &bag_counts) {
        build_bag_from_counts(bag_counts);
        calculate_quarters();
        calculate_eighty_percent();
    }

    void calculate_eighty_percent() {
        if (element_count == 0 || bag_elements.empty()) {
            return;
        }
        const unsigned long ten_percent = element_count / 10;
        const auto fifty_percent = std::min((unsigned long) (element_count * 0.5),
                                            5 * ten_percent); //round down for small values
        const auto ninety_percent = std::min((unsigned long) (element_count * 0.9),
                                             9 * ten_percent); //round down for small values
//        std::cout << "Ten Percent Size: " << ten_percent << std::endl;
//        std::cout << "Fifty Percent Boundary: " << fifty_percent << std::endl;
//        std::cout << "Ninety Percent Boundary: " << ninety_percent << std::endl;

        eighty_values.clear();
        eighty_values.push_back((float) bag_elements.front().at(0));
        unsigned long current_element = 0;
        for (auto it = bag_elements.begin(); it != bag_elements.end(); ++it) {
            current_element += (unsigned long) it->at(1);
            if (eighty_values.size() == 3 && current_element >= ninety_percent) {
                eighty_values.push_back((float) it->at(0));
                break;
            }
            if (eighty_values.size() == 2 && current_element >= fifty_percent) {
                eighty_values.push_back((float) it->at(0));
            }
            if (eighty_values.size() == 1 && current_element > ten_percent) {
                eighty_values.push_back((float) it->at(0));
            }
        }
        while (eighty_values.size() < 5) {
            eighty_values.push_back((float) bag_elements.back().at(0));
        }
    }

    void calculate_quarters() {
        const unsigned long quarter_size = element_count / 4;
//        std::cout << "Quarter Size: " << quarter_size << std::endl;
        unsigned long next_quarter = quarter_size;
        unsigned long current_element = 0;
        quarter_values.clear();
        quarter_values.push_back((float) bag_elements.front().at(0));
        for (auto it = bag_elements.begin(); it != bag_elements.end(); ++it) {
//            std::cout << "[" << it->at(0) << ", " <<  (unsigned long)it->at(1) << "]" << std::endl;
            current_element += (unsigned long) it->at(1);
//            std::cout << "current_element: " << current_element << std::endl;
            while (current_element >= next_quarter) {
//                std::cout << "adding quarter current_element: " << current_element
//                          << " with value: " << it->at(0) << std::endl;
                quarter_values.push_back((float) it->at(0));
                if (quarter_values.size() == 4) {
                    break;
                }
                next_quarter += quarter_size;
            }
            if (quarter_values.size() == 4) {
                break;
            }
        }
        quarter_values.push_back((float) bag_elements.back().at(0));
//        std::cout << "Done calculate_quarters" << std::endl;
    }

    void build_bag_from_counts(std::array<std::array<double, 2>, 256> &bag_counts) {
        bag_elements.clear();
        for (auto &bag_count: bag_counts) {
            if (bag_count.at(1) > 0) {
                bag_elements.push_back(bag_count);
            }
        }
        std::sort(bag_elements.begin(), bag_elements.end(), bag_entry_compare);
    }

    bool bag_and_check_range_for_bias_goal(std::array<std::array<double, 2>, 256> &bag_counts, int bias,
                                           double wide_target_range) {
        double bias_range = calculate_bias_range(bias);
        if (bias_range < wide_target_range) {
            return false;
        }
        bag(bag_counts);
        return (bias_fit == FIT_BIAS_FOR_100 && full_range() <= bias_range) ||
               (bias_fit == FIT_BIAS_FOR_80 && ten_to_90_range() <= bias_range) ||
               (bias_fit == FIT_BIAS_FOR_50 && q2_to_q3_range() <= bias_range);
    }
};

#endif //MICROML_TENSOR_STATS_HPP
