//
// Created by Erik Hyrkas on 12/9/2022.
//

#ifndef MICROML_GENERATED_DATASETS_HPP
#define MICROML_GENERATED_DATASETS_HPP

#include "../training_data/training_dataset.hpp"

using namespace std;

namespace microml {

    class TestAdditionGeneratedDataSource : public TrainingDataSet {
    public:
        explicit TestAdditionGeneratedDataSource(size_t dataset_size) {
            this->dataset_size = dataset_size;
            this->current_offset = 0;
            for (int i = 0; i < dataset_size; i++) {
                std::vector<float> given{(float) (i), (float) (i + 1)};
                std::vector<std::shared_ptr<BaseTensor>> next_given{std::make_shared<FullTensor>(given)};

                std::vector<float> result{(float) (i + i + 1)};
                std::vector<std::shared_ptr<BaseTensor>> expectation{std::make_shared<FullTensor>(result)};

                pairs.push_back(std::make_shared<TrainingPair>(next_given, expectation));
            }
        }

        size_t record_count() override {
            return dataset_size;
        }

        void shuffle() override {
            // todo: likely can be optimized
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(pairs.begin(), pairs.end(), g);
        }

        void shuffle(size_t start_offset, size_t end_offset) override {
            // todo: likely can be optimized
            std::random_device rd;
            std::mt19937 g(rd());
            const size_t end = dataset_size - end_offset;
            // weird that I had to cast it down to an unsigned long
            // feels like a bug waiting to happen with a large data set.
            std::shuffle(pairs.begin() + (unsigned long) start_offset, pairs.end() - (unsigned long) end, g);
        }

        void restart() override {
            current_offset = 0;
        }

        // populate a batch vector of vectors, reusing the structure. This is to save the time we'd otherwise use
        // to allocate.
        std::vector<std::shared_ptr<TrainingPair>> next_batch(size_t batch_size) override {
            std::vector<std::shared_ptr<TrainingPair>> result;
            for (size_t batch_offset = 0; batch_offset < batch_size; batch_offset++) {
                if (current_offset >= dataset_size) {
                    break;
                }
                std::shared_ptr<TrainingPair> next = next_record();
                if (next) {
                    result.push_back(next);
                }
            }
            return result;
        }

        std::shared_ptr<TrainingPair> next_record() override {
            if (current_offset >= dataset_size) {
                return nullptr;
            }
            std::shared_ptr<TrainingPair> result = pairs.at(current_offset);
            if (result) {
                current_offset++;
            }
            return result;
        }

        std::vector<std::vector<size_t>> getGivenShapes() override {
            return std::vector<std::vector<size_t>>{{1, 2, 1}};
        }

        std::vector<std::vector<size_t>> getExpectedShapes() override {
            return std::vector<std::vector<size_t>>{{1, 1, 1}};
        }

    private:
        size_t dataset_size;
        std::vector<std::shared_ptr<TrainingPair>> pairs;
        size_t current_offset;
    };
}
#endif //MICROML_GENERATED_DATASETS_HPP
