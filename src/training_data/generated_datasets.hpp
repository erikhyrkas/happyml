//
// Created by Erik Hyrkas on 12/9/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_GENERATED_DATASETS_HPP
#define HAPPYML_GENERATED_DATASETS_HPP

#include "../training_data/training_dataset.hpp"

using namespace std;

namespace happyml {

    class TestAdditionGeneratedDataSource : public TrainingDataSet {
    public:
        explicit TestAdditionGeneratedDataSource(size_t dataset_size) {
            this->datasetSize = dataset_size;
            this->currentOffset = 0;
            for (int i = 0; i < dataset_size; i++) {
                vector<float> given{(float) (i), (float) (i + 1)};
                vector<shared_ptr<BaseTensor>> next_given{make_shared<FullTensor>(given)};

                vector<float> result{(float) (i + i + 1)};
                vector<shared_ptr<BaseTensor>> expectation{make_shared<FullTensor>(result)};

                pairs.push_back(make_shared<TrainingPair>(next_given, expectation));
            }
        }

        size_t recordCount() override {
            return datasetSize;
        }

        void shuffle() override {
            // todo: likely can be optimized
            random_device rd;
            mt19937 g(rd());
            std::shuffle(pairs.begin(), pairs.end(), g);
        }

        void shuffle(size_t start_offset, size_t end_offset) override {
            // todo: likely can be optimized
            random_device rd;
            mt19937 g(rd());
            const size_t end = datasetSize - end_offset;
            // weird that I had to cast it down to an unsigned long
            // feels like a bug waiting to happen with a large data set.
            std::shuffle(pairs.begin() + (unsigned long) start_offset, pairs.end() - (unsigned long) end, g);
        }

        void restart() override {
            currentOffset = 0;
        }

        // populate a batch vector of vectors, reusing the structure. This is to save the time we'd otherwise use
        // to allocate.
        vector<shared_ptr<TrainingPair>> nextBatch(size_t batch_size) override {
            vector<shared_ptr<TrainingPair>> result;
            for (size_t batch_offset = 0; batch_offset < batch_size; batch_offset++) {
                if (currentOffset >= datasetSize) {
                    break;
                }
                shared_ptr<TrainingPair> next = nextRecord();
                if (next) {
                    result.push_back(next);
                }
            }
            return result;
        }

        shared_ptr<TrainingPair> nextRecord() override {
            if (currentOffset >= datasetSize) {
                return nullptr;
            }
            shared_ptr<TrainingPair> result = pairs.at(currentOffset);
            if (result) {
                currentOffset++;
            }
            return result;
        }

        vector<vector<size_t>> getGivenShapes() override {
            return vector<vector<size_t>>{{1, 2, 1}};
        }

        vector<vector<size_t>> getExpectedShapes() override {
            return vector<vector<size_t>>{{1, 1, 1}};
        }

    private:
        size_t datasetSize;
        vector<shared_ptr<TrainingPair>> pairs;
        size_t currentOffset;
    };
}
#endif //HAPPYML_GENERATED_DATASETS_HPP
