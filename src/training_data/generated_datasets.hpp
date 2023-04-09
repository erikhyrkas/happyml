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

        void restart() override {
            currentOffset = 0;
        }

        shared_ptr<TrainingPair> nextRecord() override {
            if (currentOffset >= datasetSize) {
                return nullptr;
            }
            auto shuffled_offset = shuffler_ != nullptr? shuffler_->getShuffledIndex(currentOffset):currentOffset;
            shared_ptr<TrainingPair> result = pairs.at(shuffled_offset);
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
