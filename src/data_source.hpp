//
// Created by Erik Hyrkas on 11/2/2022.
//

#ifndef MICROML_DATA_SOURCE_HPP
#define MICROML_DATA_SOURCE_HPP

#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include "tensor.hpp"
#include "data.hpp"

namespace microml {

    class BaseMicromlDataSource {
    public:
        virtual size_t record_count() = 0;

        virtual void shuffle() = 0;

        virtual void shuffle(size_t start_offset, size_t end_offset) = 0;

        virtual void restart() = 0;

        virtual std::vector<std::shared_ptr<TrainingPair>> next_batch(size_t batch_size) = 0;

        virtual std::shared_ptr<TrainingPair> next_record() = 0;

        virtual std::vector<std::vector<size_t>> getGivenShapes() = 0;

        virtual std::vector<std::vector<size_t>> getExpectedShapes() = 0;
    };

    class EmptyDataSource : public BaseMicromlDataSource {
        size_t record_count() override {
            return 0;
        }

        void shuffle() override {}

        void shuffle(size_t start_offset, size_t end_offset) override {}

        void restart() override {}

        std::vector<std::shared_ptr<TrainingPair>> next_batch(size_t batch_size) override {
            std::vector<std::shared_ptr<TrainingPair>> result;
            return result;
        }

        std::shared_ptr<TrainingPair> next_record() override {
            return nullptr;
        }

        std::vector<std::vector<size_t>> getGivenShapes() override {
            return std::vector<std::vector<size_t>>{{1, 1, 1}};
        }

        std::vector<std::vector<size_t>> getExpectedShapes() override {
            return std::vector<std::vector<size_t>>{{1, 1, 1}};
        }
    };

    class DataSourcePortion : public BaseMicromlDataSource {
    public:
        //
        DataSourcePortion(const std::shared_ptr<BaseMicromlDataSource> &dataSource, size_t first_record_offset,
                          size_t last_record_offset) {
            this->dataSource = dataSource;
            this->first_record_offset = first_record_offset;
            this->last_record_offset = last_record_offset;
            this->count = last_record_offset - first_record_offset;
            this->current_offset = first_record_offset;
            if (first_record_offset > last_record_offset) {
                throw std::exception("First offset must be before last offset");
            }
            if (last_record_offset >= dataSource->record_count()) {
                throw std::exception("Record offset out of bounds");
            }
        }

        size_t record_count() override {
            return count;
        }

        void shuffle(size_t start_offset, size_t end_offset) override {
            const size_t new_first = first_record_offset + start_offset;
            const size_t new_end = first_record_offset + end_offset;
            if (new_first > last_record_offset || new_end > last_record_offset) {
                throw std::exception("shuffle offset out of range");
            }
            restart();
            dataSource->shuffle(new_first, new_end);
        }

        void shuffle() override {
            shuffle(first_record_offset, last_record_offset);
        }

        void restart() override {
            current_offset = first_record_offset;
        }

        std::vector<std::shared_ptr<TrainingPair>> next_batch(size_t batch_size) override {
            std::vector<std::shared_ptr<TrainingPair>> result;
            const size_t target_last_offset = current_offset + batch_size;
            if (target_last_offset <= last_record_offset) {
                for (size_t batch_offset = 0; batch_offset < batch_size; batch_offset++) {
                    result.push_back(next_record());
                }
            }
            return result;
        }

        std::shared_ptr<TrainingPair> next_record() override {
            std::shared_ptr<TrainingPair> result = dataSource->next_record();
            if (result) {
                current_offset++;
            }
            return result;
        }

    private:
        std::shared_ptr<BaseMicromlDataSource> dataSource;
        size_t first_record_offset;
        size_t last_record_offset;
        size_t count;
        size_t current_offset;
    };

    class TestAdditionGeneratedDataSource : public BaseMicromlDataSource {
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

    class TestXorDataSource : public BaseMicromlDataSource {
    public:
        explicit TestXorDataSource() {
            this->dataset_size = 4;
            this->current_offset = 0;

            pairs.push_back(std::make_shared<TrainingPair>(std::vector<float>{0.f, 0.f}, std::vector<float>{0.f}));
            pairs.push_back(std::make_shared<TrainingPair>(std::vector<float>{0.f, 1.f}, std::vector<float>{1.f}));
            pairs.push_back(std::make_shared<TrainingPair>(std::vector<float>{1.f, 0.f}, std::vector<float>{1.f}));
            pairs.push_back(std::make_shared<TrainingPair>(std::vector<float>{1.f, 1.f}, std::vector<float>{0.f}));
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


    class BaseMicromlDataEncoder {
        virtual void encode(std::string &text, BaseTensor &tensor) = 0;
    };

// take in comma delimited numbers and convert to a tensor
    class TextToNumbersEncoder : public BaseMicromlDataEncoder {

        void encode(std::string &text, BaseTensor &tensor) override {

        }
    };

// TODO: this should be a data set
// SimpleTsvDataSource
// Loads a single tsv into memory as a data source. This is not a scalable option for large data sets,
// but fine for testing a single file that is relatively small and fits in memory.
    class SimpleTsvDataSource : public BaseMicromlDataSource {
    public:
        SimpleTsvDataSource(const std::string &filename, const std::shared_ptr<BaseMicromlDataEncoder> &encoder) {

        }

    private:
        std::vector<std::vector<BaseTensor>> rows;
    };

}

#endif //MICROML_DATA_SOURCE_HPP
