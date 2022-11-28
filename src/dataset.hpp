//
// Created by Erik Hyrkas on 11/2/2022.
//

#ifndef MICROML_DATASET_HPP
#define MICROML_DATASET_HPP

#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include "tensor.hpp"
#include "data.hpp"
#include "dataencoder.hpp"

namespace microml {

    class TrainingDataSet {
    public:
        virtual size_t record_count() = 0;

        virtual void shuffle() = 0;

        virtual void shuffle(size_t start_offset, size_t end_offset) = 0;

        virtual void restart() = 0;

        virtual std::vector<std::shared_ptr<TrainingPair>> next_batch(size_t batch_size) = 0;

        virtual std::shared_ptr<TrainingPair> next_record() = 0;

        virtual std::vector<std::vector<size_t>> getGivenShapes() = 0;

        virtual vector<size_t> getGivenShape() {
            return getGivenShapes()[0];
        }

        virtual std::vector<std::vector<size_t>> getExpectedShapes() = 0;

        virtual vector<size_t> getExpectedShape() {
            return getExpectedShapes()[0];
        }
    };

    class EmptyTrainingDataSet : public TrainingDataSet {
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

    class PartialTrainingDataSet : public TrainingDataSet {
    public:
        //
        PartialTrainingDataSet(const std::shared_ptr<TrainingDataSet> &dataSource, size_t first_record_offset,
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
        std::shared_ptr<TrainingDataSet> dataSource;
        size_t first_record_offset;
        size_t last_record_offset;
        size_t count;
        size_t current_offset;
    };




// SimpleTsvDataSource
// Loads a single tsv into memory as a data source. This is not a scalable option for large data sets,
// but fine for testing a single file that is relatively small and fits in memory.
    class SimpleTsvTrainingDataSet : public TrainingDataSet {
    public:
        SimpleTsvTrainingDataSet(const std::string &filename, const std::shared_ptr<TrainingDataInputEncoder> &encoder) {

        }

    private:
        std::vector<std::vector<BaseTensor>> rows;
    };

}

#endif //MICROML_DATASET_HPP
