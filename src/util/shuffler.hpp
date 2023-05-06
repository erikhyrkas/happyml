//
// Created by Erik Hyrkas on 4/9/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_SHUFFLER_HPP
#define HAPPYML_SHUFFLER_HPP
#include <vector>
#include <random>
#include <algorithm>

namespace happyml {
    // Exists for in-place shuffling, which is ideal for datasets backed by files.
    // What's more, it can be shared between two datasets that are inputs to the
    // same model so that the matching records stay shuffled together.
    // The shuffler isn't shuffled until you call shuffle(), this is so that
    // unnecessary shuffles don't happen, since the very first step of the
    // training loop is to shuffle.
    class Shuffler {
    public:
        Shuffler(size_t size) : shuffled_elements_(size) {
            for (size_t i = 0; i < size; i++) {
                shuffled_elements_[i] = i;
            }
        }

        size_t getShuffledIndex(size_t index) const {
            return shuffled_elements_[index];
        }

        void shuffle() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(shuffled_elements_.begin(), shuffled_elements_.end(), gen);
        }

        size_t getSize() {
            return shuffled_elements_.size();
        }

    private:
        std::vector<size_t> shuffled_elements_;
    };
}
#endif //HAPPYML_SHUFFLER_HPP
