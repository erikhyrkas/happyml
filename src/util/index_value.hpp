//
// Created by Erik Hyrkas on 1/13/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_INDEX_VALUE_HPP
#define HAPPYML_INDEX_VALUE_HPP

class IndexValue {
public:
    IndexValue(size_t index, float value) : index(index), value(value) {
    }

    bool operator<(const IndexValue &indexValue) const {
        return value < indexValue.value;
    }

    [[nodiscard]] size_t getIndex() const {
        return index;
    }

    [[nodiscard]] float getValue() const {
        return value;
    }

private:
    size_t index;
    float value;
};

#endif //HAPPYML_INDEX_VALUE_HPP
