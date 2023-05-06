//
// Created by Erik Hyrkas on 4/22/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_COLUMN_METADATA_HPP
#define HAPPYML_COLUMN_METADATA_HPP

namespace happyml {
    struct BinaryColumnMetadata {
        char purpose; // 'I' (image), 'T' (text), 'N' (number), 'L' (label)
        // For scalars (numbers):
        // If the data flagged as is_standardized, it was standardized first then normalized.
        // We always normalize data if it is a scalar.
        bool is_standardized; // true if any tensor has a standard deviation > 1
        float mean; // used to unstandardize values
        float standard_deviation; // used to unstandardize values
        bool is_normalized; // should be true for any scalar if it isn't raw data
        float min_value; // used to denormalize values
        float max_value; // used to denormalize values
        uint64_t source_column_count;
        uint64_t rows;
        uint64_t columns;
        uint64_t channels;
        vector<string> ordered_labels;
    };

}

#endif //HAPPYML_COLUMN_METADATA_HPP
