//
// Created by Erik Hyrkas on 5/8/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_ENCODER_DECODER_BUILDER_HPP
#define HAPPYML_ENCODER_DECODER_BUILDER_HPP

#include "../training_data/data_decoder.hpp"
#include "../training_data/data_encoder.hpp"
#include "column_metadata.hpp"
#include "file_reader.hpp"

namespace happyml {

    shared_ptr<RawDecoder> build_decoder(bool raw, const shared_ptr<BinaryColumnMetadata> &metadata) {
        shared_ptr<RawDecoder> decoder;
        if (raw) {
            decoder = make_shared<RawDecoder>();
        } else {
            auto purpose = metadata->purpose;
            // purpose: 'I' (image), 'T' (text), 'N' (number), 'L' (label)
            if ('L' == purpose) {
                auto ordered_labels = metadata->ordered_labels;
                decoder = make_shared<BestTextCategoryDecoder>(ordered_labels);
            } else if ('N' == purpose) {
                decoder = make_shared<RawDecoder>(metadata->is_normalized,
                                                  metadata->is_standardized,
                                                  metadata->min_value,
                                                  metadata->max_value,
                                                  metadata->mean,
                                                  metadata->standard_deviation);
            } else if( 'I' == purpose) {
                decoder = make_shared<ImageDecoder>();
            } else {
                decoder = make_shared<RawDecoder>();
            }
        }
        return decoder;
    }

    vector<shared_ptr<RawDecoder>> build_given_decoders(bool raw, BinaryDatasetReader &reader) {
        vector<shared_ptr<RawDecoder >> given_decoders;
        size_t given_column_count = reader.get_given_column_count();
        for (size_t i = 0; i < given_column_count; i++) {
            const shared_ptr<BinaryColumnMetadata> &metadata = reader.get_given_metadata(i);
            shared_ptr<RawDecoder> decoder = build_decoder(raw, metadata);
            given_decoders.push_back(decoder);
        }
        return given_decoders;
    }

    vector<shared_ptr<HappyMLVariantEncoder>> build_given_encoders(BinaryDatasetReader &reader) {
        vector<shared_ptr<HappyMLVariantEncoder >> encoders;
        size_t given_column_count = reader.get_given_column_count();
        for (size_t i = 0; i < given_column_count; i++) {
            const shared_ptr<BinaryColumnMetadata> &metadata = reader.get_given_metadata(i);
            shared_ptr<HappyMLVariantEncoder> encoder = make_shared<HappyMLVariantEncoder>(metadata);
            encoders.push_back(encoder);
        }
        return encoders;
    }

    vector<shared_ptr<RawDecoder>> build_expected_decoders(bool raw, BinaryDatasetReader &reader) {
        vector<shared_ptr<RawDecoder >> expected_decoders;
        size_t expected_column_count = reader.get_expected_column_count();
        for (size_t i = 0; i < expected_column_count; i++) {
            const shared_ptr<BinaryColumnMetadata> &metadata = reader.get_expected_metadata(i);
            shared_ptr<RawDecoder> decoder = build_decoder(raw, metadata);
            expected_decoders.push_back(decoder);
        }
        return expected_decoders;
    }

    vector<shared_ptr<RawDecoder>> build_expected_decoders(bool raw, shared_ptr<BinaryDataSet> &dataset) {
        vector<shared_ptr<RawDecoder >> expected_decoders;
        auto expected_metadata = dataset->getExpectedMetadata();
        size_t expected_column_count = expected_metadata.size();
        for (size_t i = 0; i < expected_column_count; i++) {
            shared_ptr<RawDecoder> decoder = build_decoder(raw, expected_metadata[i]);
            expected_decoders.push_back(decoder);
        }
        return expected_decoders;
    }
}

#endif //HAPPYML_ENCODER_DECODER_BUILDER_HPP
