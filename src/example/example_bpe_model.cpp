//
// Created by Erik Hyrkas on 4/1/2023.
//
#include <memory>
#include "../ml/happyml_dsl.hpp"
#include "../training_data/data_decoder.hpp"

using namespace std;
using namespace happyml;

int main() {
    try {
        vector<std::string> tokens;
        // block allows us to free up some memory after we've parsed the file.
        {
            // example dataset from https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots
            // See the README.md in this folder for details on preparing it.
            auto reader = DelimitedTextFileReader("..\\data\\wiki_movie_plots_deduped.csv", ',', true);
            std::string current_token;
            char previous_character = 0;

            while (reader.hasNext()) {
                // we've stripped out the csv encoding, but now we want to join the columns together as text.
                auto line = join_strings(reader.nextRecord(), "\n");
                for (const auto &c: line) {
                    append_character(c, previous_character, current_token, tokens);
                }
            }
            if (!current_token.empty()) {
                tokens.push_back(current_token);
            }
        }
        BytePairEncoderModel bpe;
        bpe.train(tokens);
        if (!bpe.save("../repo/", "bpe_example")) {
            cerr << "Error Saving model!" << endl;
        }

        const vector<string> validationData = sampleData(tokens, 0.01);
        double const compression = bpe.validate_compression_rate(validationData);
        cout << "Final compression rate: " << compression << endl;

        auto baseString = "This is a fun string of testing.";
        auto testString = bpe.encode(baseString);
        auto decodedTextString = bpe.decode(testString);
        cout << "Example: " << baseString << endl;
        cout << "Encoded: " << string(testString.begin(), testString.end()) << endl;
        cout << "Decoded: " << decodedTextString << endl;

    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}