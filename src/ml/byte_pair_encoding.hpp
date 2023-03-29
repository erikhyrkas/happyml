//
// Created by Erik Hyrkas on 3/26/2023.
//

#ifndef HAPPYML_BYTE_PAIR_ENCODING_HPP
#define HAPPYML_BYTE_PAIR_ENCODING_HPP

#include <unordered_map>
#include <set>
#include <queue>
#include <string>
#include <utility>
#include <sstream>
#include <regex>
#include <iomanip>
#include <filesystem>
#include "../util/data_util.hpp"

using namespace std;

namespace happyml {

    class BytePairEncodingModel {
    public:
        // Constructs a BytePairEncodingModel with optional show_progress and delimiter_code parameters.
        // show_progress: If true, training progress will be printed; default is true.
        // delimiter_code: The delimiter code to be used; default is 256.
        explicit BytePairEncodingModel(bool show_progress = true, const uint16_t delimiter_code = 256) {
            setDelimiterCode(delimiter_code);
            show_progress_ = show_progress;
        }

        // Sets the delimiter code and delimiter string.
        // delimiter_code: The delimiter code to be used.
        void setDelimiterCode(const uint16_t delimiter_code) {
            delimiter_code_ = delimiter_code;
            delimiter_ = u16string(1, delimiter_code);
        }

        // Sets the BPE codes from an unordered_map of BPE codes.
        // bpe_codes: An unordered_map of BPE codes to be set in the model.
        void setBpeCodes(unordered_map<u16string, u16string> &bpe_codes) {
            vector<pair<u16string, u16string>> ordered_bpe_codes(bpe_codes.begin(), bpe_codes.end());
            // Sort the BPE codes in reverse order of their values (the order in which they were learned)
            sort(ordered_bpe_codes.begin(), ordered_bpe_codes.end(),
                 [](const pair<u16string, u16string> &a, const pair<u16string, u16string> &b) {
                     return a.second > b.second;
                 });
            ordered_bpe_codes_.swap(ordered_bpe_codes);
        }

        // Configures the BPE model with bpe_codes and delimiter_code.
        // bpe_codes: An unordered_map of BPE codes.
        // delimiter_code: The delimiter code to be used.
        void configure(unordered_map<u16string, u16string> bpe_codes, uint16_t delimiter_code) {
            setDelimiterCode(delimiter_code);
            setBpeCodes(bpe_codes);
        }

        // Encodes a string using the BPE codes in the model.
        // text: The input string to be encoded.
        // Returns the encoded u16string.
        u16string encode(const string &text) {
            if (text.empty()) {
                return {};
            }
            u16string text16bit(text.begin(), text.end());
            u16string encoded = delimiter_ + text16bit + delimiter_;
            for (auto replacement = ordered_bpe_codes_.rbegin();
                 replacement != ordered_bpe_codes_.rend(); ++replacement) {
                u16string_replace_all(encoded, replacement->first, replacement->second);
            }
            return encoded;
        }

        // Decodes an encoded u16string using the BPE codes in the model.
        // encoded: The input u16string to be decoded.
        // Returns the decoded string.
        string decode(const u16string &encoded) {
            if (encoded.empty()) {
                return {};
            }
            u16string decoded = encoded;
            for (const auto &replacement: ordered_bpe_codes_) {
                u16string_replace_all(decoded, replacement.second, replacement.first);
            }
            string result(decoded.begin() + (int) delimiter_.size(), decoded.end() - ((int) delimiter_.size()));
            return result;
        }

        // Trains a BPE model from a vector of strings.
        // data: A vector of input strings for training. (See: string_to_tokens() and load_file_to_tokens() for how to build.)
        // early_stopping_patience: The number of iterations without improvement before stopping; default is 15.
        // early_stopping_improvement_minimum: The minimum improvement required for resetting the no-improvement counter; default is 0.00001.
        // min_frequency: The minimum frequency for a pair to be considered; default is 2.
        // num_merges: The maximum number of merges to perform; default is -1 (no limit).
        void train(const vector<string> &data,
                   int early_stopping_patience = 15,
                   double early_stopping_improvement_minimum = 0.00001,
                   size_t min_frequency = 2,
                   int num_merges = -1) {

            if (show_progress_) {
                cout << "Byte Pair Encoding Model Training started: " << std::fixed << std::setprecision(2);
            }
            vector<string> train_data, validation_data;
            if (early_stopping_patience >= 0) {
                splitData(data, train_data, validation_data);
                if (validation_data.empty()) {
                    validation_data = train_data;
                }
            } else {
                train_data = data;
            }

            uint16_t current_code = delimiter_code_ + 1;
            double best_validation_score = INFINITY;

            unordered_map<u16string, u16string> bpe_codes;

            if (!ordered_bpe_codes_.empty()) {
                if (show_progress_) {
                    cout << "Current Code Staring at: " << current_code << endl;
                    cout << "Loading existing bpe codes..." << endl;
                }
                for (const auto &element: ordered_bpe_codes_) {
                    bpe_codes[element.first] = element.second;
                    uint16_t next =
                            std::max(find_max_16bit_value(element.first), find_max_16bit_value(element.second)) + 1;
                    current_code = std::max(next, current_code);
                }
                if (show_progress_) {
                    cout << "Current Code now: " << current_code << endl;
                    cout << "Finished Loading existing bpe codes." << endl;
                }
            }

            unordered_map<u16string, size_t> vocab = buildVocab(train_data);
            size_t merge_count = 0;
            int no_improvement_counter = 0;
            while (!vocab.empty() && (merge_count < 1 || merge_count < num_merges)) {
                pair<u16string, size_t> most_frequent = findMostFrequentPair(vocab, min_frequency);
                if (most_frequent.second == 0) {
                    break;
                }
                if (show_progress_) {
                    cout << "Merge count: " << merge_count << " Current Code: " << current_code;
                    cout << " BPE Pairs: " << bpe_codes.size();
                    if (best_validation_score != INFINITY) {
                        cout << " Best Compression: " << best_validation_score;
                    }
                    cout << endl;
                }
                if (early_stopping_patience >= 0) {
                    double current_validation_score = compression_rate(validation_data, bpe_codes, delimiter_code_);
                    if (current_validation_score < (best_validation_score - early_stopping_improvement_minimum)) {
                        best_validation_score = current_validation_score;
                        no_improvement_counter = 0;
                    } else {
                        no_improvement_counter++;
                        if (no_improvement_counter > early_stopping_patience) {
                            break;
                        }
                    }
                }

                u16string current_code_string(1, current_code);
                auto most_frequent_string = most_frequent.first;

                bpe_codes[most_frequent_string] = current_code_string;
                updateCodeForMostFrequentPair(vocab, most_frequent, current_code_string);
                mergePairs(vocab, most_frequent_string, current_code_string);

                current_code++;
                merge_count++;

                if (current_code > 59000) {
                    if (show_progress_) {
                        cout << "Exiting early because Current Code hit the soft limit of 59,000." << endl;
                    }
                    break;
                }
            }
            vector<pair<u16string, u16string>> ordered_bpe_codes(bpe_codes.begin(), bpe_codes.end());
            sort(ordered_bpe_codes.begin(), ordered_bpe_codes.end(),
                 [](const pair<u16string, u16string> &a, const pair<u16string, u16string> &b) {
                     return a.second > b.second;
                 });
            ordered_bpe_codes_.swap(ordered_bpe_codes);
        }

        // Returns the BPE codes in the model as a vector of pairs.
        vector<pair<u16string, u16string>> getBpeCodes() {
            return ordered_bpe_codes_;
        }

        // Returns the delimiter used to separate character pairs as a u16string.
        u16string getDelimiter() {
            return delimiter_;
        }

        // Merges a pair of characters with another pair in the vocabulary.
        // vocab: The vocabulary to be updated.
        // most_frequent_string: The most frequent pair of characters.
        // new_code: The new code for the most frequent pair.
        static void mergePairs(unordered_map<u16string, size_t> &vocab,
                               const u16string &most_frequent_string,
                               const u16string &new_code) {
            unordered_map<u16string, size_t> new_vocab;
            for (auto &entry: vocab) {
                auto original_first = entry.first;
                auto original_second = entry.second;
                size_t found_pos = original_first.find(most_frequent_string);
                if (found_pos != u16string::npos) {
                    u16string new_pair = original_first;
                    u16string_replace_all(new_pair, most_frequent_string, new_code);
                    original_second--;
                    if (new_vocab.find(new_pair) != new_vocab.end()) {
                        new_vocab[new_pair]++;
                    } else {
                        new_vocab[new_pair] = 1;
                    }
                }

                if (original_second > 0) {
                    new_vocab[original_first] = original_second;
                }
            }

            vocab.swap(new_vocab);
        }

        // Updates the code for the most frequent pair in the vocabulary.
        // vocab: The vocabulary to be updated.
        // most_frequent: The most frequent pair and its frequency as a pair.
        // new_code: The new code for the most frequent pair.
        static void updateCodeForMostFrequentPair(unordered_map<u16string, size_t> &vocab,
                                                  const pair<u16string, size_t> &most_frequent,
                                                  const u16string &new_code) {
            auto most_frequent_string = most_frequent.first;
            auto most_frequent_count = most_frequent.second;
            vocab.erase(most_frequent_string);
            unordered_map<u16string, size_t> new_vocab;
            for (auto &entry: vocab) {
                auto original_first = entry.first;
                auto original_second = entry.second;
                if (original_first.find(most_frequent_string) != u16string::npos) {
                    original_second -= most_frequent_count;
                    u16string_replace_all(original_first, most_frequent_string, new_code);
                }
                if (original_second > 0) {
                    new_vocab[original_first] = original_second;
                }
            }
            new_vocab[new_code] = most_frequent_count;
            vocab.swap(new_vocab);
        }


        // Builds a vocabulary from a vector of input strings.
        // data: A vector of input strings for building the vocabulary.
        // Returns an unordered_map of character pairs and their frequencies.
        unordered_map<u16string, size_t> buildVocab(const vector<string> &data) {
            unordered_map<u16string, size_t> vocab;
            for (const string &line: data) {
                // Originally, I just added the delimiter before and after the line, but
                // to support calling train() multiple times, I decided to encode()
                // this should have the same effect on the first pass, but result
                // in a more compact vocabulary the second call to train()
                // It LOOKS like it works, which is why I'm leaving it in.
                u16string line16 = encode(line);
                for (size_t i = 0; i < line16.size() - 1; ++i) {
                    u16string pair = u16string(1, line16[i]) + u16string(1, line16[i + 1]);
                    // If the character pair is already in the vocab unordered_map, increment its frequency; otherwise, add it to the vocab unordered_map with a frequency of 1.
                    if (vocab.find(pair) != vocab.end()) {
                        vocab[pair]++;
                    } else {
                        vocab[pair] = 1;
                    }
                }
            }
            return vocab;
        }

        // Finds the most frequent pair of characters in the vocabulary.
        // vocab: The input vocabulary.
        // min_frequency: The minimum frequency for a pair to be considered.
        // Returns a pair of the most frequent u16string and its frequency.
        static pair<u16string, size_t> findMostFrequentPair(const unordered_map<u16string, size_t> &vocab,
                                                            size_t min_frequency) {
            pair<u16string, size_t> most_frequent(u"", 0);
            for (const auto &entry: vocab) {
                if (entry.first.size() > 1 && entry.second > most_frequent.second && entry.second >= min_frequency) {
                    most_frequent = entry;
                }
            }
            return most_frequent;
        }

        // Computes the compression rate for a set of validation data using the BPE codes and delimiter.
        // validation_data: A vector of input strings for validation.
        // bpe_codes: The BPE codes to use.
        // delimiter: The delimiter code to use.
        // Returns the compression rate as a double.
        static double compression_rate(const vector<string> &validation_data,
                                       const unordered_map<u16string, u16string> &bpe_codes,
                                       uint16_t delimiter) {
            happyml::BytePairEncodingModel bpe;
            bpe.configure(bpe_codes, delimiter);

            double total_original_length = 0.0;
            double total_encoded_length = 0.0;

            for (const auto &text: validation_data) {
                total_original_length += (double) text.length();
                u16string encoded = bpe.encode(text);
                total_encoded_length += (double) encoded.length();
            }
            if (total_original_length < 1) {
                return 0.0;
            }
            return total_encoded_length / total_original_length;
        }


        bool save(const string &modelFolderPath, const string &knowledgeLabel, bool overwrite = true) {
            string fullKnowledgePath = buildKnowledgePath(modelFolderPath, knowledgeLabel, overwrite);
            string filePath = fullKnowledgePath + "/model.bpe";
            std::ofstream file(filePath, std::ios::binary);

            if (!file.is_open()) {
                return false;
            }

            file.write(reinterpret_cast<const char *>(&delimiter_code_), sizeof(delimiter_code_));

            for (const auto &code_pair: ordered_bpe_codes_) {
                for (const auto &str: {code_pair.first, code_pair.second}) {
                    auto str_length = static_cast<uint16_t>(str.size());
                    file.write(reinterpret_cast<const char *>(&str_length), sizeof(str_length));
                    file.write(reinterpret_cast<const char *>(str.data()), str_length * sizeof(char16_t));
                }
            }

            file.close();
            return true;
        }

        bool load(const string &modelFolderPath, const string &knowledgeLabel) {
            string path = modelFolderPath + "/" + knowledgeLabel + "/model.bpe";
            std::ifstream file(path, std::ios::binary);

            if (!file.is_open()) {
                return false;
            }

            uint16_t delimiter_code;
            file.read(reinterpret_cast<char *>(&delimiter_code), sizeof(delimiter_code));
            setDelimiterCode(delimiter_code);

            while (!file.eof()) {
                pair<u16string, u16string> code_pair;
                for (auto &str: {&code_pair.first, &code_pair.second}) {
                    uint16_t str_length;
                    file.read(reinterpret_cast<char *>(&str_length), sizeof(str_length));

                    if (file.eof()) {
                        break;
                    }

                    str->resize(str_length);
                    file.read(reinterpret_cast<char *>(str->data()), str_length * sizeof(char16_t));
                }

                if (!file.eof()) {
                    ordered_bpe_codes_.push_back(std::move(code_pair));
                }
            }

            file.close();
            return true;
        }

    private:
        vector<pair<u16string, u16string>> ordered_bpe_codes_; // the ordered list of byte-pair encodings
        uint16_t delimiter_code_{}; // a value we can do math with and our base starting point.
        u16string delimiter_; // saves us from recomputing the u16 string
        bool show_progress_; // do we print out text while training?
    };
}

#endif //HAPPYML_BYTE_PAIR_ENCODING_HPP
