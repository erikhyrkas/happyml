//
// Created by Erik Hyrkas on 3/26/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_BYTE_PAIR_ENCODER_HPP
#define HAPPYML_BYTE_PAIR_ENCODER_HPP

#include <unordered_map>
#include <set>
#include <queue>
#include <string>
#include <utility>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <execution>
#include "../util/data_util.hpp"
#include "../util/timers.hpp"

using namespace std;

namespace happyml {

    class BytePairEncoderModel {
    public:
        // Constructs a BytePairEncoderModel with optional show_progress and delimiter_code parameters.
        // name: can be used to differentiate trained models when saved and loaded
        // show_progress: If true, training progress will be printed; default is true.
        // delimiter_code: The delimiter code to be used; default is 256.
        explicit BytePairEncoderModel(string name = "default",
                                      bool show_progress = true,
                                      const uint16_t delimiter_code = 256)
                : show_progress_(show_progress),
                  name_(std::move(name)),
                  next_code_(delimiter_code + 1) {
            setDelimiterCode(delimiter_code);
        }

        string getName() {
            return name_;
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
            for (const auto &element: ordered_bpe_codes_) {
                uint16_t const next = std::max(find_max_16bit_value(element.first),
                                               find_max_16bit_value(element.second)) + 1;
                next_code_ = std::max(next, next_code_);
            }
        }

        // Configures the BPE model with bpe_codes and delimiter_code.
        // bpe_codes: An unordered_map of BPE codes.
        // delimiter_code: The delimiter code to be used.
        void configure(unordered_map<u16string, u16string> bpe_codes, uint16_t delimiter_code) {
            // set delimiter code first because it will update current_code
            setDelimiterCode(delimiter_code);
            // set bpe codes second because it will update current_code to max.
            setBpeCodes(bpe_codes);
        }

        vector<u16string> encode(const vector<string> &tokens) {
            vector<u16string> bpe_encoded_tokens;
            bpe_encoded_tokens.reserve(tokens.size());
            for (const string &token: tokens) {
                bpe_encoded_tokens.push_back(encode(token));
            }
            return bpe_encoded_tokens;
        }

        // Encodes a string using the BPE codes in the model.
        // token: The input string to be encoded.
        // Returns the encoded u16string.
        u16string encode(const string &token) {
            if (token.empty()) {
                return {};
            }
            u16string const text16bit(token.begin(), token.end());
            // looks simpler, but we need to optimize:
            // u16string encoded = delimiter_ + text16bit + delimiter_;
            u16string encoded;
            encoded.reserve(delimiter_.size() * 2 + text16bit.size());
            encoded += delimiter_;
            encoded += text16bit;
            encoded += delimiter_;
            shared_ptr<u16string> buffer = make_shared<u16string>();
            shared_ptr<u16string> encoded_ptr = make_shared<u16string>(std::move(encoded));

            for (auto replacement = ordered_bpe_codes_.rbegin();
                 replacement != ordered_bpe_codes_.rend(); ++replacement) {
                u16string_replace_all_to_buffer(*encoded_ptr, *buffer, replacement->first, replacement->second);
                buffer.swap(encoded_ptr);
            }
            return *encoded_ptr;
        }

        // Decodes an encoded u16string using the BPE codes in the model.
        // encoded: The input u16string to be decoded.
        // Returns the decoded string.
        string decode(const u16string &encoded) {
            if (encoded.empty()) {
                return {};
            }
            u16string decoded = encoded;
            u16string buffer;
            for (const auto &replacement: ordered_bpe_codes_) {
                u16string_replace_all_to_buffer(decoded, buffer, replacement.second, replacement.first);
                buffer.swap(decoded);
            }
            string result(decoded.begin() + (int) delimiter_.size(), decoded.end() - ((int) delimiter_.size()));
            return result;
        }

        // Trains the BPE model on a file. Efficient for a single large file.
        // WARNING: This implementation will overwrite any existing training
        // in the model.
        bool train_on_file(const string &filename) {
            if (show_progress_) {
                cout << "Training BPE on file \"" << filename << "\"" << endl;
                cout << "Building vocab..." << endl;
            }
            unordered_map<u16string, size_t> vocab;
            std::ifstream file(filename);
            if (file.is_open()) {
                buildVocab(file, show_progress_, vocab);
                file.close();
            } else {
                std::cerr << "Unable to open file: " << filename << std::endl;
                return false;
            }
            unordered_map<u16string, u16string> bpe_codes;
            train_on_vocab(-1,
                           0.0001,
                           2,
                           -1,
                           {},
                           0,
                           bpe_codes,
                           vocab);
            return true;
        }

        // This function trains the BPE model on an entire folder
        // of text files.
        bool train_on_folder(const string &folder) {
            if (show_progress_) {
                cout << "Training BPE on folder \"" << folder << "\"" << endl;
            }
            bool files_found = false;
            int file_count = 0;

            vector<string> file_paths;
            for (const auto &directory_entry: filesystem::directory_iterator(folder)) {
                if (directory_entry.is_regular_file()) {
                    file_paths.push_back(directory_entry.path().string());
                    files_found = true;
                }
            }
            unordered_map<u16string, size_t> vocab;
            for (const auto &file_path: file_paths) {
                file_count++;
                if (show_progress_) {
                    cout << "Loading byte pairs for file: " << file_path << " (" << file_count << "/"
                         << file_paths.size() << ")" << endl;
                }

                std::ifstream file(file_path);
                if (file.is_open()) {
                    buildVocab(file, false, vocab);
                    file.close();
                } else {
                    {
                        std::cerr << "Unable to open file: " << file_path << std::endl;
                        return false;
                    }
                }
            }

            if (show_progress_) {
                cout << "BPE Training..."
                     << endl;
            }
            unordered_map<u16string, u16string> bpe_codes;
            train_on_vocab(-1,
                           0.0001,
                           2,
                           -1,
                           {},
                           0,
                           bpe_codes,
                           vocab);
            return files_found;
        }


        // Trains a BPE model from a vector of strings.
        // NOTE: leveraging early_stopping_patience will dramatically slow training, however it will
        //  allow you to have greater control over how far you take merges. Unless you are really
        //  passionate about validating while training, I'd consider using the validate_compression_rate()
        //  after training is complete.
        //
        // data: A vector of input strings for training. (See: string_to_tokens() and load_file_to_tokens() for how to build.)
        // early_stopping_patience: The number of iterations without improvement before stopping; default is 15.
        // early_stopping_improvement_minimum: The minimum improvement required for resetting the no-improvement counter; default is 0.00001.
        // min_frequency: The minimum frequency for a pair to be considered; default is 2.
        // num_merges: The maximum number of merges to perform; default is -1 (no limit).
        void train(const vector<string> &data,
                   int early_stopping_patience = -1,
                   double early_stopping_improvement_minimum = 0.00001,
                   size_t min_frequency = 2,
                   int num_merges = -1) {
            ElapsedTimer totalTimer;

            if (show_progress_) {
                cout << "Byte Pair Encoder Model Training started: " << std::fixed << std::setprecision(2);
            }

            long total_validation_length = 0;
            vector<string> train_data, validation_data;
            if (early_stopping_patience >= 0) {
                splitData(data, train_data, validation_data);
                if (validation_data.empty()) {
                    validation_data = train_data;
                }
                for (const auto &text: validation_data) {
                    total_validation_length += (long) text.length();
                }
            } else {
                train_data = data;
            }

            unordered_map<u16string, u16string> bpe_codes;

            if (!ordered_bpe_codes_.empty()) {
                if (show_progress_) {
                    cout << "Current Code Staring at: " << next_code_ << endl;
                    cout << "Loading existing bpe codes..." << endl;
                }
                for (const auto &element: ordered_bpe_codes_) {
                    bpe_codes[element.first] = element.second;
                    uint16_t const next =
                            std::max(find_max_16bit_value(element.first), find_max_16bit_value(element.second)) + 1;
                    next_code_ = std::max(next, next_code_);
                }
                if (show_progress_) {
                    cout << "Next Code now: " << next_code_ << endl;
                    cout << "Finished Loading existing bpe codes." << endl;
                }

            }

            if (show_progress_) {
                cout << "Building Vocab..." << endl;
            }
            unordered_map<u16string, size_t> vocab = buildVocab(train_data);
            train_on_vocab(early_stopping_patience, early_stopping_improvement_minimum, min_frequency, num_merges,
                           validation_data,
                           total_validation_length, bpe_codes, vocab);
            if (show_progress_) {
                int64_t const elapsed = totalTimer.getMilliseconds();
                cout << endl << "Finished BPE training in ";
                if (elapsed < 2000) {
                    cout << elapsed << " milliseconds." << endl;
                } else if (elapsed < 120000) {
                    cout << (elapsed / 1000) << " seconds." << endl;
                } else {
                    cout << (elapsed / 60000) << " minutes." << endl;
                }
            }
        }

        void train_on_vocab(int early_stopping_patience,
                            double early_stopping_improvement_minimum,
                            size_t min_frequency,
                            int num_merges,
                            const vector<string> &validation_data,
                            long total_validation_length,
                            unordered_map<u16string, u16string> &bpe_codes,
                            unordered_map<u16string, size_t> &vocab) {

            const uint16_t max_code = 0x7FFE; // 0x7FFF (32,767) is reserved for the padding delimiter
            double best_validation_score = INFINITY;
            size_t merge_count = 0;
            int no_improvement_counter = 0;
            ElapsedTimer mergeTimer;
            while (!vocab.empty() && (merge_count < 1 || merge_count < num_merges)) {
                auto most_frequent = findMostFrequentPair(vocab, min_frequency);
                if (most_frequent.second == 0) {
                    break;
                }
                if (show_progress_) {
                    cout << "Merge count: " << merge_count << " Largest Code: " << next_code_;
                    if (best_validation_score != INFINITY) {
                        cout << " Best Compression: " << best_validation_score;
                    }
                    auto mergeTime = mergeTimer.getMilliseconds();
                    if (mergeTime > 120000) {
                        const auto min = mergeTime / 60000;
                        const auto sec = (mergeTime % 60000) / 1000;
                        printf("%5zd m %zd s ", min, sec);
                    } else if (mergeTime > 2000) {
                        const auto sec = (mergeTime / 1000);
                        printf("%5zd s ", sec);
                    } else {
                        printf("%5zd ms ", mergeTime);
                    }
                    cout << endl;
                }
                if (early_stopping_patience >= 0) {
                    double const current_validation_score = validate_compression_rate(validation_data,
                                                                                      bpe_codes,
                                                                                      delimiter_code_,
                                                                                      total_validation_length);
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

                u16string const current_code_string(1, next_code_);
                auto most_frequent_string = most_frequent.first;

                bpe_codes[most_frequent_string] = current_code_string;
                updateCodeForMostFrequentPair(vocab, most_frequent, current_code_string);
                mergePairs(vocab, most_frequent_string, current_code_string);

                next_code_++;
                merge_count++;

                // 32,767 codes is the limit of the current implementation
                if (next_code_ >= max_code) {
                    if (show_progress_) {
                        cout << "Exiting early because Current Code hit the limit of " << max_code << "." << endl;
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

        [[nodiscard]] uint16_t getLargestCode() const {
            return next_code_;
        }

        // Merges a pair of characters with another pair in the vocabulary.
        // vocab: The vocabulary to be updated.
        // most_frequent_string: The most frequent pair of characters.
        // new_code: The new code for the most frequent pair.
        static void mergePairs(unordered_map<u16string, size_t> &vocab,
                               const u16string &most_frequent_string,
                               const u16string &new_code) {
            unordered_map<u16string, size_t> new_vocab;
            new_vocab.reserve(vocab.size());

            u16string buffer;
            for (auto &entry: vocab) {
                auto &original_first = entry.first;
                auto &original_second = entry.second;
                size_t const found_pos = original_first.find(most_frequent_string);
                if (found_pos != u16string::npos) {
                    u16string new_pair = original_first;
                    // replace all will update the new_pair variable.
                    u16string_replace_all_to_buffer(new_pair, buffer, most_frequent_string, new_code);
                    new_pair.swap(buffer);
                    auto existing_entry = new_vocab.find(new_pair);
                    if (existing_entry != new_vocab.end()) {
                        existing_entry->second++;
                    } else {
                        new_vocab.emplace_hint(new_vocab.end(), new_pair, 1);
                    }
                } else {
                    new_vocab.emplace_hint(new_vocab.end(), original_first, original_second);
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
            new_vocab.reserve(vocab.size());

            u16string buffer;
            for (auto &entry: vocab) {
                auto original_first = entry.first;
                auto original_second = entry.second;
                if (original_first.find(most_frequent_string) != u16string::npos) {
                    original_second -= most_frequent_count;
                    u16string_replace_all_to_buffer(original_first, buffer, most_frequent_string, new_code);
                    buffer.swap(original_first);
                }
                if (original_second > 0) {
                    new_vocab.emplace_hint(new_vocab.end(), std::move(original_first), original_second);
                }
            }

            new_vocab.emplace_hint(new_vocab.end(), new_code, most_frequent_count);

            vocab.swap(new_vocab);
        }


        // Builds a vocabulary from a vector of input strings.
        // tokens: A vector of input strings for building the vocabulary.
        // Returns an unordered_map of character pairs and their frequencies.
        unordered_map<u16string, size_t> buildVocab(const vector<string> &tokens) {
            unordered_map<u16string, size_t> vocab;
            for (const string &token: tokens) {
                // Originally, I just added the delimiter before and after the line, but
                // to support calling train() multiple times, I decided to encode()
                // this should have the same effect on the first pass, but result
                // in a more compact vocabulary the second call to train()
                // It LOOKS like it works, which is why I'm leaving it in.
                u16string line16 = encode(token);
                for (size_t i = 0; i < line16.size() - 1; ++i) {
                    u16string const pair = u16string(1, line16[i]) + u16string(1, line16[i + 1]);
                    // If the character pair is already in the vocab unordered_map, increment its frequency; otherwise, add it to the vocab unordered_map with a frequency of 1.
                    auto exiting_entry = vocab.find(
                            pair);
                    if (exiting_entry != vocab.end()) {
                        exiting_entry->second++;
                    } else {
                        vocab.emplace_hint(vocab.end(), pair, 1);
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
        static double validate_compression_rate(const vector<string> &validation_data,
                                                const unordered_map<u16string, u16string> &bpe_codes,
                                                uint16_t delimiter,
                                                const long total_validation_length) {
            if (total_validation_length < 1) {
                return 0.0;
            }
            BytePairEncoderModel bpe;
            bpe.configure(bpe_codes, delimiter);
            return bpe.validate_compression_rate(validation_data, total_validation_length);
        }

        double validate_compression_rate(const vector<string> &validation_data, long total_validation_length = 0) {
            if (total_validation_length < 1) {
                for (const auto &text: validation_data) {
                    total_validation_length += (long) text.length();
                }
                if (total_validation_length < 1) {
                    return 0.0;
                }
            }

            long total_encoded_length = 0;
            for (const auto &i: validation_data) {
                auto encoded_string = encode(i);
                total_encoded_length += static_cast<long>(encoded_string.length());
            }

            return total_encoded_length / (double) total_validation_length;
        }


        // reminder: overwrite will remove all files in that folder because it assumes this is the only
        // model in that folder. It is not a good practice with happyml for two different models to
        // share output folders.
        bool save(const string &modelFolderPath, const string &knowledgeLabel, bool overwrite = true) {
            string const fullKnowledgePath = initialize_knowledge_path_directory(modelFolderPath, knowledgeLabel,
                                                                                 overwrite);
            string const filePath = fullKnowledgePath + "/" + name_ + ".bpe";
            std::ofstream file(filePath, std::ios::binary);

            if (!file.is_open()) {
                return false;
            }

            file.write(reinterpret_cast<const char *>(&delimiter_code_), sizeof(delimiter_code_));

            for (const auto &code_pair: ordered_bpe_codes_) {
                for (const auto &str: {code_pair.first, code_pair.second}) {
                    auto str_length = static_cast<uint16_t>(str.size());
                    file.write(reinterpret_cast<const char *>(&str_length), sizeof(str_length));
                    file.write(reinterpret_cast<const char *>(str.data()),
                               static_cast<streamsize>(str_length * sizeof(char16_t)));
                }
            }

            file.close();
            return true;
        }

        bool load(const string &modelFolderPath, const string &knowledgeLabel) {
            string const path = modelFolderPath + "/" + knowledgeLabel + "/" + name_ + ".bpe";
            std::ifstream file(path, std::ios::binary);

            if (!file.is_open()) {
                return false;
            }

            uint16_t delimiter_code = 0;
            file.read(reinterpret_cast<char *>(&delimiter_code), sizeof(delimiter_code));
            setDelimiterCode(delimiter_code);

            while (!file.eof()) {
                pair<u16string, u16string> code_pair;
                for (auto &str: {&code_pair.first, &code_pair.second}) {
                    uint16_t str_length = 0;
                    file.read(reinterpret_cast<char *>(&str_length), sizeof(str_length));

                    if (file.eof()) {
                        break;
                    }

                    str->resize(str_length);
                    file.read(const_cast<char *>(reinterpret_cast<const char *>(str->data())),
                              static_cast<streamsize>(str_length * sizeof(char16_t)));
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
        uint16_t next_code_;
        bool show_progress_; // do we print text while training?
        string name_;

        // Sets the delimiter code and delimiter string.
        // delimiter_code: The delimiter code to be used.
        void setDelimiterCode(const uint16_t delimiter_code) {
            delimiter_code_ = delimiter_code;
            delimiter_ = u16string(1, delimiter_code);
            next_code_ = delimiter_code_ + 1;
        }

        void buildVocab(std::ifstream &file, bool show_progress, unordered_map<u16string, size_t> &vocab) {
            std::string token;
            char last_char = 0;
            char buffer[256 * 1024];
            std::streamsize bytesRead;
            size_t total_bytes_read = 0;
            streampos currentPos = file.tellg();
            file.seekg(0, ios::end);
            size_t file_size = file.tellg() / (1024 * 1024);
            file.seekg(currentPos);
            while ((bytesRead = file.read(buffer, sizeof(buffer)).gcount()) > 0) {
                total_bytes_read += bytesRead;
                if (show_progress) {
                    std::cout << "Read " << total_bytes_read / (1024 * 1024) << " of " << file_size
                              << " megabytes of byte pairs\r"
                              << std::flush;
                }
                for (int i = 0; i < bytesRead; ++i) {
                    char c = buffer[i];
                    append_character(c, last_char, token, [&vocab, this](const std::string &new_token) {
                        u16string line16 = this->encode(new_token);
                        for (size_t i = 0; i < line16.size() - 1; ++i) {
                            u16string const pair = u16string(1, line16[i]) + u16string(1, line16[i + 1]);
                            auto existing_entry = vocab.find(pair);
                            if (existing_entry != vocab.end()) {
                                existing_entry->second++;
                            } else {
                                vocab.emplace_hint(vocab.end(), pair, 1);
                            }
                        }
                    });
                }
            }

            // Process any remaining data after the buffer read
            if (!token.empty()) {
                u16string line16 = encode(token);
                for (size_t i = 0; i < line16.size() - 1; ++i) {
                    u16string const pair = u16string(1, line16[i]) + u16string(1, line16[i + 1]);
                    auto existing_entry = vocab.find(pair);
                    if (existing_entry != vocab.end()) {
                        existing_entry->second++;
                    } else {
                        vocab.emplace_hint(vocab.end(), pair, 1);
                    }
                }
            }
            if (show_progress) {
                std::cout << endl << "Finish." << std::endl;
            }
        }
    };


}

#endif //HAPPYML_BYTE_PAIR_ENCODER_HPP
