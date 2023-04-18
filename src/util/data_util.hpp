//
// Created by Erik Hyrkas on 3/27/2023.
//

#ifndef HAPPYML_DATA_UTIL_HPP
#define HAPPYML_DATA_UTIL_HPP

#include <filesystem>
#include <string>
#include <numeric>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <cctype>
#include <limits>
#include <random>
#include <set>
#include <functional>

using namespace std;

namespace happyml {

    std::string join_strings(const std::vector<std::string> &strings, const std::string &delimiter = "") {
        std::stringstream ss;
        for (size_t i = 0; i < strings.size(); ++i) {
            ss << strings[i];
            if (i < strings.size() - 1) {
                ss << delimiter;
            }
        }
        return ss.str();
    }

    string initialize_knowledge_path_directory(const string &modelFolderPath, const string &knowledgeLabel, bool overwrite) {
        string fullKnowledgePath = modelFolderPath + "/" + knowledgeLabel;
        if (filesystem::is_directory(fullKnowledgePath)) {
            if (!overwrite) {
                auto canonicalFullKnowledgePath = filesystem::canonical(fullKnowledgePath);
                cerr << "Knowledge path " << canonicalFullKnowledgePath
                     << " already existed, attempting to save to the new location: ";
                auto ms = std::to_string(chrono::duration_cast<chrono::milliseconds>(
                        chrono::system_clock::now().time_since_epoch()).count());
                canonicalFullKnowledgePath += "_" + ms;
                cerr << canonicalFullKnowledgePath << endl;
                fullKnowledgePath = canonicalFullKnowledgePath.generic_string();
            } else {
                filesystem::remove_all(fullKnowledgePath);
            }
        }
        filesystem::create_directories(fullKnowledgePath);
        return fullKnowledgePath;
    }


    void append_character(char current_character,
                          char &previous_character,
                          std::string &current_token,
                          const std::function<void(const std::string&)>& process_token) {
        if (current_character == '\r') {
            return;
        }

        if (std::isspace(static_cast<unsigned char>(current_character))) {
            if (current_character != previous_character) {
                if (!current_token.empty()) {
                    process_token(current_token);
                    current_token.clear();
                }
                current_token.push_back(current_character);
            }
        } else {
            if (isprint(static_cast<unsigned char>(current_character))
                && !isalnum(static_cast<unsigned char>(current_character))
                && (current_character != '.' || !isdigit(static_cast<unsigned char>(previous_character)))) {
                if (!current_token.empty()) {
                    process_token(current_token);
                    current_token.clear();
                }
                process_token(std::string(1, current_character));
            } else if (!std::isprint(static_cast<unsigned char>(current_character))) {
                current_token.push_back((char)254);
            } else {
                current_token.push_back(current_character);
            }
        }
        previous_character = current_character;
    }

//    void append_character(char current_character, char &previous_character,
//                          std::string &current_token, std::vector<std::string> &tokens) {
//        auto process_token = [&tokens](const std::string& new_token) {
//            tokens.push_back(new_token);
//        };
//        append_character(current_character, previous_character, current_token, process_token);
//    }

    void append_character(char c, char &last_char, std::string &token, std::vector<std::string> &tokens) {
        if (c == '\r') return;

        if (std::isspace(static_cast<unsigned char>(c))) {
            if (c != last_char) {
                if (!token.empty()) {
                    tokens.push_back(token);
                    token.clear();
                }
                token.push_back(c);
            }
        } else {
            if (isprint(static_cast<unsigned char>(c))
                && !isalnum(static_cast<unsigned char>(c))
                && (c != '.' || !isdigit(static_cast<unsigned char>(last_char)))) {
                if (!token.empty()) {
                    tokens.push_back(token);
                    token.clear();
                }
                tokens.push_back(std::string(1, c));
            } else if (!std::isprint(static_cast<unsigned char>(c))) {
                token.push_back((char) 254);
            } else {
                token.push_back(c);
            }
        }
        last_char = c;
    }

    std::vector<std::string> string_to_tokens(const std::string &text) {
        std::vector<std::string> tokens;
        std::string token;
        char last_char = 0;

        for (const auto &c: text) {
            append_character(c, last_char, token, tokens);
        }

        if (!token.empty()) {
            tokens.push_back(token);
        }

        return tokens;
    }

    std::vector<std::string> load_file_to_tokens(const std::string &filename) {
        std::ifstream file(filename);
        std::vector<std::string> tokens;
        std::string token;
        char last_char = 0;

        char buffer[32*1024];
        while (file.read(buffer, sizeof(buffer))) {
            for (int i = 0; i < file.gcount(); ++i) {
                char c = buffer[i];
                append_character(c, last_char, token, tokens);
            }
        }
        for (int i = 0; i < file.gcount(); ++i) {
            char c = buffer[i];
            append_character(c, last_char, token, tokens);
        }

        if (!token.empty()) {
            tokens.push_back(token);
        }

        file.close();
        return tokens;
    }

    std::vector<std::string> load_file_to_lines(const std::string &filename) {
        std::vector<std::string> lines;
        std::ifstream file(filename);

        if (!file.is_open()) {
            // handle error opening file
            return lines;
        }

        std::string line;
        while (std::getline(file, line)) {
            // remove any non-printable ascii characters
            for (char &c: line) {
                if (c < 32 || c > 126) {
                    c = (char) 254;
                }
            }
            // add newline character back to the end of the line
            line.push_back('\n');
            lines.push_back(line);
        }

        file.close();
        return lines;
    }

    vector<string> sampleData(const vector<string> &data,
                    float validation_ratio = 0.2) {
        auto validation_size = static_cast<size_t>(data.size() * validation_ratio);
        size_t const train_size = data.size() - validation_size;

        std::random_device rd;
        std::mt19937 g(rd());
        vector<size_t> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g);

        vector<string> validation_data;
        validation_data.reserve(validation_size);

        for (size_t i = train_size; i < data.size(); ++i) {
            validation_data.push_back(data[indices[i]]);
        }
        return std::move(validation_data);
    }

    void splitData(const vector<string> &data,
                   vector<string> &train_data,
                   vector<string> &validation_data,
                   float validation_ratio = 0.2) {

        auto validation_size = static_cast<size_t>(data.size() * validation_ratio);
        size_t const train_size = data.size() - validation_size;

        std::random_device rd;
        std::mt19937 g(rd());
        vector<size_t> indices(data.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g);

        train_data.reserve(train_size);
        validation_data.reserve(validation_size);

        for (size_t i = 0; i < train_size; ++i) {
            train_data.push_back(data[indices[i]]);
        }

        for (size_t i = train_size; i < data.size(); ++i) {
            validation_data.push_back(data[indices[i]]);
        }
    }

    void u16string_replace_all_to_buffer(const u16string &string_to_update,
                                         u16string &result,
                                         const u16string &substring_to_find,
                                         const u16string &substring_replacement) {
        const size_t find_length = substring_to_find.length();
        const size_t replace_length = substring_replacement.length();

        // If the substring to find is empty, simply return without modifying the input string
        if (find_length == 0) {
            return;
        }

        result.clear();
        if (replace_length > find_length) {
            result.reserve(string_to_update.length() * 2); // ensure that the resulting string has enough capacity
        } else {
            result.reserve(string_to_update.length());
        }

        size_t start_pos = 0;
        const size_t original_length = string_to_update.length();
        while (start_pos < original_length) {
            size_t const found_pos = string_to_update.find(substring_to_find, start_pos);
            if (found_pos == string::npos) {
                // if no more occurrences of the substring are found, append the rest of the string and exit the loop
                result.append(string_to_update, start_pos, original_length - start_pos);
                break;
            }
            // append the portion of the string before the found substring
            result.append(string_to_update, start_pos, found_pos - start_pos);
            // append the replacement string
            if (replace_length > 0) {
                result.append(substring_replacement);
            }
            // update the start position to the end of the found substring
            start_pos = found_pos + find_length;
        }
    }

    void u16string_replace_all(u16string &string_to_update,
                               const u16string &substring_to_find,
                               const u16string &substring_replacement) {
        u16string result;
        u16string_replace_all_to_buffer(string_to_update, result, substring_to_find, substring_replacement);
        string_to_update = std::move(result); // update the original string with the new one
    }


    void string_replace_all(string &string_to_update,
                            const string &substring_to_find,
                            const string &substring_replacement) {
        const size_t find_length = substring_to_find.length();
        const size_t replace_length = substring_replacement.length();

        // If the substring to find is empty, simply return without modifying the input string
        if (find_length == 0) {
            return;
        }

        string result;
        if (replace_length > find_length) {
            result.reserve(string_to_update.length() * 2); // ensure that the resulting string has enough capacity
        } else {
            result.reserve(string_to_update.length());
        }

        result.reserve(string_to_update.length()); // ensure that the resulting string has enough capacity

        size_t start_pos = 0;
        const size_t original_length = string_to_update.length();
        while (start_pos < original_length) {
            size_t const found_pos = string_to_update.find(substring_to_find, start_pos);
            if (found_pos == string::npos) {
                // if no more occurrences of the substring are found, append the rest of the string and exit the loop
                result.append(string_to_update, start_pos, original_length - start_pos);
                break;
            }
            // append the portion of the string before the found substring
            result.append(string_to_update, start_pos, found_pos - start_pos);
            // append the replacement string
            if (replace_length > 0) {
                result.append(substring_replacement);
            }
            // update the start position to the end of the found substring
            start_pos = found_pos + find_length;
        }
        string_to_update = std::move(result); // update the original string with the new one
    }

    uint16_t find_max_16bit_value(const std::u16string &str) {
        uint16_t max_value = std::numeric_limits<uint16_t>::min();
        for (auto c: str) {
            auto value = static_cast<uint16_t>(c);
            if (value > max_value && value < 0x7FFF) {
                max_value = value;
            }
        }
        return max_value;
    }


}



#endif //HAPPYML_DATA_UTIL_HPP
