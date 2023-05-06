//
// Created by Erik Hyrkas on 4/22/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_TEXT_FILE_SORTER_HPP
#define HAPPYML_TEXT_FILE_SORTER_HPP

#include <algorithm>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <filesystem>

namespace happyml {

    class FileSorter {
    public:
        // this can sort and remove duplicate lines
        // while removing duplicate lines is helpful, this doesn't help
        // when there are many given inputs with different expected outputs
        // however, we can take care of those later when we make a binary data set,
        // this simplified duplicate detection for the binary data writer,
        // at least in the case where the given is in the file before the expected
        // values.
        static bool sort(const std::string &file_name, const std::string &result_file_name, bool has_header = true, int chunk_size = 10000, bool delete_duplicates = true) {
            auto parent_path = std::filesystem::path(file_name).parent_path().string();

            std::ifstream input(file_name);

            if (!input.is_open()) {
                std::cerr << "Error opening input file." << std::endl;
                return false;
            }

            std::vector<std::string> lines;
            std::string line, header;

            if (has_header) {
                std::getline(input, header);
            }

            size_t chunk_counter = 0;
            std::vector<std::string> chunk_paths;
            while (std::getline(input, line)) {
                if (!line.empty()) {
                    lines.push_back(line);
                }

                if (lines.size() >= chunk_size) {
                    sort_chunk(lines);

                    std::string chunk_path = parent_path + "/chunk_" + std::to_string(chunk_counter) + ".txt";
                    std::ofstream chunk_output(chunk_path);
                    chunk_paths.push_back(chunk_path);
                    for (const auto &sorted_line: lines) {
                        chunk_output << sorted_line << '\n';
                    }
                    chunk_output.close();

                    ++chunk_counter;
                    lines.clear();
                }
            }

            if (!lines.empty()) {
                sort_chunk(lines);

                std::string chunk_path = parent_path + "/chunk_" + std::to_string(chunk_counter) + ".txt";
                std::ofstream chunk_output(chunk_path);
                chunk_paths.push_back(chunk_path);
                for (const auto &sorted_line: lines) {
                    chunk_output << sorted_line << '\n';
                }
                chunk_output.close();
                ++chunk_counter;
            }

            input.close();
            merge_chunks(parent_path, result_file_name, chunk_counter, header, has_header, delete_duplicates);
            for (const auto &chunk_path: chunk_paths) {
                std::filesystem::remove(chunk_path);
            }
            return true;
        }

    private:
        static void sort_chunk(std::vector<std::string> &lines) {
            std::sort(lines.begin(), lines.end());
        }

        static void merge_chunks(const std::string &parent_path, const std::string &output_file, size_t num_chunks, const std::string &header, bool has_header, bool delete_duplicates) {
            auto line_comparator = [](const auto &a, const auto &b) { return a.first > b.first; };
            std::priority_queue<std::pair<std::string, std::ifstream *>, std::vector<std::pair<std::string, std::ifstream *>>, decltype(line_comparator)> min_heap(line_comparator);

            std::vector<std::ifstream> chunk_files(num_chunks);

            for (size_t i = 0; i < num_chunks; ++i) {
                std::string chunk_path = parent_path + "/chunk_" + std::to_string(i) + ".txt";

                chunk_files[i].open(chunk_path);

                if (!chunk_files[i]) {
                    break;
                }

                std::string line;
                std::getline(chunk_files[i], line);

                if (chunk_files[i]) {
                    min_heap.emplace(line, &chunk_files[i]);
                }
            }

            std::string previous_line;
            std::ofstream output(output_file);

            if (has_header) {
                output << header << '\n';
            }

//            if (!min_heap.empty()) {
//                previous_line = min_heap.top().first;
//                auto chunk_file = min_heap.top().second;
//                min_heap.pop();
//
//                output << previous_line << '\n';
//
//                std::getline(*chunk_file, previous_line);
//                if (*chunk_file) {
//                    min_heap.emplace(previous_line, chunk_file);
//                }
//            }

            bool first_record = true;
            while (!min_heap.empty()) {
                auto [line, chunk_file] = min_heap.top();
                min_heap.pop();

                if (first_record || !(delete_duplicates && line == previous_line) && !line.empty()) {
                    first_record = false;
                    output << line << '\n';
                    previous_line = line;
                }
                std::getline(*chunk_file, line);
                if (*chunk_file) {
                    min_heap.emplace(line, chunk_file);
                }
            }
            output.close();
            for(auto &chunk_file: chunk_files) {
                chunk_file.close();
            }
        }
    };
}
#endif //HAPPYML_TEXT_FILE_SORTER_HPP
