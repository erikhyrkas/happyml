//
// Created by Erik Hyrkas on 3/25/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_EXECUTION_CONTEXT_HPP
#define HAPPYML_EXECUTION_CONTEXT_HPP

#include <string>
#include <memory>
#include "../ml/byte_pair_encoder.hpp"

#define DEFAULT_HAPPYML_REPO_PATH "../happyml_repo/"
#define DEFAULT_HAPPYML_DATASETS_PATH "../happyml_repo/datasets/"
#define DEFAULT_HAPPYML_TASKS_PATH "../happyml_repo/tasks/"
#define DEFAULT_HAPPYML_SCRIPTS_PATH "../happyml_repo/scripts/"

using namespace std;

namespace happyml {
    class ExecutionContext {
        // TODO: we'll eventually store state here.
        // Examples might include:
        // * external configuration
        // * session variables
        // * debugging/troubleshooting information
    public:
        [[nodiscard]] const shared_ptr<BytePairEncoderModel> &getBpeEncoder() const {
            return bpe_encoder;
        }

        void setBpeEncoder(const shared_ptr<BytePairEncoderModel> &bpeEncoder) {
            bpe_encoder = bpeEncoder;
        }

        static string get_dataset_path(const string &dataset_name) {
            return DEFAULT_HAPPYML_DATASETS_PATH + dataset_name;
        }

        static string get_task_folder_path(const string &task_name) {
            return DEFAULT_HAPPYML_TASKS_PATH + task_name;
        }

        static string get_base_task_folder_path() {
            return DEFAULT_HAPPYML_TASKS_PATH;
        }

        static bool dataset_exists(const string &dataset_name) {
            string bin_path = get_dataset_path(dataset_name) + "/dataset.bin";
            ifstream f(bin_path.c_str());
            if (f.good()) {
                f.close();
                return true;
            }
            return false;
        }

    private:
        shared_ptr<BytePairEncoderModel> bpe_encoder;
    };

    class ExecutionResult {
    public:
        explicit ExecutionResult(bool exit = false, bool success = true, const string &message = "") {
            this->success = success;
            this->message = message;
            this->exit = exit;
        }

        [[nodiscard]] bool exitRequested() const {
            return exit;
        }

        [[nodiscard]] bool isSuccessful() const {
            return success;
        }

        string getMessage() {
            return message;
        }

    private:
        bool success;
        bool exit;
        string message;
    };

    class ExecutableStatement {
    public:
        virtual shared_ptr<ExecutionResult> execute(const shared_ptr<ExecutionContext> &context) = 0;
    };

    class ParseResult {
    public:
        explicit ParseResult(const string &message = "Failure", bool success = false) {
            this->success = success;
            this->message = message;
        }

        explicit ParseResult(const shared_ptr<ExecutableStatement> &node, const string &message = "Success",
                             bool success = true) {
            this->success = success;
            this->message = message;
            this->executable = node;
        }

        [[nodiscard]] bool isSuccessful() const {
            return success;
        }

        string getMessage() {
            return message;
        }

        shared_ptr<ExecutableStatement> getExecutable() {
            return executable;
        }

    private:
        bool success;
        string message;
        shared_ptr<ExecutableStatement> executable;
    };
}

#endif //HAPPYML_EXECUTION_CONTEXT_HPP
