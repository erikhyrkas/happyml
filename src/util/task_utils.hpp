//
// Created by Erik Hyrkas on 5/14/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_TASK_UTILS_HPP
#define HAPPYML_TASK_UTILS_HPP

#include "dataset_utils.hpp"

namespace happyml {

    bool create_label_task(const string &task_name, const string &goal, const string &dataset_name, const string &dataset_file_path, const string &task_folder_path) {
        cout << "Creating label task " << task_name << " with goal " << goal << " using dataset " << dataset_name << endl;

        auto metadata = read_column_metadata(dataset_file_path);
        save_column_metadata(metadata.first, metadata.second, task_folder_path);

        //auto dataset = BinaryDatasetReader(dataset_file_path);
        return true;
    }

    bool create_happyml_task(const string &task_type, const string &task_name, const string &goal, const string &dataset_name, const string &dataset_file_path, const string &task_folder_path) {
        if (task_type == "label") {
            return create_label_task(task_name, goal, dataset_name, dataset_file_path, task_folder_path);
        }
        cout << "Unknown task type " << task_type << endl;
        return false;
    }

}
#endif //HAPPYML_TASK_UTILS_HPP
