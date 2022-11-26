//
// Created by Erik Hyrkas on 11/2/2022.
//

#ifndef MICROML_DATASET_HPP
#define MICROML_DATASET_HPP

#include <iostream>
#include <utility>

namespace microml {

    class MicromlDataset {
    public:
        explicit MicromlDataset(std::string filename) {
            this->filename = std::move(filename);
        }

    private:
        std::string filename;
    };

}
#endif //MICROML_DATASET_HPP
