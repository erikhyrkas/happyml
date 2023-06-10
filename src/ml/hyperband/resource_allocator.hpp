//
// Created by Erik Hyrkas on 6/7/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_RESOURCE_ALLOCATOR_HPP
#define HAPPYML_RESOURCE_ALLOCATOR_HPP

#include <cmath>

namespace happyml {
    class ResourceAllocator {
    public:
        ResourceAllocator(int max_resources, int reduction_factor)
                : max_resources(max_resources), reduction_factor(reduction_factor) {
        }

        // Allocate resources for a given configuration
        [[nodiscard]] int allocateResources(int iteration) const {
            return max_resources / static_cast<int>(std::pow(reduction_factor, iteration));
        }

    private:
        int max_resources;         // Maximum available resources
        int reduction_factor;      // Resource reduction factor for successive halving
    };

}
#endif //HAPPYML_RESOURCE_ALLOCATOR_HPP
