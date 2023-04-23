//
// Created by Erik Hyrkas on 4/22/2023.
//

#ifndef HAPPYML_LRU_CACHE_H
#define HAPPYML_LRU_CACHE_H
#include <list>
#include <unordered_map>
#include <stdexcept>

namespace happyml {

    template<typename Key, typename Value>
    class LruCache {
    public:
        explicit LruCache(size_t capacity=100000) : capacity_(capacity) {}

        bool contains(const Key& key) {
            return cache_map_.find(key) != cache_map_.end();
        }

        void insert(const Key& key, const Value& value) {
            if (cache_map_.size() >= capacity_) {
                // Remove the least recently used entry
                Key least_recent = lru_list_.back();
                cache_map_.erase(least_recent);
                lru_list_.pop_back();
            }

            // Add the new entry to the front of the list
            lru_list_.push_front(key);
            cache_map_[key] = {value, lru_list_.begin()};
        }

    private:
        size_t capacity_;
        std::list<Key> lru_list_;
        std::unordered_map<Key, std::pair<Value, typename std::list<Key>::iterator>> cache_map_;
    };
}
#endif //HAPPYML_LRU_CACHE_H
