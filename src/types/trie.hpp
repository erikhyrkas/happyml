//
// Created by Erik Hyrkas on 4/18/2023.
//

#ifndef HAPPYML_TRIE_HPP
#define HAPPYML_TRIE_HPP

#include <iostream>
#include <unordered_map>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>

using namespace std;

namespace happyml {
    struct TrieNode {
        std::unordered_map<char, std::shared_ptr<TrieNode>> children;
        std::string value;
        bool isEnd = false;
    };

    class Trie {
    public:
        std::shared_ptr<TrieNode> root;

        Trie() {
            root = std::make_shared<TrieNode>();
        }

        void insert(const std::string &word, std::string value) const {
            std::shared_ptr<TrieNode> node = root;
            for (char c: word) {
                if (node->children.find(c) == node->children.end()) {
                    node->children[c] = std::make_shared<TrieNode>();
                }
                node = node->children[c];
            }
            node->isEnd = true;
            node->value = std::move(value);
        }

        [[nodiscard]] bool search(const std::string &word) const {
            std::shared_ptr<TrieNode> node = root;
            for (char c: word) {
                if (node->children.find(c) == node->children.end()) {
                    return false;
                }
                node = node->children[c];
            }
            return node->isEnd;
        }

        [[nodiscard]] bool startsWith(const std::string &prefix) const {
            std::shared_ptr<TrieNode> node = root;
            for (char c: prefix) {
                if (node->children.find(c) == node->children.end()) {
                    return false;
                }
                node = node->children[c];
            }
            return true;
        }

        [[nodiscard]] std::string lookup(const std::string &word) const {
            std::shared_ptr<TrieNode> node = root;
            for (char c: word) {
                if (node->children.find(c) == node->children.end()) {
                    return "";
                }
                node = node->children[c];
            }
            if (node->isEnd) {
                return node->value;
            }
            return "";
        }

        [[nodiscard]] bool match(const std::string &word) const {
            std::shared_ptr<TrieNode> node = root;
            for (char c: word) {
                if (node->children.find(c) == node->children.end()) {
                    return false;
                }
                node = node->children[c];
            }
            return node != nullptr && node->isEnd;
        }

        // returns the longest match for a prefix
        [[nodiscard]] std::string complete(const std::string &prefix) const {
            std::string result = prefix;
            std::shared_ptr<TrieNode> node = root;

            // Traverse the trie with the given prefix
            for (char c: prefix) {
                if (node->children.find(c) == node->children.end()) {
                    return ""; // Prefix not found in the trie
                }
                node = node->children[c];
            }

            // depth-first search
            std::function<void(std::shared_ptr<TrieNode>, std::string)> dfs =
                    [&](const std::shared_ptr<TrieNode>& current_node, const std::string& current_prefix) {
                        if (current_node->isEnd) {
                            if (current_prefix.size() > result.size()) {
                                result = current_prefix;
                            }
                        }
                        for (const auto &child : current_node->children) {
                            dfs(child.second, current_prefix + child.first);
                        }
                    };

            dfs(node, result);

            return result;
        }
    };
}
#endif //HAPPYML_TRIE_HPP
