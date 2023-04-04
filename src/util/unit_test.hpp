//
// Created by Erik Hyrkas on 10/25/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_UNIT_TEST_HPP
#define HAPPYML_UNIT_TEST_HPP

#include <iostream>
#include <string>
#include <unordered_map>
#include <random>
#include <vector>

enum PseudoWordType {
    Noun, Verb, Adjective, Adverb, Punctuation, Quote, Comma, Space, Conjunction
};

struct PseudoWord {
    std::string value{};
    PseudoWordType type{};
};

struct PseudoSentencePattern {
    std::vector<PseudoWordType> pattern;
};

std::string capitalize_first_letter(const std::string &word) {
    std::string capitalized_word = word;
    capitalized_word[0] = (char) toupper(capitalized_word[0]);
    return capitalized_word;
}

std::string random_string(int min_len, int max_len) {
    static const std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<> len_dist(min_len, max_len);
    std::uniform_int_distribution<> index_dist(0, (int) alphabet.size() - 1);
    std::random_device rd;
    std::mt19937 gen(rd());
    int length = len_dist(gen);
    std::string result;
    for (int i = 0; i < length; i++) {
        result.push_back(alphabet[index_dist(gen)]);
    }
    return result;
}

std::vector<PseudoWord> generate_pseudo_vocabulary(int vocab_size, PseudoWordType type) {
    std::vector<PseudoWord> vocabulary;
    vocabulary.reserve(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        vocabulary.push_back({random_string(1, 6), type});
    }
    return vocabulary;
}

std::string generate_pseudo_corpus(int sentence_count, int vocab_size) {
    std::vector<PseudoWord> nouns = generate_pseudo_vocabulary(std::max((int) (vocab_size * 0.40), 1), Noun);
    std::vector<PseudoWord> verbs = generate_pseudo_vocabulary(std::max((int) (vocab_size * 0.35), 1), Verb);
    std::vector<PseudoWord> adjectives = generate_pseudo_vocabulary(std::max((int) (vocab_size * 0.1), 1), Adjective);
    std::vector<PseudoWord> adverbs = generate_pseudo_vocabulary(std::max((int) (vocab_size * 0.14), 1), Adverb);
    std::vector<PseudoWord> conjunctions = generate_pseudo_vocabulary(std::max((int) (vocab_size * 0.01), 1),
                                                                      Conjunction);
    std::vector<PseudoWord> punctuation = {{".", Punctuation},
                                           {"?", Punctuation},
                                           {"!", Punctuation}};
    std::vector<PseudoWord> quotes = {{"\"", Quote}};
    std::vector<PseudoWord> commas = {{",", Comma}};
    std::vector<PseudoWord> spaces = {{" ", Space}};
    std::vector<PseudoSentencePattern> patterns = {
            {{Adjective, Space, Noun,        Space, Adverb,      Space,       Verb,        Space,       Adjective, Space,       Noun,      Punctuation, Space}},
            {{Noun,      Space, Verb,        Space, Adjective,   Space,       Noun,        Punctuation, Space}},
            {{Noun,      Space, Adverb,      Space, Verb,        Space,       Noun,        Punctuation, Space}},
            {{Noun,      Space, Verb,        Space, Noun,        Punctuation, Space}},
            {{Quote,     Noun,  Space,       Verb,  Space,       Comma,       Quote,       Space,       Noun,      Space,       Verb,      Punctuation, Space}},
            {{Quote,     Noun,  Space,       Verb,  Space,       Comma,       Quote,       Space,       Noun,      Space,       Verb,      Space,       Adverb, Punctuation, Space}},
            {{Adjective, Space, Noun,        Space, Verb,        Space,       Noun,        Punctuation, Space}},
            {{Noun,      Space, Verb,        Space, Adjective,   Space,       Noun,        Space,       Adverb,    Punctuation, Space}},
            {{Adverb,    Space, Noun,        Space, Verb,        Space,       Noun,        Punctuation, Space}},
            {{Noun,      Space, Verb,        Space, Adverb,      Space,       Adjective,   Space,       Noun,      Punctuation, Space}},
            {{Adjective, Space, Noun,        Space, Verb,        Space,       Adverb,      Space,       Adjective, Space,       Noun,      Space,       Adverb, Punctuation, Space}},
            {{Quote,     Noun,  Space,       Verb,  Space,       Adjective,   Space,       Noun,        Space,     Comma,       Quote,     Space,       Noun,   Space,       Verb, Space, Adjective, Space, Noun, Punctuation, Space}},
            {{Adverb,    Space, Verb,        Space, Noun,        Space,       Adjective,   Space,       Noun,      Punctuation, Space}},
            {{Adjective, Space, Noun,        Space, Adverb,      Space,       Verb,        Space,       Noun,      Space,       Adjective, Space,       Noun,   Punctuation, Space}},
            {{Noun,      Space, Adjective,   Space, Verb,        Space,       Adverb,      Space,       Noun,      Punctuation, Space}},
            {{Adjective, Space, Noun,        Space, Verb,        Space,       Adverb,      Space,       Noun,      Space,       Verb,      Space,       Noun,   Punctuation, Space}},
            {{Noun,      Space, Conjunction, Space, Noun,        Space,       Verb,        Punctuation, Space}},
            {{Verb,      Space, Noun,        Space, Conjunction, Space,       Verb,        Punctuation, Space}},
            {{Noun,      Space, Verb,        Space, Conjunction, Space,       Noun,        Punctuation, Space}},
            {{Adjective, Space, Noun,        Space, Conjunction, Space,       Adjective,   Space,       Noun,      Punctuation, Space}},
            {{Noun,      Space, Verb,        Space, Adjective,   Space,       Conjunction, Space,       Adjective, Space,       Noun,      Punctuation, Space}},
            {{Adverb,    Space, Verb,        Space, Conjunction, Space,       Adverb,      Space,       Verb,      Punctuation, Space}},
            {{Noun,      Space, Verb,        Space, Conjunction, Space,       Adjective,   Space,       Noun,      Space,       Adjective, Punctuation, Space}},
            {{Adjective, Space, Noun,        Space, Conjunction, Space,       Adverb,      Space,       Verb,      Punctuation, Space}},
            {{Adverb,    Space, Conjunction, Space, Adjective,   Space,       Verb,        Space,       Noun,      Punctuation, Space}},
            {{Noun,      Space, Verb,        Space, Conjunction, Space,       Noun,        Space,       Adverb,    Space,       Verb,      Punctuation, Space}}
    };

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<> pattern_dist(0, (int) patterns.size() - 1);
    std::uniform_real_distribution<> newline_dist(0, 1);

    std::uniform_int_distribution<> nouns_dist(0, (int) nouns.size() - 1);
    std::uniform_int_distribution<> verbs_dist(0, (int) verbs.size() - 1);
    std::uniform_int_distribution<> conjunctions_dist(0, (int) conjunctions.size() - 1);
    std::uniform_int_distribution<> adjectives_dist(0, (int) adjectives.size() - 1);
    std::uniform_int_distribution<> adverbs_dist(0, (int) adverbs.size() - 1);
    std::uniform_int_distribution<> punctuation_dist(0, (int) punctuation.size() - 1);
    std::uniform_int_distribution<> quotes_dist(0, (int) quotes.size() - 1);
    std::uniform_int_distribution<> commas_dist(0, (int) commas.size() - 1);
    std::uniform_int_distribution<> spaces_dist(0, (int) spaces.size() - 1);


    std::string result;
    bool capitalize_next = true;
    for (int s = 0; s < sentence_count; s++) {
        PseudoSentencePattern chosenPattern = patterns[pattern_dist(rng)];

        for (PseudoWordType token: chosenPattern.pattern) {
            PseudoWord selectedWord;
            switch (token) {
                case Noun:
                    selectedWord = nouns[nouns_dist(rng)];
                    break;
                case Verb:
                    selectedWord = verbs[verbs_dist(rng)];
                    break;
                case Conjunction:
                    selectedWord = conjunctions[conjunctions_dist(rng)];
                    break;
                case Adjective:
                    selectedWord = adjectives[adjectives_dist(rng)];
                    break;
                case Adverb:
                    selectedWord = adverbs[adverbs_dist(rng)];
                    break;
                case Punctuation:
                    selectedWord = punctuation[punctuation_dist(rng)];
                    break;
                case Quote:
                    selectedWord = quotes[quotes_dist(rng)];
                    break;
                case Comma:
                    selectedWord = commas[commas_dist(rng)];
                    break;
                case Space:
                    selectedWord = spaces[spaces_dist(rng)];
                    break;
                default:
                    break;
            }
            if (capitalize_next && (token == Noun || token == Adjective || token == Adverb || token == Verb)) {
                selectedWord.value = capitalize_first_letter(selectedWord.value);
            }
            result += selectedWord.value;
            capitalize_next = token == Punctuation || token == Quote;
        }

        if (newline_dist(rng) < 0.3) {
            result += "\n";
        }
    }

    return result;
}


std::string generate_random_string(int length) {
    std::string charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, (int) charset.size() - 1);
    std::string random_string;

    for (int i = 0; i < length; i++) {
        random_string += charset[dis(gen)];
    }

    return random_string;
}

bool are_vector_of_vectors_equal(const std::vector<std::vector<float>> &v1, const std::vector<std::vector<float>> &v2) {
    if (v1.size() != v2.size()) {
        return false;
    }

    for (size_t i = 0; i < v1.size(); i++) {
        if (v1[i].size() != v2[i].size()) {
            return false;
        }
        for (size_t j = 0; j < v1[i].size(); j++) {
            if (v1[i][j] != v2[i][j]) {
                return false;
            }
        }
    }

    return true;
}


template<typename K, typename V>
bool are_maps_equal(const std::unordered_map<K, V> &map1, const std::unordered_map<K, V> &map2) {
    if (map1.size() != map2.size()) {
        return false;
    }

    typename std::unordered_map<K, V>::const_iterator it1 = map1.begin();
    typename std::unordered_map<K, V>::const_iterator it2 = map2.begin();
    while (it1 != map1.end() && it2 != map2.end()) {
        if (it1->first != it2->first || it1->second != it2->second) {
            return false;
        }
        ++it1;
        ++it2;
    }

    return true;
}

template<typename K, typename V>
bool print_map_differences(const std::unordered_map<K, V> &map1, const std::unordered_map<K, V> &map2) {
    bool differences_found = false;

    std::cout << "Entries in map1 that are not in map2:\n";
    for (const auto &[key, value]: map1) {
        auto it = map2.find(key);
        if (it == map2.end()) {
            std::cout << key << ": " << value << '\n';
            differences_found = true;
        } else if (it->second != value) {
            std::cout << key << " - " << "map1: " << value << ", map2: " << it->second << '\n';
            differences_found = true;
        }
    }

    std::cout << "Entries in map2 that are not in map1:\n";
    for (const auto &[key, value]: map2) {
        auto it = map1.find(key);
        if (it == map1.end()) {
            std::cout << key << ": " << value << '\n';
            differences_found = true;
        }
    }

    return differences_found;
}

//FAIL_TEST(e): This macro takes an exception e as input and prints a message
// showing that the test has failed, along with the source file, line number,
// and function name. Then, it throws the exception e.
#define FAIL_TEST(e) \
            std::cout << "Test failed at " \
                      << __FILE__ << ", " << __LINE__ << ", " << __func__ \
                      << std::endl; \
            throw e

// PASS_TEST(): This macro prints a message showing that the test has passed,
// along with the source file, line number, and function name.
#define PASS_TEST() \
            std::cout << "Test passed at " \
                          << __FILE__ << ", " << __LINE__ << ", " << __func__ \
                          << std::endl

//ASSERT_TRUE(arg): This macro takes a boolean expression arg as input.
// If the expression is false, it prints a failure message along with the
// source file, line number, function name, and the expression itself. Then,
// it throws an exception with the message "Test failed." If the expression
// is true, it prints a success message along with the source file, line
// number, function name, and the expression itself.
#define ASSERT_TRUE(arg) \
            if(!(arg)) { \
                std::cout << "Test failed at " \
                          << __FILE__ << ", " << __LINE__ << ", " << __func__ << ": " \
                          << #arg \
                          << std::endl; \
               throw std::exception("Test failed."); \
            } \
            std::cout << "Test passed at " \
                          << __FILE__ << ", " << __LINE__ << ", " << __func__ << ": " \
                          << #arg \
                          << std::endl

//ASSERT_FALSE(arg): This macro works similarly to ASSERT_TRUE(arg),
// but it checks if the given boolean expression arg is false.
// If it's true, it prints a failure message and throws an exception.
// If it's false, it prints a success message.
#define ASSERT_FALSE(arg) \
            if((arg)) { \
                std::cout << "Test failed at " \
                          << __FILE__ << ", " << __LINE__ << ", " << __func__ << ": " \
                          << #arg \
                          << std::endl; \
               throw std::exception("Test failed."); \
            }             \
            std::cout << "Test passed at " \
                          << __FILE__ << ", " << __LINE__ << ", " << __func__ << ": " \
                          << #arg \
                          << std::endl


//ASSERT_EQ(expected, actual): This macro takes two arguments,
// expected and actual. If they are not equal, it prints a failure
// message showing the source file, line number, function name, expected
// and actual values. If they are equal, it prints that the test passed
// along with the expected and the source file, line number, function name,
// expected and actual values.
#define ASSERT_EQ(expected, actual) \
            if((expected) != (actual)) { \
                std::cout << "Test failed at " \
                          << __FILE__ << ", " << __LINE__ << ", " << __func__ << ": " \
                          << "Expected: " << (expected) << ", Actual: " << (actual) \
                          << std::endl; \
               throw std::exception("Test failed."); \
            }             \
            std::cout << "Test passed at " \
                          << __FILE__ << ", " << __LINE__ << ", " << __func__ << ": " \
                          << "Expected: " << (expected) << ", Actual: " << (actual) \
                          << std::endl

#endif //HAPPYML_UNIT_TEST_HPP
