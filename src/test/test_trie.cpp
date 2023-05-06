//
// Created by Erik Hyrkas on 4/18/2023.
// Copyright 2023. Usable under MIT license.
//

#include <iostream>
#include "../types/trie.hpp"
#include "../util/timers.hpp"
#include "../util/unit_test.hpp"

using namespace std;
using namespace happyml;

void test_trie_basic() {
    Trie trie;
    trie.insert("apple", "fruit");
    ASSERT_TRUE(trie.search("apple"));
    ASSERT_TRUE(trie.match("apple"));
    ASSERT_TRUE("fruit" == trie.lookup("apple"));
    ASSERT_TRUE(trie.lookup("app").empty());
    ASSERT_FALSE(trie.match("app"));
    ASSERT_TRUE(trie.startsWith("app"));

    trie.insert("app", "application");
    ASSERT_TRUE("application" == trie.lookup("app"));
}

void test_trie_complete() {
    Trie trie;
    trie.insert("apple", "fruit");
    trie.insert("application", "software");
    trie.insert("apply", "verb");

    cout << trie.complete("ap") << endl;
    ASSERT_TRUE("application" == trie.complete("ap")); // returns "apply" should return "application"
    ASSERT_TRUE("application" == trie.complete("app"));
    ASSERT_TRUE(trie.complete("apz").empty());
    ASSERT_TRUE(trie.complete("xyz").empty());
}


int main() {
    try {
        EvenMoreSimpleTimer timer;
        test_trie_basic();
        timer.printMilliseconds();
        test_trie_complete();
        timer.printMilliseconds();
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}

