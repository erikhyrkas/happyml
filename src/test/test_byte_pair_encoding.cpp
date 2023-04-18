//
// Created by Erik Hyrkas on 3/26/2023.
//
#include <iostream>
#include "../util/unit_test.hpp"
#include "../util/timers.hpp"
#include "../ml/byte_pair_encoder.hpp"

using namespace std;
using namespace happyml;


void test_train_and_encode_decode() {
    BytePairEncoderModel bpe;
    vector<string> const data = {"hello world", "hello mars", "mars is nice"};
    bpe.train(data);

    string const input_text = "hello world";
    auto encoded_text = bpe.encode(input_text);
    string const decoded_text = bpe.decode(encoded_text);
    ASSERT_EQ(input_text, decoded_text);
}

void test_empty_input_encode() {
    BytePairEncoderModel bpe;
    vector<string> const data = {"hello world", "hello mars", "mars is nice"};
    bpe.train(data);
    string const input_text;
    auto encoded_text = bpe.encode(input_text);
    cout << "x: " << string(encoded_text.begin(), encoded_text.end()) << endl;
    ASSERT_TRUE(encoded_text.empty());
}

void test_empty_input_decode() {
    BytePairEncoderModel bpe;
    vector<string> const data = {"hello world", "hello mars", "mars is nice"};
    bpe.train(data);
    auto input_text = u"";
    string decoded_text = bpe.decode(input_text);
    ASSERT_TRUE(decoded_text.empty());
}

void test_training() {
    BytePairEncoderModel bpe;
    vector<string> const data = string_to_tokens("hello world. hello mars. mars is nice, so I say hello.");

    bpe.train(data);

    auto bpe_codes = bpe.getBpeCodes();
    ASSERT_TRUE(!bpe_codes.empty());
    auto hello16 = bpe.encode("hello");
    cout << "hello:" << string(hello16.begin(), hello16.end()) << ":" << bpe.decode(bpe.encode("hello")) << endl;
    auto is16 = bpe.encode("is");
    cout << "is:" << string(is16.begin(), is16.end()) << ":" << bpe.decode(bpe.encode("is")) << endl;
    ASSERT_TRUE("hello" == bpe.decode(bpe.encode("hello")));
    ASSERT_TRUE("mars" == bpe.decode(bpe.encode("mars")));
    ASSERT_TRUE("is" == bpe.decode(bpe.encode("is")));
    ASSERT_TRUE("i" == bpe.decode(bpe.encode("i")));
}

void test_training_large_file() {
    BytePairEncoderModel bpe;
    vector<string> const data = load_file_to_tokens("../data/data.txt");
    bpe.train(data);

    auto bpe_codes = bpe.getBpeCodes();
    ASSERT_TRUE(!bpe_codes.empty());
    auto hello16 = bpe.encode("hello");
    cout << "hello: " << string(hello16.begin(), hello16.end()) << endl;
    auto is16 = bpe.encode("is");
    cout << "is: " << string(is16.begin(), is16.end()) << endl;
    ASSERT_TRUE("hello" == bpe.decode(bpe.encode("hello")));
    ASSERT_TRUE("mars" == bpe.decode(bpe.encode("mars")));
    ASSERT_TRUE("is" == bpe.decode(bpe.encode("is")));
    ASSERT_TRUE("i" == bpe.decode(bpe.encode("i")));
}

void test_training4() {
    BytePairEncoderModel bpe;
//    vector<string> data = string_to_tokens(generate_pseudo_corpus(1000000, 30000));
    vector<string> const data = string_to_tokens(generate_pseudo_corpus(100, 30));

    bpe.train(data);

    auto bpe_codes = bpe.getBpeCodes();
    ASSERT_TRUE(!bpe_codes.empty());
    auto hello16 = bpe.encode("hello");
    cout << "hello: " << string(hello16.begin(), hello16.end()) << endl;
    auto is16 = bpe.encode("is");
    cout << "is: " << string(is16.begin(), is16.end()) << endl;
    ASSERT_TRUE("hello" == bpe.decode(bpe.encode("hello")));
    ASSERT_TRUE("mars" == bpe.decode(bpe.encode("mars")));
    ASSERT_TRUE("is" == bpe.decode(bpe.encode("is")));
    ASSERT_TRUE("i" == bpe.decode(bpe.encode("i")));

    cout << "Compression: " << bpe.validate_compression_rate(data) << endl;

}


void test_training5() {
    // I'm not really sure if it is advisable to train twice. I was pondering the possibility
    // and added some basic support for it, but I can't be sure the end results are right.
    // more testing is needed. It LOOKS like it works, but I have a nagging feeling that
    // there's more to understand. I don't feel like I saved enough state to properly
    // create a check point.
    BytePairEncoderModel bpe;
//    vector<string> data = string_to_tokens(generate_pseudo_corpus(1000000, 30000));
    vector<string> const data = string_to_tokens(generate_pseudo_corpus(100, 30));
    bpe.train(data);
    vector<string> const data2 = string_to_tokens(generate_pseudo_corpus(100, 30));
    bpe.train(data2);

    auto bpe_codes = bpe.getBpeCodes();
    ASSERT_TRUE(!bpe_codes.empty());
    auto hello16 = bpe.encode("hello");
    cout << "hello: " << string(hello16.begin(), hello16.end()) << endl;
    auto is16 = bpe.encode("is");
    cout << "is: " << string(is16.begin(), is16.end()) << endl;
    ASSERT_TRUE("hello" == bpe.decode(bpe.encode("hello")));
    ASSERT_TRUE("mars" == bpe.decode(bpe.encode("mars")));
    ASSERT_TRUE("is" == bpe.decode(bpe.encode("is")));
    ASSERT_TRUE("i" == bpe.decode(bpe.encode("i")));
}

void test_multiple_encodings() {
    BytePairEncoderModel bpe;
    vector<string> const data = {"hello world", "hello mars", "mars is nice"};
    bpe.train(data);

    string const input_text1 = "hello world";
    auto encoded_text1 = bpe.encode(input_text1);
    ASSERT_TRUE(!encoded_text1.empty());

    string input_text2 = "mars is nice";
    auto encoded_text2 = bpe.encode(input_text2);
    ASSERT_TRUE(!encoded_text2.empty());

    ASSERT_FALSE(encoded_text1 == encoded_text2);
}

void test_buildVocab() {
    BytePairEncoderModel model;
    // Test case 1: Check if the function returns an empty unordered_map if the input vector is empty
    vector<string> const empty_data;
    auto empty_vocab = model.buildVocab(empty_data);
    ASSERT_TRUE(empty_vocab.empty());

    // Test case 2: Check if the function returns the expected unordered_map for a simple input vector
    vector<string> const data = {"hello", "world"};
    unordered_map<u16string, size_t> const expected_vocab = {
            {u"he", 1},
            {u"el", 1},
            {u"ll", 1},
            {u"lo", 1},
            {u"ow", 1},
            {u"or", 1},
            {u"rl", 1},
            {u"ld", 1}
    };
    auto vocab = model.buildVocab(data);
    are_maps_equal(vocab, expected_vocab);

    // Test case 3: Check if the function correctly handles the case when a character pair already exists in the unordered_map
    vector<string> const data2 = {"hello", "hell"};
    unordered_map<u16string, size_t> const expected_vocab2 = {
            {u"he", 2},
            {u"el", 1},
            {u"ll", 1},
            {u"lo", 1}
    };
    auto vocab2 = model.buildVocab(data2);
    are_maps_equal(vocab2, expected_vocab2);

    // Test case 4: Check if the function correctly handles the case when the input strings contain non-ASCII characters
    vector<string> const data3 = {"こんにちは", "你好"};
    unordered_map<u16string, size_t> const expected_vocab3 = {
            {u"こん",   1},
            {u"んに",   1},
            {u"にち",   1},
            {u"ちは",   1},
            {u"你好", 1}
    };
    auto vocab3 = model.buildVocab(data3);
    are_maps_equal(vocab3, expected_vocab3);
}

void test_u16string_replace_all() {
    {
        u16string input = u"the quick brown fox jumps over the lazy dog";
        u16string const expected_output = u"the slow brown fox jumps over the lazy dog";
        u16string const replacement = u"slow";
        u16string const find = u"quick";
        u16string_replace_all(input, find, replacement);
        cout << string(input.begin(), input.end()) << endl;
        ASSERT_TRUE(input == expected_output);
    }

    // Test case 1: empty input string
    {
        u16string input;
        u16string const expected_output;
        u16string const replacement = u"foo";
        u16string const find = u"bar";

        u16string_replace_all(input, find, replacement);

        ASSERT_TRUE(input == expected_output);
    }

    // Test case 2: empty substring to find
    {
        u16string input = u"foo";
        u16string const expected_output = u"foo";
        u16string const replacement = u"bar";
        u16string const find;

        u16string_replace_all(input, find, replacement);

        ASSERT_TRUE(input == expected_output);
    }

    // Test case 3: empty substring replacement
    {
        u16string input = u"foo";
        u16string const expected_output = u"f";
        u16string const replacement;
        u16string const find = u"o";

        u16string_replace_all(input, find, replacement);
        cout << "test case 3: " << string(input.begin(), input.end()) << endl;

        ASSERT_TRUE(input == expected_output);
    }

    // Test case 4: multiple occurrences of the substring to find
    {
        u16string input = u"foo bar baz foo";
        u16string const expected_output = u"qux bar baz qux";
        u16string const replacement = u"qux";
        u16string const find = u"foo";

        u16string_replace_all(input, find, replacement);

        ASSERT_TRUE(input == expected_output);
    }

    // Test case 5: substring to find not found in the input string
    {
        u16string input = u"hello world";
        u16string const expected_output = u"hello world";
        u16string const replacement = u"foo";
        u16string const find = u"bar";

        u16string_replace_all(input, find, replacement);

        ASSERT_TRUE(input == expected_output);
    }
}

void test_findMostFrequentPair() {
    {
        unordered_map<u16string, size_t> const vocab = {
                {u"ab", 10},
                {u"bc", 20},
                {u"cd", 5},
                {u"de", 30},
                {u"ef", 10}
        };

        // Define the expected output for the test
        pair<u16string, size_t> const expected_output = {u"de", 30};

        // Call the function being tested
        pair<u16string, size_t> const actual_output = BytePairEncoderModel::findMostFrequentPair(vocab, 5);

        // Check if the actual output matches the expected output
        ASSERT_TRUE(actual_output == expected_output);
    }
    {
        // Test 1: Empty vocabulary
        unordered_map<u16string, size_t> const vocab1 = {};
        pair<u16string, size_t> const expected_output1 = {u"", 0};
        pair<u16string, size_t> const actual_output1 = BytePairEncoderModel::findMostFrequentPair(vocab1, 1);
        ASSERT_TRUE(actual_output1 == expected_output1);
    }

    {
        // Test 2: Vocabulary with one character pair
        unordered_map<u16string, size_t> const vocab2 = {{u"ab", 5}};
        pair<u16string, size_t> const expected_output2 = {u"ab", 5};
        pair<u16string, size_t> const actual_output2 = BytePairEncoderModel::findMostFrequentPair(vocab2, 1);
        ASSERT_TRUE(actual_output2 == expected_output2);
    }

    {
        // Test 3: Vocabulary with no pairs above the frequency threshold
        unordered_map<u16string, size_t> const vocab3 = {
                {u"ab", 1},
                {u"bc", 2},
                {u"cd", 3}
        };
        pair<u16string, size_t> const expected_output3 = {u"", 0};
        pair<u16string, size_t> const actual_output3 = BytePairEncoderModel::findMostFrequentPair(vocab3, 5);
        ASSERT_TRUE(actual_output3 == expected_output3);
    }

    {
        // Test 4: Vocabulary with multiple pairs above the frequency threshold
        unordered_map<u16string, size_t> const vocab4 = {
                {u"ab", 5},
                {u"bc", 10},
                {u"cd", 5},
                {u"de", 10},
                {u"ef", 5},
                {u"fg", 10},
                {u"gh", 5}
        };
        pair<u16string, size_t> const expected_output4a = {u"bc", 10};
        pair<u16string, size_t> const expected_output4b = {u"de", 10};
        pair<u16string, size_t> const expected_output4c = {u"fg", 10};
        pair<u16string, size_t> const actual_output4 = BytePairEncoderModel::findMostFrequentPair(vocab4, 5);
        ASSERT_TRUE(actual_output4 == expected_output4a || actual_output4 == expected_output4b ||
                    actual_output4 == expected_output4c);
    }
}

void test_save_load() {
    BytePairEncoderModel bpe1;
    vector<string> const data = string_to_tokens(generate_pseudo_corpus(100, 30));
    bpe1.train(data);
    auto bpe_codes1 = bpe1.getBpeCodes();
    ASSERT_TRUE(!bpe_codes1.empty());
    auto hello16_1 = bpe1.encode("hello");
    bpe1.save("../repo", "bpe_test", true);

    BytePairEncoderModel bpe;
    bpe.load("../repo", "bpe_test");
    auto bpe_codes = bpe.getBpeCodes();
    ASSERT_TRUE(!bpe_codes.empty());
    ASSERT_TRUE(bpe_codes1.size() == bpe_codes.size());
    auto hello16 = bpe.encode("hello");
    cout << "hello: " << string(hello16.begin(), hello16.end()) << endl;
    ASSERT_TRUE(hello16_1 == hello16);
    auto is16 = bpe.encode("is");
    cout << "is: " << string(is16.begin(), is16.end()) << endl;
    ASSERT_TRUE("hello" == bpe.decode(bpe.encode("hello")));
    ASSERT_TRUE("mars" == bpe.decode(bpe.encode("mars")));
    ASSERT_TRUE("is" == bpe.decode(bpe.encode("is")));
    ASSERT_TRUE("i" == bpe.decode(bpe.encode("i")));
}

void test_train_folder() {
    {
        BytePairEncoderModel bpe;
        bpe.train_on_folder("../data/misc");
        bpe.save("../repo", "bpe_test_large_folder", true);
    }
    {
        BytePairEncoderModel bpe;
        bpe.load("../repo", "bpe_test_large_folder");
        ASSERT_TRUE(!bpe.getBpeCodes().empty());

        auto hello16 = bpe.encode("hello");
        cout << "hello: " << string(hello16.begin(), hello16.end()) << endl;
        auto is16 = bpe.encode("is");
        cout << "is: " << string(is16.begin(), is16.end()) << endl;

        ASSERT_TRUE("hello" == bpe.decode(bpe.encode("hello")));
        ASSERT_TRUE("mars" == bpe.decode(bpe.encode("mars")));
        ASSERT_TRUE("is" == bpe.decode(bpe.encode("is")));
        ASSERT_TRUE("i" == bpe.decode(bpe.encode("i")));

        const vector<string> data = load_file_to_tokens("../data/data.txt");
        cout << "Compression: " << bpe.validate_compression_rate(data) << endl;
    }
}

void test_train_file() {
    {
        BytePairEncoderModel bpe;
        bpe.train_on_file("../data/data.txt");
        bpe.save("../repo", "bpe_data", true);
    }
    {
        BytePairEncoderModel bpe;
        bpe.load("../repo", "bpe_data");
        ASSERT_TRUE(!bpe.getBpeCodes().empty());

        auto hello16 = bpe.encode("hello");
        cout << "hello: " << string(hello16.begin(), hello16.end()) << endl;
        auto is16 = bpe.encode("is");
        cout << "is: " << string(is16.begin(), is16.end()) << endl;

        ASSERT_TRUE("hello" == bpe.decode(bpe.encode("hello")));
        ASSERT_TRUE("mars" == bpe.decode(bpe.encode("mars")));
        ASSERT_TRUE("is" == bpe.decode(bpe.encode("is")));
        ASSERT_TRUE("i" == bpe.decode(bpe.encode("i")));
    }
}

void test_validate_file() {
    BytePairEncoderModel bpe;
    bpe.load("../happyml_repo", "default_token_encoder");
    ASSERT_TRUE(!bpe.getBpeCodes().empty());

    auto hello16 = bpe.encode("hello");
    cout << "hello: " << string(hello16.begin(), hello16.end()) << endl;
    auto is16 = bpe.encode("is");
    cout << "is: " << string(is16.begin(), is16.end()) << endl;

    ASSERT_TRUE("hello" == bpe.decode(bpe.encode("hello")));
    ASSERT_TRUE("mars" == bpe.decode(bpe.encode("mars")));
    ASSERT_TRUE("is" == bpe.decode(bpe.encode("is")));
    ASSERT_TRUE("i" == bpe.decode(bpe.encode("i")));
    const vector<string> data = load_file_to_tokens("../data/text/data.txt");

    cout << "Compression: " << bpe.validate_compression_rate(data) << endl;
}


int main() {
    try {
        EvenMoreSimpleTimer timer;
//        test_validate_file();
//        timer.printMilliseconds();

//        test_train_file();
//        timer.printMilliseconds();

//        test_train_folder();
//        timer.printMilliseconds();

//        test_training_large_file();
//        timer.printMilliseconds();

        test_save_load();
        timer.printMilliseconds();

        test_training5();
        timer.printMilliseconds();

        test_training4();
        timer.printMilliseconds();

        test_training();
        timer.printMilliseconds();

        test_findMostFrequentPair();
        timer.printMilliseconds();

        test_buildVocab();
        timer.printMilliseconds();

        test_u16string_replace_all();
        timer.printMilliseconds();

        test_train_and_encode_decode();
        timer.printMilliseconds();

        test_empty_input_decode();
        timer.printMilliseconds();

        test_multiple_encodings();
        timer.printMilliseconds();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}