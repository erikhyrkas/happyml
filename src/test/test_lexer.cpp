//
// Created by Erik Hyrkas on 12/31/2022.
// Copyright 2022. Usable under MIT license.
//

#include <iostream>
#include "../lang/happml_script_init.hpp"
#include "../util/unit_test.hpp"
#include "../util/timers.hpp"

using namespace std;
using namespace happyml;

void testMatchTextSequence1() {
    auto textSequencePattern = make_shared<TextSequencePatternMatcher>("erik", true);
    shared_ptr<PatternMatchResult> match;
    string line;

    line = "er";
    match = textSequencePattern->defaultMatch(line, 0);
    ASSERT_FALSE(match);

    string line2 = "erik is coding";
    match = textSequencePattern->defaultMatch(line2, 0);
    ASSERT_TRUE(match);
    ASSERT_TRUE(match->getMatchLength() == 4);

    string line3 = "erik";
    match = textSequencePattern->defaultMatch(line3, 0);
    ASSERT_TRUE(match);
    ASSERT_TRUE(match->getMatchLength() == 4);

    string line4 = "EriK";
    match = textSequencePattern->defaultMatch(line4, 0);
    ASSERT_TRUE(match);
    ASSERT_TRUE(match->getMatchLength() == 4);

    string line5 = "Erik";
    match = textSequencePattern->defaultMatch(line5, 0);
    ASSERT_TRUE(match);
    ASSERT_TRUE(match->getMatchLength() == 4);

    string line6 = "bErik";
    match = textSequencePattern->defaultMatch(line6, 0);
    ASSERT_FALSE(match);

    string line7 = "bErik";
    match = textSequencePattern->defaultMatch(line7, 1);
    ASSERT_TRUE(match);
    ASSERT_TRUE(match->getMatchLength() == 4);

    string line8 = "is Erik coding?";
    match = textSequencePattern->defaultMatch(line8, 3);
    ASSERT_TRUE(match);
    ASSERT_TRUE(match->getMatchLength() == 4);
}


void testMatchTextSequence2() {
    auto textSequencePattern = make_shared<TextSequencePatternMatcher>("Erik", false);
    shared_ptr<PatternMatchResult> match;
    string line;

    line = "er";
    match = textSequencePattern->defaultMatch(line, 0);
    ASSERT_FALSE(match);

    string line2 = "erik is coding";
    match = textSequencePattern->defaultMatch(line2, 0);
    ASSERT_FALSE(match);

    string line3 = "erik";
    match = textSequencePattern->defaultMatch(line3, 0);
    ASSERT_FALSE(match);

    string line4 = "EriK";
    match = textSequencePattern->defaultMatch(line4, 0);
    ASSERT_FALSE(match);

    string line5 = "Erik";
    match = textSequencePattern->defaultMatch(line5, 0);
    ASSERT_TRUE(match);
    ASSERT_TRUE(match->getMatchLength() == 4);

    string line6 = "bErik";
    match = textSequencePattern->defaultMatch(line6, 0);
    ASSERT_FALSE(match);

    string line7 = "bErik";
    match = textSequencePattern->defaultMatch(line7, 1);
    ASSERT_TRUE(match);
    ASSERT_TRUE(match->getMatchLength() == 4);

    string line8 = "is Erik coding?";
    match = textSequencePattern->defaultMatch(line8, 3);
    ASSERT_TRUE(match);
    ASSERT_TRUE(match->getMatchLength() == 4);
}

void testComment() {
    auto commentPattern = createCommentPattern();
    shared_ptr<Match> match;
    string line;

    line = "# test";
    match = commentPattern->match(line, 0);
    ASSERT_TRUE(match);
    ASSERT_TRUE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "# test");

    string line2 = "#\nhi";
    match = commentPattern->match(line2, 0);
    ASSERT_TRUE(match);
    ASSERT_TRUE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "#");

    string line3 = "abc # def\nhi";
    match = commentPattern->match(line3, 4);
    ASSERT_TRUE(match);
    ASSERT_TRUE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "# def");
    ASSERT_TRUE(match->getSource() == "unknown");

    string line4 = "abc #\tdef\r\nhi";
    match = commentPattern->match(line4, 4);
    ASSERT_TRUE(match);
    ASSERT_TRUE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "#\tdef\r");

}


void testDoubleQuoteString() {
    auto doubleQuoteStringPattern = createStringPattern();
    shared_ptr<Match> match;
    string line;

    line = "\"test string\"";
    match = doubleQuoteStringPattern->match(line, 0);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "\"test string\"");

    line = "abc\"test string\"def";
    match = doubleQuoteStringPattern->match(line, 3);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "\"test string\"");

    line = "abc\"test\nstring\"def";
    match = doubleQuoteStringPattern->match(line, 3);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "\"test\nstring\"");

    line = R"(abc"test\"string"def)";
    match = doubleQuoteStringPattern->match(line, 3);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "\"test\\\"string\"");


    line = R"(abc"test\"string\""def)";
    match = doubleQuoteStringPattern->match(line, 3);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "\"test\\\"string\\\"\"");

    line = R"(abc"\""def)";
    match = doubleQuoteStringPattern->match(line, 3);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "\"\\\"\"");
}

void testSingleQuoteString() {
    auto singleQuoteStringPattern = createStringPattern();
    shared_ptr<Match> match;
    string line;

    line = "'test string'";
    match = singleQuoteStringPattern->match(line, 0);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "'test string'");

    line = "abc'test string'def";
    match = singleQuoteStringPattern->match(line, 3);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "'test string'");

    line = "abc'test\nstring'def";
    match = singleQuoteStringPattern->match(line, 3);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "'test\nstring'");

    line = "abc'test\\\'string'def";
    match = singleQuoteStringPattern->match(line, 3);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "'test\\\'string'");


    line = "abc'test\\'string\\''def";
    match = singleQuoteStringPattern->match(line, 3);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "'test\\'string\\''");

    line = "abc'\\''def";
    match = singleQuoteStringPattern->match(line, 3);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "'\\''");
}

void testWord() {
    auto wordPattern = createWordPattern();
    shared_ptr<Match> match;
    string line;

    line = "test string";
    match = wordPattern->match(line, 0);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "test");

    line = "test string";
    match = wordPattern->match(line, 5);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "string");

    line = "test string";
    match = wordPattern->match(line, 6);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "tring");

    line = "test\tstring\n";
    match = wordPattern->match(line, 5);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "string");

    line = "test\tstring\n";
    match = wordPattern->match(line, 0);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "test");
}

void testNumber() {
    auto numberPattern = createNumberPattern();
    shared_ptr<Match> match;
    string line;

    line = ".5";
    match = numberPattern->match(line, 0);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == ".5");

    line = "0.5";
    match = numberPattern->match(line, 0);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "0.5");

    line = "1";
    match = numberPattern->match(line, 0);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "1");

    line = "1000000 abcasdf";
    match = numberPattern->match(line, 0);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "1000000");

    line = "1000000.25234 abcasdf";
    match = numberPattern->match(line, 0);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "1000000.25234");

    line = "1000000.25.234 abcasdf";
    match = numberPattern->match(line, 0);
    ASSERT_TRUE(match);
    ASSERT_FALSE(match->isSkip());
    ASSERT_TRUE(match->getValue() == "1000000.25");
}

void textLexer1() {
    auto lexer = initializeHappymlLexer();

    auto result = lexer->lex(
            "# This is a lexer test\nlet x = 0.5 # other comment\ntrain fast model mymodel using mydataset");
    ASSERT_TRUE(result);
    ASSERT_TRUE("success" == result->getMessage());
    auto matchStream = result->getMatchStream();
    ASSERT_TRUE(matchStream);
    ASSERT_TRUE(12 == matchStream->size());

    while(matchStream->hasNext()) {
        auto next = matchStream->next();
        cout << next->render() << endl;
    }

    result = lexer->lex("*");
    ASSERT_TRUE(result);
    ASSERT_FALSE(result->getMatchStream());
}

int main() {
    try {
        EvenMoreSimpleTimer timer;
        testMatchTextSequence1();
        timer.printMilliseconds();
        testMatchTextSequence2();
        timer.printMilliseconds();
        testComment();
        timer.printMilliseconds();
        testDoubleQuoteString();
        timer.printMilliseconds();
        testSingleQuoteString();
        timer.printMilliseconds();
        testWord();
        timer.printMilliseconds();
        testNumber();
        timer.printMilliseconds();
        textLexer1();
        timer.printMilliseconds();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }
}