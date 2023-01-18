//
// Created by Erik Hyrkas on 12/31/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_PATTERN_HPP
#define HAPPYML_PATTERN_HPP

#include <cstring>
#include <algorithm>
#include <sstream>
#include "token.hpp"


using namespace std;

namespace happyml {
    string asLower(const string &original) {
        string temp = original;
        transform(temp.begin(), temp.end(),
                  temp.begin(), tolower);
        return temp;
    }

    enum FrequencyQualifier {
        zeroOrOne,
        zeroOrMore,
    };

    class PatternMatchResult {
    public:
        explicit PatternMatchResult(size_t matchLength) {
            this->matchLength = matchLength;
        }

        [[nodiscard]] size_t getMatchLength() const {
            return matchLength;
        }

    private:
        size_t matchLength;
    };

    class PatternMatcher {
    public:
        shared_ptr<PatternMatchResult> defaultMatch(const string &text,
                                                    size_t offset) {
            return match(text, offset, text.length());
        }

        virtual shared_ptr<PatternMatchResult> match(const string &text,
                                                     size_t offset,
                                                     size_t scanLimit) = 0;

    };

    class AnyPatternMatchers : public PatternMatcher {
    public:
        explicit AnyPatternMatchers(const vector<shared_ptr<PatternMatcher>> &patterns) {
            this->patterns = patterns;
        }

        shared_ptr<PatternMatchResult> match(const string &text, size_t offset, size_t scanLimit) override {
            for (const auto &patternMatcher: patterns) {
                auto nextResult = patternMatcher->match(text, offset, scanLimit);
                if (nextResult) {
                    return nextResult;
                }
            }
            return nullptr;
        }

    private:
        vector<shared_ptr<PatternMatcher>> patterns;
    };

    class AllPatternMatchers : public PatternMatcher {
    public:
        explicit AllPatternMatchers(const vector<shared_ptr<PatternMatcher>> &patterns) {
            this->patterns = patterns;
        }

        shared_ptr<PatternMatchResult> match(const string &text, size_t offset, size_t scanLimit) override {
            size_t combined = 0;
            for (const auto &patternMatcher: patterns) {
                size_t nextOffset = offset + combined;
                shared_ptr<PatternMatchResult> nextResult = patternMatcher->match(text, nextOffset, scanLimit);
                if (!nextResult) {
                    return nullptr;
                }
                combined += nextResult->getMatchLength();
            }
            return make_shared<PatternMatchResult>(combined);
        }

    private:
        vector<shared_ptr<PatternMatcher>> patterns;
    };

    class FrequencyPatternMatcher : public PatternMatcher {
    public:
        FrequencyPatternMatcher(const shared_ptr<PatternMatcher> &patternMatcher,
                                FrequencyQualifier frequencyQualifier) {
            this->patternMatcher = patternMatcher;
            this->frequencyQualifier = frequencyQualifier;
        }

        shared_ptr<PatternMatchResult> match(const string &text, size_t offset, size_t scanLimit) override {
            size_t numberOfMatches = 0;
            size_t currentOffset = offset;
            shared_ptr<PatternMatchResult> nextMatch = patternMatcher->match(text, currentOffset, scanLimit);
            while (nextMatch) {
                currentOffset += nextMatch->getMatchLength();
                numberOfMatches++;
                if (frequencyQualifier == FrequencyQualifier::zeroOrOne) {
                    break;
                }
                nextMatch = patternMatcher->match(text, currentOffset, scanLimit);
            }
            bool matched;
            size_t matchedLength = currentOffset - offset;
            if (numberOfMatches <= 1) {
                matched = frequencyQualifier == FrequencyQualifier::zeroOrOne ||
                          frequencyQualifier == FrequencyQualifier::zeroOrMore;
            } else {
                matched = frequencyQualifier == FrequencyQualifier::zeroOrMore;
            }
            if (!matched) {
                return nullptr;
            }
            return make_shared<PatternMatchResult>(matchedLength);
        }

    private:
        shared_ptr<PatternMatcher> patternMatcher;
        FrequencyQualifier frequencyQualifier;
    };

    class Pattern {
    public:
        Pattern(const string &label, const bool skip,
                const shared_ptr<PatternMatcher> &patternMatcher) {
            this->label = label;
            this->skip = skip;
            this->patternMatcher = patternMatcher;
        }

        shared_ptr<Match> match(const string &text, size_t offset, const string &source = "unknown") {
            shared_ptr<Match> result = nullptr;
            if (offset < text.length()) {
                auto matchSize = patternMatcher->match(text, offset, text.length());
                if (matchSize) {
                    result = make_shared<Match>(matchSize->getMatchLength(),
                                                getLabel(),
                                                text.substr(offset, matchSize->getMatchLength()),
                                                isSkip(),
                                                offset,
                                                source);
                }
            }
            return result;
        }

        string getLabel() {
            return label;
        }

        [[nodiscard]] bool isSkip() const {
            return skip;
        }

    private:
        string label;
        bool skip;
        shared_ptr<PatternMatcher> patternMatcher;
    };

    class AlphaPatternMatcher : public PatternMatcher {
    public:
        shared_ptr<PatternMatchResult> match(const string &text, size_t offset, size_t scanLimit) override {
            if (offset < std::min(text.length(), scanLimit)) {
                auto nextChar = text[offset];
                if (::isalpha(nextChar)) {
                    return make_shared<PatternMatchResult>(1);
                }
            }
            return nullptr;
        }
    };

    class DigitPatternMatcher : public PatternMatcher {
    public:
        shared_ptr<PatternMatchResult> match(const string &text, size_t offset, size_t scanLimit) override {
            if (offset < std::min(text.length(), scanLimit)) {
                auto nextChar = text[offset];
                if (::isdigit(nextChar)) {
                    return make_shared<PatternMatchResult>(1);
                }
            }
            return nullptr;
        }
    };

    class AlphaNumericPatternMatcher : public PatternMatcher {
    public:
        shared_ptr<PatternMatchResult> match(const string &text, size_t offset, size_t scanLimit) override {
            if (offset < std::min(text.length(), scanLimit)) {
                auto nextChar = text[offset];
                if (::isalnum(nextChar)) {
                    return make_shared<PatternMatchResult>(1);
                }
            }
            return nullptr;
        }
    };

    // TODO: this could probably be refactored to be "NotPattern" and take in
    //  any pattern, but I didn't have the energy to figure it out at the
    //  time.
    class NotTextPatternMatcher : public PatternMatcher {
    public:
        explicit NotTextPatternMatcher(const string &textSequence, bool caseInsensitive = false) {
            this->caseInsensitive = caseInsensitive;

            if (caseInsensitive) {
                this->textSequence = asLower(textSequence);
            } else {
                this->textSequence = textSequence;
            }
        }

        shared_ptr<PatternMatchResult> match(const string &text, size_t offset, size_t scanLimit) override {
            const auto maxLen = offset + 1;
            if (maxLen > std::min(text.length(), scanLimit)) {
                return nullptr;
            }
            string sub = text.substr(offset, textSequence.length());
            if ((!caseInsensitive && sub == textSequence) ||
                (caseInsensitive && asLower(sub) == textSequence)) {
                return nullptr;
            }
            return make_shared<PatternMatchResult>(textSequence.length());
        }

    private:
        string textSequence;
        bool caseInsensitive;
    };


    class TextSequencePatternMatcher : public PatternMatcher {
    public:
        explicit TextSequencePatternMatcher(const string &textSequence, bool caseInsensitive = false) {
            this->caseInsensitive = caseInsensitive;

            if (caseInsensitive) {
                this->textSequence = asLower(textSequence);
            } else {
                this->textSequence = textSequence;
            }
        }

        shared_ptr<PatternMatchResult> match(const string &text, size_t offset, size_t scanLimit) override {
            const auto max_len = offset + textSequence.length();
            if (max_len <= std::min(text.length(), scanLimit)) {
                string sub = text.substr(offset, textSequence.length());
                if ((!caseInsensitive && sub == textSequence) ||
                    (caseInsensitive && asLower(sub) == textSequence)) {
                    return make_shared<PatternMatchResult>(textSequence.length());
                }
            }
            return nullptr;
        }

    private:
        string textSequence;
        bool caseInsensitive;
    };


    shared_ptr<Pattern> createToken(const string &label, const string &keyword) {
        const auto patternMatcher = make_shared<TextSequencePatternMatcher>(keyword, true);
        auto pattern = make_shared<Pattern>(label, false, patternMatcher);
        return pattern;
    }

    shared_ptr<Pattern> createKeywordToken(const string &keyword) {
        return createToken("_" + keyword, keyword);
    }

    shared_ptr<Pattern> createSkippedToken(const string &label, const string &text) {
        const auto patternMatcher = make_shared<TextSequencePatternMatcher>(text, true);
        auto pattern = make_shared<Pattern>(label, true, patternMatcher);
        return pattern;
    }

    shared_ptr<Pattern> createCommentPattern() {
        const auto commentToken = make_shared<TextSequencePatternMatcher>("#", true);
        const auto notNewlineToken = make_shared<NotTextPatternMatcher>("\n");
        const auto repeatTokens = make_shared<FrequencyPatternMatcher>(notNewlineToken,
                                                                       FrequencyQualifier::zeroOrMore);
        vector<shared_ptr<PatternMatcher>> patterns{commentToken, repeatTokens};
        const auto combined = make_shared<AllPatternMatchers>(patterns);
        auto pattern = make_shared<Pattern>("_comment", true, combined);
        return pattern;
    }

    shared_ptr<PatternMatcher> createStringPatternMatcher(const string &token) {
        const auto quoteToken = make_shared<TextSequencePatternMatcher>(token, true);
        const auto notQuoteToken = make_shared<NotTextPatternMatcher>(token);
        const auto escapeToken = make_shared<TextSequencePatternMatcher>("\\" + token);
        vector<shared_ptr<PatternMatcher>> anyOrEscape{escapeToken, notQuoteToken};
        const auto anyTokenOrEscapeToken = make_shared<AnyPatternMatchers>(anyOrEscape);
        const auto zeroOrMoreOfNotEscapeToken = make_shared<FrequencyPatternMatcher>(anyTokenOrEscapeToken,
                                                                                     FrequencyQualifier::zeroOrMore);
        vector<shared_ptr<PatternMatcher>> combinedPatterns{quoteToken, zeroOrMoreOfNotEscapeToken, quoteToken};
        const auto combined = make_shared<AllPatternMatchers>(combinedPatterns);
        return combined;
    }

    shared_ptr<Pattern> createStringPattern() {
        vector<shared_ptr<PatternMatcher>> eitherQuoteStringPatterns{createStringPatternMatcher("\""),
                                                                     createStringPatternMatcher("'")};
        const auto eitherQuoteString = make_shared<AnyPatternMatchers>(eitherQuoteStringPatterns);

        auto pattern = make_shared<Pattern>("_string", false, eitherQuoteString);
        return pattern;
    }

    shared_ptr<Pattern> createWordPattern() {
        const auto alphaToken = make_shared<AlphaPatternMatcher>();
        const auto alphaNumericToken = make_shared<AlphaNumericPatternMatcher>();
        const auto repeatTokens = make_shared<FrequencyPatternMatcher>(alphaNumericToken,
                                                                       FrequencyQualifier::zeroOrMore);
        vector<shared_ptr<PatternMatcher>> patterns{alphaToken, repeatTokens};
        const auto wordPatternMatcher = make_shared<AllPatternMatchers>(patterns);
        auto pattern = make_shared<Pattern>("_word", false, wordPatternMatcher);
        return pattern;
    }

    shared_ptr<Pattern> createNumberPattern() {
        const auto numberToken = make_shared<DigitPatternMatcher>();
        const auto periodToken = make_shared<TextSequencePatternMatcher>(".");
        const auto optionalPeriod = make_shared<FrequencyPatternMatcher>(periodToken, FrequencyQualifier::zeroOrOne);
        const auto repeatTokens = make_shared<FrequencyPatternMatcher>(numberToken, FrequencyQualifier::zeroOrMore);
        vector<shared_ptr<PatternMatcher>> patterns{repeatTokens, optionalPeriod, repeatTokens};
        const auto numberPatternMatcher = make_shared<AllPatternMatchers>(patterns);
        auto pattern = make_shared<Pattern>("_number", false, numberPatternMatcher);
        return pattern;
    }
}
#endif //HAPPYML_PATTERN_HPP
