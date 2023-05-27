//
// Created by Erik Hyrkas on 5/26/2023.
// Copyright 2023. Usable under MIT license.
//

#ifndef HAPPYML_HAPPYML_VARIANT_HPP
#define HAPPYML_HAPPYML_VARIANT_HPP

#include <string>

namespace happyml {
    struct HappyMLVariant {
        enum class Type {
            INT,
            FLOAT,
            STRING,
            BOOL,
            NONE
        };

        HappyMLVariant() : type_(Type::NONE) {}

        explicit HappyMLVariant(int value) : type_(Type::INT), int_value_(value) {}

        explicit HappyMLVariant(float value) : type_(Type::FLOAT), float_value_(value) {}

        explicit HappyMLVariant(std::string value) : type_(Type::STRING), string_value_(std::move(value)) {}

        explicit HappyMLVariant(bool value) : type_(Type::BOOL), bool_value_(value) {}

        // assignment operators
        HappyMLVariant &operator=(int value) {
            type_ = Type::INT;
            int_value_ = value;
            return *this;
        }

        HappyMLVariant &operator=(float value) {
            type_ = Type::FLOAT;
            float_value_ = value;
            return *this;
        }

        HappyMLVariant &operator=(std::string value) {
            type_ = Type::STRING;
            string_value_ = std::move(value);
            return *this;
        }

        HappyMLVariant &operator=(bool value) {
            type_ = Type::BOOL;
            bool_value_ = value;
            return *this;
        }

        void print(ostream &out) const {
            switch (type_) {
                case Type::INT:
                    out << int_value_;
                    break;
                case Type::FLOAT:
                    out << float_value_;
                    break;
                case Type::STRING:
                    out << string_value_;
                    break;
                case Type::BOOL:
                    out << bool_value_;
                    break;
                case Type::NONE:
                    out << "None";
                    break;
            }
        }

        [[nodiscard]] string to_string() const {
            stringstream ss;
            print(ss);
            return ss.str();
        }

        [[nodiscard]] float as_float() const {
            if (type_ == Type::FLOAT) {
                return float_value_;
            } else if (type_ == Type::INT) {
                return static_cast<float>(int_value_);
            } else if (type_ == Type::STRING) {
                return std::stof(string_value_);
            } else if (type_ == Type::BOOL) {
                return bool_value_ ? 1.0f : 0.0f;
            } else {
                throw std::runtime_error("Cannot convert to float.");
            }
        }

        Type type_;
        int int_value_{};
        float float_value_{};
        std::string string_value_;
        bool bool_value_{};
    };
}
#endif //HAPPYML_HAPPYML_VARIANT_HPP
