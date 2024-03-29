//
// Created by Erik Hyrkas on 11/26/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_FLATTEN_LAYER_HPP
#define HAPPYML_FLATTEN_LAYER_HPP


namespace happyml {
    class FlattenLayer : public BaseLayer {
    public:
        shared_ptr<BaseTensor> forward(const vector<shared_ptr<BaseTensor>> &input, bool forTraining) override {
            PROFILE_BLOCK(profileBlock);
            if (input.size() != 1) {
                throw runtime_error("Cannot flatten multiple inputs at the same time. Please merge.");
            }
            const auto &nextInput = input[0];
            originalCols = nextInput->columnCount();
            originalRows = nextInput->rowCount();
            if (originalRows == 1) {
                // This flatten function was added unnecessarily. We could throw an exception.
                return nextInput;
            }
            return make_shared<RowFlattenTensorView>(nextInput);
        }

        vector<shared_ptr<BaseTensor>> backward(const shared_ptr<BaseTensor> &output_error) override {
            PROFILE_BLOCK(profileBlock);
            if (originalRows == output_error->rowCount() && originalCols == output_error->columnCount()) {
                // This flatten function was added unnecessarily. We could throw an exception.
                return {output_error};
            }
            return {make_shared<ReshapeTensorView>(output_error, originalRows, originalCols)};
        }

    private:
        size_t originalRows{};
        size_t originalCols{};
    };
}
#endif //HAPPYML_FLATTEN_LAYER_HPP
