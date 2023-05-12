//
// Created by Erik Hyrkas on 11/26/2022.
// Copyright 2022. Usable under MIT license.
//

#ifndef HAPPYML_BIAS_LAYER_HPP
#define HAPPYML_BIAS_LAYER_HPP

namespace happyml {
    class BiasLayer : public happyml::NeuralNetworkLayerFunction {
    public:
        BiasLayer(const string &label, const vector<size_t> &inputShape, const vector<size_t> &outputShape,
                  uint8_t bits,
                  const shared_ptr<happyml::BaseOptimizer> &optimizer) {
            this->label = label;
            this->registration_id = optimizer->registerForBiasChanges();
            this->inputShapes = vector<vector<size_t >>{inputShape};
            this->outputShape = outputShape;
            // In my experiments, at least for the model I was testing, we found the correct results faster by starting at 0 bias.
            // This may be a mistake.
//            this->bias = make_shared<UniformTensor>(outputShape[0], outputShape[1], outputShape[2], 0.f);
            // Original code started with a random value between -0.5 and 0.5:
            this->bias = make_shared<TensorFromXavier>(outputShape[0], outputShape[1], outputShape[2], 42);
            this->bits = bits;
            this->optimizer = optimizer;
            // With models that are not fully 32-bit, if you don't scale the loss
            // you'll have precision errors that are difficult to deal with.
            // I chose to scale the learning rate, which brings a lot of potential issues
            // but is relatively straight forward to do and fast.
            // The biggest issue is that the caller might try to use a learning rate that is too big,
            // and it will not be possible to find good results.
            // There is an nvidia paper on the topic, and they scale the values before they store the weights
            // and then scale those weights back down when they use them. The advantage of this is that
            // a learning rate of X on a 32-bit model, it will work the same if you change some portions
            // to 16-bit. With my approach, if you change any of the portions of the model's precision, you may
            // have to pick new a new learning rate to get good results.
            // I went this route because it is expensive to scale up and down tensors using a view with
            // this framework when you end up with a huge stack of views, since that scaling will constantly
            // get re-applied with every future view that sits over weights.
            // There are situations where I have hundreds of views over a tensor, and adding a single view to
            // the weights will change that to thousands because many of the views sit over multiple weight
            // tensors.
            if (bits == 32) {
                // NOTE: I am taking a small shortcut here. Even without mixed precision, it's important to
                //  reduce the rate we train bias. If bias is trained at the same rate as weights, my observation
                //  is that it can "overpower" the weights, where it causes us to wildly oscillate above and below
                //  our target without ever reaching it.
                // I made this number up. it seemed to work well for both mixed-precision models and for models
                // that are entirely 32-bit.
                mixedPrecisionLearningRateScale = 0.1f;
            } else if (bits == 16) {
                if (optimizer->getLearningRate() < 0.45) {
                    // I made this number up. it seemed to work well for mixed-precision models.
                    mixedPrecisionLearningRateScale = 2.f;
                } else {
                    mixedPrecisionLearningRateScale = 1.f;
                }
            } else {
                if (optimizer->getLearningRate() < 0.3) {
                    // I made this number up. it seemed to work well for mixed-precision models.
                    mixedPrecisionLearningRateScale = 3.0f;
                } else {
                    mixedPrecisionLearningRateScale = 1.0f;
                }
            }
            this->current_batch_size = 0;
        }

        vector<vector<size_t>> getInputShapes() {
            return inputShapes;
        }

        vector<size_t> getOutputShape() {
            return outputShape;
        }

        void saveKnowledge(const string &fullKnowledgePath) override {
            string path = fullKnowledgePath + "/" + label + ".tensor";
            bias->save(path);
        }

        void loadKnowledge(const string &fullKnowledgePath) override {
            string path = fullKnowledgePath + "/" + label + ".tensor";
            this->bias = make_shared<FullTensor>(path);
        }

        // predicting
        shared_ptr<happyml::BaseTensor> forward(const vector<shared_ptr<happyml::BaseTensor>> &input, bool forTraining) override {
            PROFILE_BLOCK(profileBlock);
            if (input.size() > 1) {
                throw runtime_error("BiasNeuron only supports a single input.");
            }
            if (forTraining) {
                current_batch_size++;
            }

            return make_shared<TensorAddTensorView>(input[0], bias);
        }

        // learning
        vector<shared_ptr<BaseTensor>> backward(const shared_ptr<happyml::BaseTensor> &output_error) override {
            PROFILE_BLOCK(profileBlock);

            const auto adjusted_bias_error = make_shared<TensorMultiplyByScalarView>(output_error,
                                                                                     mixedPrecisionLearningRateScale / (float) current_batch_size);
            auto adjusted_bias = optimizer->calculateBiasChange(registration_id,
                                                                bias,
                                                                output_error);
            bias = materializeTensor(adjusted_bias, bits);

            current_batch_size = 0;
            // TODO: partial derivative of bias would always be 1, so we pass along original error. I'm fairly sure this is right.
            // but I notice that the quarter float doesn't handle big shifts in scale very well
            return {output_error};
        }

    private:
        int registration_id;
        shared_ptr<happyml::BaseTensor> bias;
        int current_batch_size;
        uint8_t bits;
        float mixedPrecisionLearningRateScale;
        vector<vector<size_t>> inputShapes;
        vector<size_t> outputShape;
        shared_ptr<happyml::BaseOptimizer> optimizer;
        string label;
    };
}
#endif //HAPPYML_BIAS_LAYER_HPP
