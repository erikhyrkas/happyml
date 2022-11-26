//
// Created by ehyrk on 11/24/2022.
//

#include <memory>
#include "model.hpp"

using namespace std;
using namespace microml;
using namespace micromldsl;

int main() {
    try {
        auto xorDataSource = make_shared<TestXorDataSource>();

        auto builder = createSGDModel()
                ->setLearningRate(0.1)
                ->setLossFunction(micromldsl::mse);

        auto neuralNetwork = builder->build();

        auto activation = make_shared<TanhActivationFunction>();
        auto optimizer = make_shared<SGDOptimizer>(0.1);

        bool use_32_bit = false;

        // todo: this is super awkward. Need a better builder.
        auto fc1 = make_shared<NeuralNetworkNode>(optimizer->createFullyConnectedNeurons(2, 3, use_32_bit));
        neuralNetwork->addHead(fc1);

        auto bias1 = make_shared<NeuralNetworkNode>(optimizer->createBias(3, 3, use_32_bit));
        fc1->add(bias1);

        auto act1 = make_shared<NeuralNetworkNode>(make_shared<NeuralNetworkActivationFunction>(activation));
        bias1->add(act1);

        auto fc2 = make_shared<NeuralNetworkNode>(optimizer->createFullyConnectedNeurons(3, 1, true));
        act1->add(fc2);

        auto bias2 = make_shared<NeuralNetworkNode>(optimizer->createBias(1, 1, true));
        fc2->add(bias2);

        auto act2 = make_shared<NeuralNetworkOutputNode>(make_shared<NeuralNetworkActivationFunction>(activation));
        bias2->add(act2);
        neuralNetwork->addOutput(act2);

        neuralNetwork->train(xorDataSource, 8000);

//        pairs.push_back(std::make_shared<TrainingPair>(std::vector<float>{0.f, 0.f}, std::vector<float>{0.f}));
//        pairs.push_back(std::make_shared<TrainingPair>(std::vector<float>{0.f, 1.f}, std::vector<float>{1.f}));
//        pairs.push_back(std::make_shared<TrainingPair>(std::vector<float>{1.f, 0.f}, std::vector<float>{1.f}));
//        pairs.push_back(std::make_shared<TrainingPair>(std::vector<float>{1.f, 1.f}, std::vector<float>{0.f}));

        auto result1 = neuralNetwork->predict(vector<shared_ptr<BaseTensor>>{make_shared<FullTensor>(vector<float>{0.0f, 0.0f})});
        cout << "predict: 0 xor 0 = " << fixed << setprecision(4) << result1[0]->get_val(0) << " correct value is " << 0 << endl;

        auto result2 = neuralNetwork->predict(vector<shared_ptr<BaseTensor>>{make_shared<FullTensor>(vector<float>{0.0f, 1.0f})});
        cout << "predict: 0 xor 1 = " << result2[0]->get_val(0) << " correct value is " << 1 << endl;

        auto result3 = neuralNetwork->predict(vector<shared_ptr<BaseTensor>>{make_shared<FullTensor>(vector<float>{1.0f, 0.0f})});
        cout << "predict: 1 xor 0 = " << result3[0]->get_val(0) << " correct value is " << 1 << endl;

        auto result4 = neuralNetwork->predict(vector<shared_ptr<BaseTensor>>{make_shared<FullTensor>(vector<float>{1.0f, 1.0f})});
        cout << "predict: 1 xor 1 = " << result4[0]->get_val(0) << " correct value is " << 0 << endl;

    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}