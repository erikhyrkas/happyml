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

        builder->addInput(2, 3, NodeType::full, ActivationType::tanh)
                ->addOutput(1, ActivationType::tanh);

        auto neuralNetwork = builder->build();

        neuralNetwork->train(xorDataSource, 8000);

        // Dataset:
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