//
// Created by Erik Hyrkas on 12/18/2022.
//

#include <iostream>
#include "../ml/model.hpp"

using namespace micromldsl;
using namespace microml;
using namespace std;

void testTanh() {
    auto valueRange = columnVector({-100.f, -10.f, -1.f, -0.75, -0.5, -0.25f, 0.f, 0.25f, 0.5f, 0.75f, 1.f, 10.f, 100.f});
    auto tanhActivationFunction = make_shared<TanhActivationFunction>();
    auto result = tanhActivationFunction->activate(valueRange);
    auto expected = columnVector({-1.000, -1.000, -0.761594, -0.635149, -0.462, -0.245, 0.000, 0.245, 0.462, 0.635149, 0.761594, 1.000, 1.000});
    assertEqual(expected, result);
}

void testTanhDerivative() {
    auto valueRange = columnVector({-100.f, -10.f, -1.f, -0.75, -0.5, -0.25f, 0.f, 0.25f, 0.5f, 0.75f, 1.f, 10.f, 100.f});
    auto tanhActivationFunction = make_shared<TanhActivationFunction>();
    auto result = tanhActivationFunction->derivative(valueRange);
    auto expected = columnVector({0.000, 0.000, 0.420, 0.596586, 0.786448, 0.940, 1.000, 0.940, 0.786448, 0.596586, 0.420, 0.000, 0.000});
    assertEqual(expected, result);
}

void testTanhApprox() {
    auto valueRange = columnVector({-100.f, -10.f, -1.f, -0.75, -0.5, -0.25f, 0.f, 0.25f, 0.5f, 0.75f, 1.f, 10.f, 100.f});
    auto tanhActivationFunction = make_shared<TanhApproximationActivationFunction>();
    auto result = tanhActivationFunction->activate(valueRange);
    auto expected = columnVector({-1.000, -1.000, -0.761594, -0.635149, -0.462, -0.245, 0.000, 0.245, 0.462, 0.635149, 0.761594, 1.000, 1.000});
    assertEqual(expected, result);
}

void testTanhApproxDerivative() {
    auto valueRange = columnVector({-100.f, -10.f, -1.f, -0.75, -0.5, -0.25f, 0.f, 0.25f, 0.5f, 0.75f, 1.f, 10.f, 100.f});
    auto tanhActivationFunction = make_shared<TanhApproximationActivationFunction>();
    auto result = tanhActivationFunction->derivative(valueRange);
    auto expected = columnVector({0.000, 0.000, 0.420, 0.596586, 0.786448, 0.940, 1.000, 0.940, 0.786448, 0.596586, 0.420, 0.000, 0.000});
    assertEqual(expected, result);
}

int main() {
    try {
        testTanh();
        testTanhDerivative();
        testTanhApprox();
        testTanhApproxDerivative();
    } catch (const exception &e) {
        cout << e.what() << endl;
    }

    return 0;
}