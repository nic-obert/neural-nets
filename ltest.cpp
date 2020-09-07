#include "DenseLayer.cpp"
#include "Activations.cpp"
#include <iostream>
#include <time.h>
#include "Losses.cpp"
#include "Optimizers.cpp"
#include "neural_network.hh"

int main() {
    srand(time(NULL));

    double data[8] = {
        1.2, -2, 2.1, 0.9, 0.1, -1.4, 0.7
    };
    int hotOne = 0;
    double learningRate = 0.1;

    Network<Activations::Relu, Activations::SoftMax, Losses::CrossEntropy, Optimizers::SGD> network(
        8, // inputs number
        4, // layers number
        4, // outputs number
        8, // neurons per layer
        0.1 // learning rate
        );
    
    for (int epoch = 0; epoch < 10; epoch++) {

        network.feed(data, hotOne);
        network.backwardAndOptimize(hotOne);
        network.printLoss();

    }

    return 0;
}