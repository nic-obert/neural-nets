#include "neural_network.hh"
#include <string>
#include <fstream>
#include <iostream>

int main() {

    Network<Activations::Relu, Activations::SoftMax, Losses::CrossEntropy, Optimizers::SGD> network(
        8, // inputs number
        4, // layers number
        2, // outputs number
        8, // neuron per layer
        0.001 // learning rate
    );

    std::string fileName = "set.txt";
    std::ifstream file;

    double data[8];
    unsigned int hotOne;

    unsigned int samples;
    unsigned int correct;

    for (int epoch = 0; epoch < 1000; epoch++) {
        file.open(fileName);
        samples = 0;
        correct = 0;

        while (file >> data[0] >> data[1] >> data[2] >> data[3] >> data[4] >> data[5] >> data[6] >> data[7] >> hotOne) {
            //std::cout << std::endl;
            network.feed(data, hotOne);
            //network.printNetworkOutput();
            //network.printLoss();
            network.backwardAndOptimize(hotOne);

            samples++;
            if (network.getOutput()[hotOne] > 0.5) {
                correct++;
            }
        }

        file.close();
        if (!(epoch % 100)) {
            std::cin.get();
            std::cout << std::endl;
            std::cout << "samples: " << samples << " correct: " << correct << std::endl;
            std::cout << "loss: " << network.getLoss() << std::endl;
            std::cout << "accuracy: " << correct * 100 / samples << std::endl;
        }
        
    }
    

}
