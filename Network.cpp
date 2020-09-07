#pragma once
#include <iostream>
#include <time.h>
#include <cmath>
#include "DenseLayer.cpp"
#include "Activations.cpp"
#include "Losses.cpp"
#include "Optimizers.cpp"


template <
            typename InnerActivationType,
            typename OutputActivationType,
            typename LossType,
            typename OptimizerType
        >
class Network {
private:

    unsigned int layersNumber;
    unsigned int outputsNumber;
    unsigned int inputsNumber;
    unsigned int neuronPerLayer;

    InnerActivationType** innerActivations;
    OutputActivationType* outputActivation;

    DenseLayer** layers;
    
    LossType* lossFunction;
    double loss;

    OptimizerType* optimizer;


public:

    Network(
        unsigned int _inputsNumber,
        unsigned int _layersNumber,
        unsigned int _outputsNumber,
        unsigned int _neuronPerLayer,
        double _learningRate
        ) 
    : inputsNumber(_inputsNumber),
      layersNumber(_layersNumber),
      outputsNumber(_outputsNumber),
      neuronPerLayer(_neuronPerLayer)
    {
        // set the random seed based on current time
        srand(time(NULL));

        // initialize layers and activations
        layers = new DenseLayer*[layersNumber];
        innerActivations = new InnerActivationType*[layersNumber-1];
    

        layers[0] = new DenseLayer(inputsNumber, neuronPerLayer);
        innerActivations[0] = new InnerActivationType(neuronPerLayer);

        for (int layer = 1; layer < layersNumber - 1; layer ++) {
            layers[layer] = new DenseLayer(neuronPerLayer, neuronPerLayer);
            innerActivations[layer] = new InnerActivationType(neuronPerLayer);
        }

        layers[layersNumber-1] = new DenseLayer(neuronPerLayer, outputsNumber);
        outputActivation = new OutputActivationType(outputsNumber);

        // initialize loss function
        lossFunction = new LossType(outputsNumber);

        // initialize optimizer
        optimizer = new OptimizerType(_learningRate);

    }


    ~Network() {
        // delete layers and activation functions
        for (unsigned int layer = 0; layer < layersNumber-1; layer ++) {
            delete layers[layer];
            delete innerActivations[layer];
        }
        delete layers[layersNumber-1];
        
        delete[] layers;
        delete[] innerActivations;

        delete outputActivation;

        delete lossFunction;
        delete optimizer;
        
    }


    void forward(const double* values) {
        // forward pass through input layer
        layers[0]->forward(values);
        innerActivations[0]->forward(layers[0]->outputs);

        // forward pass through hidden layers
        for (int layer = 1; layer < layersNumber-1; layer ++) {
            layers[layer]->forward(innerActivations[layer-1]->outputs);
            innerActivations[layer]->forward(layers[layer]->outputs);
        }
        
        // forward pass through output layer
        layers[layersNumber-1]->forward(innerActivations[layersNumber-2]->outputs);
        outputActivation->forward(layers[layersNumber-1]->outputs);
    }


    void backward(const unsigned int hotOne) {
        /*
            performs backward propagation and
            calculates the gradients of the network's
            layers
        */
       lossFunction->backward(getOutput(), hotOne);

       outputActivation->backward(lossFunction->gradient);
       layers[layersNumber-1]->backward(outputActivation->gradient);

       for (unsigned int layer = layersNumber-2; layer > -1; layer--) {
           innerActivations[layer]->backward(layers[layer+1]->inputsGradient);
           layers[layer]->backward(innerActivations[layer]->gradient);
       }
    }


    void backwardAndOptimize(const unsigned int hotOne) {
        lossFunction->backward(getOutput(), hotOne);

        outputActivation->backward(lossFunction->gradient);
        layers[layersNumber-1]->backward(outputActivation->gradient);
        optimizer->optimize(layers[layersNumber-1]);

        for (unsigned int layer = layersNumber-2; layer > -1; layer--) {
            innerActivations[layer]->backward(layers[layer+1]->inputsGradient);
            layers[layer]->backward(innerActivations[layer]->gradient);
            optimizer->optimize(layers[layer]);
       }
    }


    void optimize() {
        /*
            performs an optimization based on
            the network's optimizer
        */
       for (unsigned int layer = 0; layer < layersNumber; layer++) {
           optimizer->optimize(layers[layer]);
       }
    }


    void feed(const double *values) {
        /*
            takes just input values, no labels
            does not optimize nor learn, just forwards
        */
        forward(values);
    }


    void feed(const double* values, const unsigned int hotOne) {
        /*
            takes input values together with labels
            for supervised machine learning
            hotOne label is the right classification of a sample
            it is an unsigned integer representing the index of
            the expected class in the output of the softmax function
        */
        forward(values);
        loss = lossFunction->forward(getOutput()[hotOne]);
    }


    const double* getOutput() const {
        return outputActivation->outputs;
    }


    const double getLoss() const {
        return loss;
    }


    void store(const char* fileName) const {
        // TODO make a store function that stores the network in a file
    }


    // DEBUGGING METHODS


    void printLayersOutputs() const {
        for (unsigned int layer = 0; layer < layersNumber; layer ++) {
            layers[layer]->printOutputs();
            std::cout << "\n";
        }
    }


    void printActivationsOutputs() const {
        for (unsigned int layer = 0; layer < layersNumber-1; layer++) {
            innerActivations[layer]->printOutputs();
            std::cout << "\n";
        }
    }


    void printFullOutputs() const {
        // print outputs of both layers and activation functions
        for (unsigned int layer = 0; layer < layersNumber-1; layer++) {
            std::cout << "Layer " << layer << ": ";
            layers[layer]->printOutputs();
            std::cout << "Activ " << layer << ": ";
            innerActivations[layer]->printOutputs();
            std::cout << "\n";
        }
        std::cout << "Layer " << layersNumber-1 << ": ";
        layers[layersNumber-1]->printOutputs();
        std::cout << "Activ " << layersNumber-1 << ": ";
        outputActivation->printOutputs();
    }
    

    void printLayerOutput(unsigned int layer) const {
        // print outputs of a given layer
        layers[layer]->printOutputs();
    }


    void printLayerInput(unsigned int layer) const {
        // print inputs of a given layer
        layers[layer]->printInputs();
    }


    void printNetworkOutput() const {
        // prints the output of the network (output of last activation function)
        outputActivation->printOutputs();
    }


    void printLayerWeights(unsigned int layer) const {
        // prints the weights of a given layer
        layers[layer]->printWeights();
    }


    void printLayerBiases(unsigned int layer) const {
        // prints the biases of a given layer
        layers[layer]->printBiases();
    }


    void printLayerWeightsGradients(unsigned int layer) const {
        // prints the weights gradients of a given layer
        layers[layer]->printWeightsGradients();
    }


    void printLayerBiasesGradients(unsigned int layer) const {
        // prints the biases gradients of a given layer
        layers[layer]->printBiasesGradient();
    }


    void printLoss() const {
        std::cout << loss << std::endl;
    }
};

