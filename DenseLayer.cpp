#pragma once
#include <iostream>
#include "Activations.cpp"

class DenseLayer {


public:

    unsigned int inputsNumber;
    unsigned int neuronsNumber;

    double** weights;
    double* biases;

    double* outputs;

    double* layerInputs;
    double** weightsGradients;
    double* biasesGradient;
    double* inputsGradient;

    // --------- CONSTRUCTOR / DESTRUCTOR

    DenseLayer();


    DenseLayer(
        unsigned int _inputsNumber,
        unsigned int _neuronsNumber
        )
    : inputsNumber(_inputsNumber),
      neuronsNumber(_neuronsNumber)
    {
        weights = new double*[neuronsNumber];
        biases = new double[neuronsNumber];

        outputs = new double[neuronsNumber];

        layerInputs = new double[inputsNumber];
        weightsGradients = new double*[neuronsNumber];
        biasesGradient = new double[neuronsNumber];
        inputsGradient = new double[inputsNumber];

        
        // create 2 dimensional array for weights
        for (unsigned int neuron = 0; neuron < neuronsNumber; neuron++) {
            weights[neuron] = new double[inputsNumber];
            biases[neuron] = 0;
            weightsGradients[neuron] = new double[inputsNumber];
            for (unsigned int input = 0; input < inputsNumber; input++) {
                weights[neuron][input] = (rand() % 19 + (-9)) * 0.1;
            }
        }
    }


    ~DenseLayer() {

        for (unsigned int neuron = 0; neuron < neuronsNumber; neuron ++) {
            delete[] weights[neuron];
            delete[] weightsGradients[neuron];
        }

        delete[] biases;
        delete[] weights;
        delete[] outputs;
        delete[] layerInputs;
        delete[] weightsGradients;
        delete[] biasesGradient;
        delete[] inputsGradient;

    }

    // -------- FUNCTIONS

    

    virtual void forward(const double *inputs) {
        // copying inputs for backpropagation
        for (unsigned int input = 0; input < inputsNumber; input++) {
            layerInputs[input] = inputs[input];
        }
        // activation function(inputs * weights + bias)
        for (unsigned int neuron = 0; neuron < neuronsNumber; neuron++) {
            // multiplying by weights
            for (unsigned int input = 0; input < inputsNumber; input++) {
                outputs[neuron] += inputs[input] * weights[neuron][input];
            }
            // adding bias
            outputs[neuron] += biases[neuron];

        }

    }


    virtual void backward(const double* activationGradient) {
        /* 
            calculates the gradients of this layer's weights
            and biases
            here is called the backward method of activation functions
            takes as argument the gradient of its output (input of its activation function)
        */

        for (unsigned int neuron = 0; neuron < neuronsNumber; neuron++) {
            for (unsigned int weight = 0; weight < inputsNumber; weight++) {
                /*
                    following the chain rule:
                        multiply the partial derivatives (layerInputs[weight] * outputGradient[neuron])
                        the partial derivative of a weight is its input since the partial
                        derivative of x multiplied by y is y ( f(x) = x*y --> f'(x) = y )
                */
                weightsGradients[neuron][weight] = layerInputs[weight] * activationGradient[neuron];

                // calculate the neuron's impact on the network
                inputsGradient[neuron] += activationGradient[neuron] * weights[neuron][weight];
            }
            /*
                calculate the bias' impact on loss function
                the partial derivative of weighted sum (i * w + b) with respect to the bias
                is int 1 since the partial derivative of a sum is 1 and the bias is summed
                following the chain rule:
                    the impact of the bias on the loss function is calculated
                    by multiplying its partial derivative on the neuron's output
                    by the partial derivative of the neuron's output on the loss function (outputGradient[neuron])
            */
            biasesGradient[neuron] = activationGradient[neuron];
        }

    }


    // -------- PRINTING / DEBUGGING

    void printOutputs() const {
        for (unsigned int output = 0; output < neuronsNumber; output++) {
            std::cout << outputs[output] << " ";
        }
        std::cout << "\n";
    }


    void printWeights() const{
        for (unsigned int neuron = 0; neuron < neuronsNumber; neuron++) {
            for (int weight = 0; weight < inputsNumber; weight++) {
                std::cout << weights[neuron][weight] << " ";
            }
            std::cout << "\n";
        }
    }


    void printBiases() const{
        for (unsigned int neuron = 0; neuron < neuronsNumber; neuron++) {
            std::cout << biases[neuron] << " ";
        }
        std::cout << "\n";
    }


    void printWeightsGradients() const {
        for (unsigned int neuron = 0; neuron < neuronsNumber; neuron++) {
            for (unsigned int weight = 0; weight < inputsNumber; weight++) {
                std::cout << weightsGradients[neuron][weight] << " ";
            }
            std::cout << "\n";
        }
    }


    void printBiasesGradient() const {
        for (unsigned int neuron = 0; neuron < neuronsNumber; neuron++) {
            std::cout << biases[neuron] << " ";
        }
        std::cout << "\n";
    }


    void printInputs() const {
        for (unsigned int input = 0; input < inputsNumber; input++) {
            std::cout << layerInputs[input] << " ";
        }
        std::cout << "\n";
    }
    
};