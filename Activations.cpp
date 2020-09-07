#pragma once
#include <iostream>
#include <cmath>


namespace Activations {

// BASE CLASS TEMPLATES

    struct InnerActivation {
        double* inputs;
        unsigned int inputsNumber;
        double* gradient;
        double* outputs;

        InnerActivation(unsigned int _inputsNumber) 
        : inputsNumber(_inputsNumber)
        {
            inputs = new double[inputsNumber];
            gradient = new double[inputsNumber];
            outputs = new double[inputsNumber];
        };

        ~InnerActivation() {
            delete[] inputs;
            delete[] gradient;
            delete[] outputs;
        }


        virtual void forward(const double* functionInputs) {}

        virtual void backward(const double* outputGradient) {}

        void printOutputs() const {
            for (unsigned int output = 0; output < inputsNumber; output++) {
                std::cout << outputs[output] << " ";
            }
            std::cout << "\n";
        }
    };


    struct OutputActivation {
        double* inputs;
        unsigned int inputsNumber;
        double* gradient;
        double* outputs;

        OutputActivation(unsigned int _inputsNumber) 
        : inputsNumber(_inputsNumber)
        {
            inputs = new double[inputsNumber];
            gradient = new double[inputsNumber];
            outputs = new double[inputsNumber];
        };

        ~OutputActivation() {
            delete[] inputs;
            delete[] gradient;
            delete[] outputs;
        }

        virtual void forward(const double* functionInputs) {}

        virtual void backward(const double* outputGradient) {}

        void printOutputs() const {
            for (unsigned int output = 0; output < inputsNumber; output++) {
                std::cout << outputs[output] << " ";
            }
            std::cout << "\n";
        }
    };


// ACTUAL ACTIVATION FUNCTIONS

// INNER ACTIVATION FUNCTION (InnerActivation)

    struct Relu : public InnerActivation {
        Relu(unsigned int _inputsNumber)
        : InnerActivation(_inputsNumber) {}

        void forward(const double* reluInput) override {
            /*
                for every layer output if it's less than 0
                output is 0
                otherwise it's left as it is (== input)
            */
            for (unsigned int input = 0; input < inputsNumber; input++) {
                // copying inputs for backpropagation
                inputs[input] = reluInput[input];
                // output = input * [0/1] based on result of (input > 0)
                outputs[input] = reluInput[input] * (reluInput[input] > 0);
            }
        }

        void backward(const double* outputGradient) override {
            /*
                iterate through relu's gradient
                it's input was negative or 0 then
                it has no impact on the next layer's input and thus 
                it's partial derivative is 0
                otherwise it has an impact of 1
                following the chain rule:
                    0 * next_layer = 0
                    1 * next_layer = next_layer
            */
            for (unsigned int input = 0; input < inputsNumber; input++) {
                // gradient = input * [0/1] based on result of (input > 0)
                gradient[input] = outputGradient[input] * (inputs[input] > 0);
            }
        }
    };


    struct Sigmoid : public InnerActivation {
        Sigmoid(unsigned int _inputsNumber)
        : InnerActivation(_inputsNumber) {}

        void forward(const double* sigmoidInput) override {
            for (unsigned int input = 0; input < inputsNumber; input++) {
                outputs[input] = 1 / (1 + exp(-sigmoidInput[input]));
            }
        }

        void backward(const double* outputGradient) override {
            // TODO add backward function
            //return (forward(value) * (1 - forward(value)));
        }        
    };


// OUTPUT ACTIVATION FUNCTIONS (OutputActivation)

    struct SoftMax : public OutputActivation {
        SoftMax(unsigned int _inputsNumber)
        : OutputActivation(_inputsNumber) {}


        void forward(const double* smInputs) override {
            /*
                takes as input the outputs of a neural network and
                turns them into a normalized probability distribution
            */
            double biggestValue = smInputs[0];
            for (unsigned int input = 0; input < inputsNumber; input ++) {
                // copy inputs for backpropagation
                inputs[input] = smInputs[input];
                if (smInputs[input] > biggestValue) {
                    biggestValue = smInputs[input];
                }
            }

            double expSum = 0;
            for (int value = 0; value < inputsNumber; value ++) {
                outputs[value] = exp(smInputs[value] - biggestValue);
                expSum += outputs[value];
            }

            for (int value = 0; value < inputsNumber; value ++) {
                outputs[value] /= expSum;
            }
        }

        void backward(const double* outputGradient) override {
            /*
                Softmax activation function will always be used
                along with a cross-entopy loss function, thus 
                the gradient can be calculated in just one of the two backward
                functions (in this case is calculated in cross-entropy's)
            */
            // copy loss function's gradient
            for (unsigned int output = 0; output < inputsNumber; output++) {
                gradient[output] = outputGradient[output];
            }
        }
        
    };
  

}
