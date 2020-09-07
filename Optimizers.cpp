#pragma once
#include "DenseLayer.cpp"

namespace Optimizers {

    // Optimizer base class
    struct Optimizer {

        virtual void optimize(DenseLayer* layer) {}
    };


    // Stochastic Gradinet Deschent optimizer
    struct SGD : public Optimizer{
        double learningRate;

        SGD();

        SGD(double _learningRate) : learningRate(_learningRate) {}

        void optimize(DenseLayer* layer) override
        {
            /* 
            Stochastic Gradient Descent 
            */

        // changing weights
        for (unsigned int neuron = 0; neuron < layer->neuronsNumber; neuron++) {
            for (unsigned int weight = 0; weight < layer->inputsNumber; weight++) {
                layer->weights[neuron][weight] -= learningRate * layer->weightsGradients[neuron][weight];
            }
        }
        // changing biases
        for (unsigned int bias = 0; bias < layer->neuronsNumber; bias++) {
            layer->biases[bias] -= learningRate * layer->biasesGradient[bias];
        }
        }
    
    };

}