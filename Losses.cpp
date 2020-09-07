#pragma once
#include <cmath>

namespace Losses {

    struct LossFunction {
        unsigned int inputsNumber;
        double* gradient;

        LossFunction(unsigned int _inputsNumber)
        : inputsNumber(_inputsNumber) {
            gradient = new double[inputsNumber];
        }

        ~LossFunction() {
            delete[] gradient;
        }

        virtual double forward(const double prediction) {return 0;}

        virtual void backward(const double* networkOutput, unsigned int y) {}
    };


    struct CrossEntropy : public LossFunction {
        CrossEntropy(unsigned int _inputsNumber)
        : LossFunction(_inputsNumber) {}

        double forward(const double prediction) override {
            /*
                takes as input a probability distribution
                (e.g. the output of a softmax function)
            */
            return -(log(prediction));
        }

        void backward(const double* softmaxOutput, const unsigned int hotOne) override {
            for (unsigned int output = 0; output < inputsNumber; output++) {
                gradient[output] = softmaxOutput[output];
            }
            gradient[hotOne] -= 1;
        }
    };
    
    
}