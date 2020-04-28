
#include "ReLUNeuron.hh"

ReLUNeuron::ReLUNeuron() {
    fWeightedInput = 0.0;
}

ReLUNeuron::ReLUNeuron(double activation) {
    fWeightedInput = activation;
}

double ReLUNeuron::Activation() {
    if(fWeightedInput < 0.) return 0.;
    else return fWeightedInput;    
}

double ReLUNeuron::ActivationDerivative() {
    if(fWeightedInput < 0.) return 0.;
    else return 1.;
}

