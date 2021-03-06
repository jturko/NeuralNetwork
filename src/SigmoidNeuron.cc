
#include <cmath>

#include "SigmoidNeuron.hh"

SigmoidNeuron::SigmoidNeuron() {
    fWeightedInput = 0.0;
}

SigmoidNeuron::SigmoidNeuron(double activation) {
    fWeightedInput = activation;
}

double SigmoidNeuron::Activation() {
    if(fIsInput) return fWeightedInput;

    return 1./(1.+exp(-fWeightedInput));
}

double SigmoidNeuron::ActivationDerivative() {
    return Activation()*(1.-Activation());
}
