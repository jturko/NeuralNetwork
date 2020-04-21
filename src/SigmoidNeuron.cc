
#include <cmath>

#include "SigmoidNeuron.hh"

SigmoidNeuron::SigmoidNeuron() {
    fActivationRaw = 0.0;
}

SigmoidNeuron::SigmoidNeuron(double activation) {
    fActivationRaw = activation;
}

double SigmoidNeuron::Activation() {
    return 1./(1.+exp(-fActivationRaw));
}

double SigmoidNeuron::ActivationDerivative() {
    return Activation()*(1.-Activation());
}
