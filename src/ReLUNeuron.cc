
#include "ReLUNeuron.hh"

ReLUNeuron::ReLUNeuron() {
    fActivationRaw = 0.0;
}

ReLUNeuron::ReLUNeuron(double activation) {
    fActivationRaw = activation;
}

double ReLUNeuron::Activation() {
    if(fActivationRaw < 0.) return 0.;
    else return fActivationRaw;    
}

double ReLUNeuron::ActivationDerivative() {
    if(fActivationRaw < 0.) return 0.;
    else return 1.;
}
