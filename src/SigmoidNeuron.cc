
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

void SigmoidNeuron::Print() {
    cout<<"---------------------------------"<<endl;
    cout<<"SigmoidNeuron:"<<endl;
    cout<<"ActivationRaw() = "<<ActivationRaw()<<endl;
    cout<<"Activation() = "<<Activation()<<endl;
    cout<<"ActivationDerivative() = "<<ActivationDerivative()<<endl;
    cout<<"---------------------------------"<<endl;
}

