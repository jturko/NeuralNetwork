
#include "Neuron.hh"

Neuron::Neuron() {
    fActivationRaw = 0.0;
}

Neuron::Neuron(double activation) {
    fActivationRaw = activation;
}

double Neuron::Activation() {
    return ActivationReLU();
}

double Neuron::ActivationDerivative() {
    return ActivationDerivativeReLU();
}

double Neuron::ActivationReLU() {
    if(fActivationRaw < 0.) return 0.;
    else return fActivationRaw;    
}

double Neuron::ActivationDerivativeReLU() {
    if(fActivationRaw < 0.) return 0.;
    else return 1.;
}

void Neuron::Print() {
    cout<<"---------------------------------"<<endl;
    cout<<"Neuron:"<<endl;
    cout<<"ActivationRaw() = "<<ActivationRaw()<<endl;
    cout<<"Activation() = "<<Activation()<<endl;
    cout<<"ActivationDerivative() = "<<ActivationDerivative()<<endl;
    cout<<"---------------------------------"<<endl;
}

