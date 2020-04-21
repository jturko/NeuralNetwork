
#include "Neuron.hh"

Neuron::Neuron() {
    fActivationRaw = 0.0;
}

Neuron::Neuron(double activation) {
    fActivationRaw = activation;
}

void Neuron::Print() {
    cout<<"---------------------------------"<<endl;
    cout<<"Neuron:"<<endl;
    cout<<"ActivationRaw() = "<<ActivationRaw()<<endl;
    cout<<"Activation() = "<<Activation()<<endl;
    cout<<"ActivationDerivative() = "<<ActivationDerivative()<<endl;
    cout<<"---------------------------------"<<endl;
}

