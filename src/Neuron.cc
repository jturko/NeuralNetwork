
#include "Neuron.hh"

Neuron::Neuron() {
    fActivationRaw = 0.0;
}

Neuron::Neuron(double activation) {
    fActivationRaw = activation;
}

double Neuron::ActivationReLU() {
    if(fActivationRaw < 0.) return 0;
    else return fActivationRaw;    
}

double Neuron::Activation() {
    return ActivationReLU();
}

void Neuron::Print() {
    cout<<"---------------------------------"<<endl;
    cout<<"Neuron:"<<endl;
    cout<<"ActivationRaw() = "<<ActivationRaw()<<endl;
    cout<<"ActivationReLU() = "<<ActivationReLU()<<endl;
    cout<<"---------------------------------"<<endl;
}
