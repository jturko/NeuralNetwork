
#include "Neuron.hh"

Neuron::Neuron() {
    fWeightedInput = 0.0;
    fIsInput = false;
}

Neuron::Neuron(double activation) {
    fWeightedInput = activation;
}

void Neuron::Print() {
    cout<<"---------------------------------"<<endl;
    cout<<"Neuron:"<<endl;
    cout<<"WeightedInput() = "<<WeightedInput()<<endl;
    cout<<"Activation() = "<<Activation()<<endl;
    cout<<"ActivationDerivative() = "<<ActivationDerivative()<<endl;
    cout<<"---------------------------------"<<endl;
}

