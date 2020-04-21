
#ifndef RELUNEURON_HH
#define RELUNEURON_HH

#include <iostream>

#include "Neuron.hh"

using namespace std;

class ReLUNeuron : public Neuron
{
  public:
    ReLUNeuron();
    ReLUNeuron(double activation);
    
    double Activation();
    double ActivationDerivative();
};

#endif

