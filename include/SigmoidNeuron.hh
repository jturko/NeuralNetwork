
#ifndef SIGMOIDNEURON_HH
#define SIGMOIDNEURON_HH

#include <iostream>

#include "Neuron.hh"

using namespace std;

class SigmoidNeuron : public Neuron
{
  public:
    SigmoidNeuron();
    SigmoidNeuron(double activation);
    
    double Activation();
    double ActivationDerivative();
};

#endif

