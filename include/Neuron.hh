
#ifndef NEURON_HH
#define NEURON_HH

#include <iostream>

using namespace std;

class Neuron
{
  public:
    Neuron();
    Neuron(double activation);
    
    double ActivationRaw() { return fActivationRaw; } 
    void ActivationRaw(double activation) { fActivationRaw = activation; }
    
    double Activation();
    double ActivationReLU();

    double ActivationDerivative();
    double ActivationDerivativeReLU();

    void Print();

  private:
    double fActivationRaw;
};

#endif

