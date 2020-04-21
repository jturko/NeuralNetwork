
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
    
    virtual double Activation() = 0; 
    virtual double ActivationDerivative() = 0;

    void Print();

  protected:
    double fActivationRaw;
};

#endif

