
#ifndef NEURON_HH
#define NEURON_HH

#include <iostream>

using namespace std;

class Neuron
{
  public:
    Neuron();
    Neuron(double activation);
    
    double WeightedInput() { return fWeightedInput; } 
    void WeightedInput(double activation) { fWeightedInput = activation; }
    
    virtual double Activation() = 0; 
    virtual double ActivationDerivative() = 0;

    void Print();

    bool IsInput() { return fIsInput; }
    void IsInput(bool val) { fIsInput = val; }
    
  protected:
    double fWeightedInput;
    bool fIsInput;
};

#endif

