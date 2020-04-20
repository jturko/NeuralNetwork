
#ifndef SIGMOIDNEURON_HH
#define SIGMOIDNEURON_HH

#include <iostream>

using namespace std;

class SigmoidNeuron
{
  public:
    SigmoidNeuron();
    SigmoidNeuron(double activation);
    
    double ActivationRaw() { return fActivationRaw; } 
    void ActivationRaw(double activation) { fActivationRaw = activation; }
    
    double Activation();
    double ActivationDerivative();

    void Print();

  private:
    double fActivationRaw;
};

#endif

