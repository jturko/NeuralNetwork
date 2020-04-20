
#ifndef RELUNEURON_HH
#define RELUNEURON_HH

#include <iostream>

using namespace std;

class ReLUNeuron
{
  public:
    ReLUNeuron();
    ReLUNeuron(double activation);
    
    double ActivationRaw() { return fActivationRaw; } 
    void ActivationRaw(double activation) { fActivationRaw = activation; }
    
    double Activation();
    double ActivationDerivative();

    void Print();

  private:
    double fActivationRaw;
};

#endif

