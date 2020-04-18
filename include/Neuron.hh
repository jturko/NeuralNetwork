
#include <iostream>

using namespace std;

class Neuron
{
  public:
    Neuron();
    Neuron(double activation);
    
    double ActivationRaw() { return fActivationRaw; } 
    void ActivationRaw(double activation) { fActivationRaw = activation; }
    
    double ActivationReLU();
    double Activation();

    void Print();

  private:
    double fActivationRaw;
};

