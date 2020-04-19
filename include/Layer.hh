
#ifndef LAYER_HH
#define LAYER_HH

#include <vector>

#include "Neuron.hh"
#include "Matrix.hh"

class Layer 
{
  public:
    Layer(int n);
    
    int NumNeurons() { return fNumNeurons; }
    std::vector<Neuron*> Neurons() { return fNeurons; }

    Matrix * ColumnVector();
    Matrix * RowVector();    

    void Print();

  private:
    int fNumNeurons;
    std::vector<Neuron*> fNeurons;

};

#endif

