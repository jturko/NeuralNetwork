
#ifndef LAYER_HH
#define LAYER_HH

#include <vector>

#include "Neuron.hh"
#include "Matrix.hh"

class Layer 
{
  public:
    Layer(int nNeurons);
    
    int nNeurons() { return fnNeurons; }
    std::vector<Neuron*> Neurons() { return fNeurons; }

    Matrix * ColumnVector();
    Matrix * RowVector();    

    void SetActivationsRaw(Matrix m);
    void SetActivationRaw(int neuron, double value);

  private:
    int fnNeurons;
    std::vector<Neuron*> fNeurons;

};

#endif

