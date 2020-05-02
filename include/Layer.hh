
#ifndef LAYER_HH
#define LAYER_HH

#include <vector>
#include <string>

#include "Neuron.hh"
#include "Matrix.hh"

using namespace std;

class Layer 
{
  public:
    Layer(int nNeurons, string type);
    
    int nNeurons() { return fnNeurons; }
    vector<Neuron*> Neurons() { return fNeurons; }

    Matrix * ColumnVector();
    Matrix * RowVector();    
    
    Matrix * ColumnVectorRaw();
    Matrix * RowVectorRaw();    
    
    Matrix * ColumnVectorDerivative();
    Matrix * RowVectorDerivative();    

    void WeightedInputs(vector<double> vals);
    void WeightedInputs(Matrix * m);
    void WeightedInput(int neuron, double value);
    vector<double> WeightedInputs();

    string NeuronType() { return fNeuronType; }

    vector<double> Activations();
    
    bool IsInput() { return fIsInput; }
    void IsInput(bool val);

  private:
    string fNeuronType;
    int fnNeurons;
    vector<Neuron*> fNeurons;
    
    bool fIsInput;

};

#endif

