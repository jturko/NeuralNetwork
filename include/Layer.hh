
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

    double WeightedInput(int neuron) { return fNeurons.at(neuron)->WeightedInput(); }

    string NeuronType() { return fNeuronType; }

    vector<double> Activations();
    vector<double> WeightedInputs();
    
  private:
    string fNeuronType;
    int fnNeurons;
    vector<Neuron*> fNeurons;

};

#endif

