
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

    void ActivationsRaw(Matrix * m);
    void ActivationRaw(int neuron, double value);

    double ActivationRaw(int neuron) { return fNeurons.at(neuron)->ActivationRaw(); }

    string NeuronType() { return fNeuronType; }

    vector<double> Activations();
    vector<double> ActivationsRaw();
    
  private:
    string fNeuronType;
    int fnNeurons;
    vector<Neuron*> fNeurons;

};

#endif

