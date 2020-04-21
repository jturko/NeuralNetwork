
#include <vector>
#include <string>

#include "Layer.hh"

using namespace std;

class NeuralNetwork 
{
  public:
    NeuralNetwork(vector<int> topology, string neuronType);

    vector<int> Topology() { return fTopology; }
    int nLayers() { return fTopology.size(); }
    
    void BuildNetwork();
    void ForwardPropagate(bool verbose=false);
    void BackwardPropagate(bool verbose=false);

    Layer * InputLayer() { return fLayers.at(0); }
    void InputLayer(Layer * input);
    void InputLayer(vector<double> values);
    
    Layer * OutputLayer() { return fLayers.at(fLayers.size()-1); }
    void OutputLayer(Layer * output);
    void OutputLayer(vector<double> values);

  private:
    string fNeuronType;
    vector<int> fTopology;
    vector<Layer *> fLayers;
    vector<Matrix *> fMatrices; 
    vector<Matrix *> fBiasMatrices; 

};

