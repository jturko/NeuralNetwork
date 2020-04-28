
#include <vector>
#include <string>

#include "Layer.hh"

using namespace std;

class NeuralNetwork 
{
  public:
    NeuralNetwork(vector<int> topology, string neuronType);
    
    void BuildNetwork();
    void ForwardPropagate(bool verbose=false);
    double CalculateCost();
    void BackwardPropagate(bool verbose=false);
    void UpdateNetwork();

    vector<int> Topology() { return fTopology; }
    int nLayers() { return fTopology.size(); }
    
    string NeuronType() { return fNeuronType; }

    Layer * InputLayer() { return fLayers.at(0); }
    void InputLayer(Layer * input);
    void InputLayer(vector<double> values);
    
    Layer * OutputLayer() { return fLayers.at(fLayers.size()-1); }
    void OutputLayer(Layer * output);
    void OutputLayer(vector<double> values);
    
    Layer * TargetLayer() { return fTargetLayer; }
    void TargetLayer(Layer * target);
    void TargetLayer(vector<double> values);

    double Cost() { return fCost; }

    vector<Layer *> Layers() { return fLayers; }
    vector<Matrix *> Matrices() { return fMatrices; }
    vector<Matrix *> BiasMatrices() { return fBiasMatrices; }
    vector<Matrix *> ErrorMatrices() { return fErrorMatrices; }

  private:
    string fNeuronType;
    vector<int> fTopology;
    vector<Layer *> fLayers;
    vector<Matrix *> fMatrices; 
    vector<Matrix *> fBiasMatrices; 
    vector<Matrix *> fErrorMatrices; 

    Layer * fTargetLayer;
    bool fTargetLayerSet;
    double fCost;
    Matrix * fCostDerivatives;

    double fLearningRate;

};

