
#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <fstream>

#include "Layer.hh"

using namespace std;

class NeuralNetwork 
{
  public:
    NeuralNetwork(vector<int> topology, string neuronType, bool print_errors = true);
    
    void BuildNetwork();
    
    void ForwardPropagate();
    void BackwardPropagate();
    double CalculateCost();
    void AddToGradient();
    void UpdateNetwork();
    void SGD( vector <pair <vector<double>,vector<double> > > training_data, int batch_size, double learning_rate = 0.1); // stochastic gradient decent

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

    void Verbose(bool val) { fVerbose = val; }
    void PrintErrors(bool val) { fPrintErrors = val; }

  private:
    string fNeuronType;
    vector<int> fTopology;
    vector<Layer *> fLayers;
    
    vector<Matrix *> fMatrices; 
    vector<Matrix *> fBiasMatrices; 
    vector<Matrix *> fErrorMatrices; 

    vector<Matrix *> fGradientMatrices;
    vector<Matrix *> fBiasGradientMatrices;

    Layer * fTargetLayer;
    bool fTargetLayerSet;
    double fCost;
    Matrix * fCostDerivatives;

    double fLearningRate;
    int fBatchSize;

    bool fVerbose;
    bool fPrintErrors;
    ofstream fErrorsFile;

};

