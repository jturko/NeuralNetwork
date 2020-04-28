
#include <iostream>
#include <vector>
#include <string>

#include <fstream>
#include <cstdlib>
#include <ctime>

#include "Neuron.hh"
#include "Matrix.hh"
#include "Layer.hh"
#include "NeuralNetwork.hh"
#include "Utils.hh"

using namespace std;

int main(int argc, char * argv[]) 
{
    srand((unsigned)time(NULL)); 
    for(int i=0; i<100; i++) rand();

    //Utils::read_mnist();

    vector<int> topology;
    topology.push_back(2);
    topology.push_back(3);
    topology.push_back(2);

    string neuronType = "SigmoidNeuron";

    // configure the input layer starting values
    Layer * input = new Layer(topology.at(0), neuronType);
    double value = 0.;
    for(int i=0; i<topology.at(0); i++) {
        input->WeightedInput(i, Utils::RndmGaus(0.,1.));
    }

    // build the network based on the given topology
    // the matrices are initialied as random
    NeuralNetwork * network = new NeuralNetwork(topology, neuronType);
    network->InputLayer(input);
    network->TargetLayer(input);
    network->ForwardPropagate(true);
    network->BackwardPropagate(true);

    return 0;
}

