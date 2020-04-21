
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

    vector<int> topology;
    topology.push_back(2);
    topology.push_back(3);
    topology.push_back(1);

    string neuronType = "SigmoidNeuron";

    // configure the input layer
    Layer * input = new Layer(topology.at(0), neuronType);
    double value = 0.;
    for(int i=0; i<topology.at(0); i++) {
        input->ActivationRaw(i, Utils::Rndm(-5.,5.));
    }

    // build the network based on the given topology
    // the matrices are initialied as random
    NeuralNetwork * network = new NeuralNetwork(topology, neuronType);
    network->InputLayer(input);
    network->ForwardPropagate(true);

    return 0;
}

