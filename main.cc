
#include <iostream>
#include <vector>

#include "Neuron.hh"
#include "Matrix.hh"
#include "Layer.hh"
#include "NeuralNetwork.hh"
#include "Utils.hh"

using namespace std;

int main(int argc, char * argv[]) 
{
    srand((unsigned)time(NULL)); rand();
    cout<<"Running NeuralNetwork..."<<endl;

    vector<int> topology;
    topology.push_back(10);
    topology.push_back(20);
    topology.push_back(20);
    topology.push_back(10);
    topology.push_back(1);

    // configure the input layer
    Layer * input = new Layer(topology.at(0));
    double value = 0.;
    for(int i=0; i<topology.at(0); i++) {
        value += 0.25;
        input->SetActivationRaw(i, value);
    }

    // build the network based on the given topology
    // the matrices are initialied as random
    NeuralNetwork * network = new NeuralNetwork(topology);
    network->InputLayer(input);
    network->ForwardPropagate(true);

    return 0;
}

