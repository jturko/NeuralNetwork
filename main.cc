
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
    topology.push_back(10);
    topology.push_back(2);

    string neuronType = "SigmoidNeuron";
    bool verbose = false;
    
    // build the network based on the given topology
    // the matrices are initialied as random
    NeuralNetwork * network = new NeuralNetwork(topology, neuronType);
    network->Verbose(verbose);
    
    double learning_rate = 1.0;
    int n_epochs = 1000;
    int batch_size = 1000;
    int total_examples = n_epochs * batch_size;

    // create random training data
    cout<<" --> generating training data ..."<<flush;
    vector< pair< vector<double>, vector<double> > > training_data;
    for(int epoch = 0; epoch < n_epochs; epoch++) {
        for(int example = 0; example < batch_size; example++) {
            vector<double> data;
            for(int neurons = 0; neurons < topology.front(); neurons++) {
                double rndm = Utils::Rndm(-2., 2.);
                data.push_back(rndm);
            }
            training_data.push_back(make_pair(data, data));
        }   
    }    
    cout<<" done!"<<endl;

    // train the network using SGD
    network->SGD(training_data, batch_size, learning_rate);
    //network->Print();

    vector<double> test;
    for(int neurons = 0; neurons < topology.front(); neurons++) {
        double rndm = Utils::Rndm(-2., 2.);
        test.push_back(rndm);
    }
    network->InputLayer(test);
    network->ForwardPropagate();
    cout<<" TEST INPUT:"<<endl;
    network->InputLayer()->ColumnVector()->Print();
    cout<<" NETWORK OUTPUT:"<<endl;
    network->OutputLayer()->ColumnVector()->Print();

    return 0;
}

