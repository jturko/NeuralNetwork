
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
    topology.push_back(1);
    topology.push_back(5);
    topology.push_back(topology.front()+1);

    string neuronType = "SigmoidNeuron";
    bool verbose = false;
    
    // build the network based on the given topology
    // the matrices are initialied as random
    NeuralNetwork * network = new NeuralNetwork(topology, neuronType);
    network->Verbose(verbose);
    
    double learning_rate = 0.5;
    int n_epochs = 10000;
    int batch_size = 100;
    int total_examples = n_epochs * batch_size;

    // create random training data
    cout<<" --> generating training data ..."<<flush;
    vector< pair< vector<double>, vector<double> > > training_data;
    for(int epoch = 0; epoch < n_epochs; epoch++) {
        for(int example = 0; example < batch_size; example++) {
            vector<double> input_data;
            vector<double> target_data;
            int count = 0;
            for(int neurons = 0; neurons < topology.front(); neurons++) {
                double rndm = (double)floor(Utils::Rndm(0.0000, 1.9999));
                input_data.push_back(rndm);
                if(rndm > 0.) count++;
            }
            target_data.resize(input_data.size()+1);
            target_data.at(count) = 1.0;
            training_data.push_back(make_pair(input_data, target_data));
        }   
    }    
    cout<<" done!"<<endl;

    // train the network using SGD
    network->SGD(training_data, batch_size, learning_rate);
    //network->Print();

    vector<double> test;
    for(int neurons = 0; neurons < topology.front(); neurons++) {
        double rndm = (double)floor(Utils::Rndm(0.0000, 1.9999));
        test.push_back(rndm);
    }
    network->InputLayer(test);
    network->ForwardPropagate();
    cout<<" TEST INPUT:"<<endl;
    network->InputLayer()->ColumnVector()->Print();
    cout<<" NETWORK OUTPUT:"<<endl;
    network->OutputLayer()->ColumnVector()->Print();
    cout<<" NETWORK OUTPUT RAW VALUES: "<<endl;
    network->OutputLayer()->ColumnVectorRaw()->Print();

    return 0;
}

