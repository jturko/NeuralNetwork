
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
    
    // TRAINING THE NETWORK TO IDENTIFY THE LARGEST NUMBER

    vector<int> topology;
    topology.push_back(5);
    topology.push_back(100);
    topology.push_back(topology.front());

    string neuronType = "SigmoidNeuron";
    bool verbose = false;
    
    // build the network based on the given topology
    // the matrices are initialied as random
    NeuralNetwork * network = new NeuralNetwork(topology, neuronType);
    network->Verbose(verbose);
    
    double learning_rate = 1.0;
    int n_epochs = 2500;
    int batch_size = 100;
    int total_examples = n_epochs * batch_size;

    // create random training data
    cout<<" --> generating training data ..."<<flush;
    vector< pair< vector<double>, vector<double> > > training_data;
    for(int epoch = 0; epoch < n_epochs; epoch++) {
        for(int example = 0; example < batch_size; example++) {
            vector<double> input_data;
            vector<double> target_data;
            int biggest_element = 0; double biggest_value = -1.0;
            for(int neurons = 0; neurons < topology.front(); neurons++) {
                double rndm = Utils::Rndm(0., 10.);
                input_data.push_back(rndm);
                if(rndm > biggest_value) {
                    biggest_value = rndm;
                    biggest_element = neurons;
                }
            }
            target_data.resize(input_data.size());
            target_data.at(biggest_element) = 1.0;
            training_data.push_back(make_pair(input_data, target_data));
        }   
    }    
    cout<<" done!"<<endl;

    // train the network using SGD
    network->SGD(training_data, batch_size, learning_rate);
    //network->Print();

    // testing the network
    int n_tests = 1000;
    int n_correct = 0;
    for(int test = 0; test < n_tests; test++) {
        vector<double> test_data;
        int correct_biggest_element = 0;
        double correct_biggest_value = -1.;
        for(int neuron = 0; neuron < topology.front(); neuron++) {
           double rndm = Utils::Rndm(0., 10.);
           test_data.push_back(rndm);
            if(correct_biggest_value < rndm) {
                correct_biggest_element = neuron;
                correct_biggest_value = rndm;
            }
        }
        network->InputLayer(test_data);
        network->ForwardPropagate();
        
        vector<double> result = network->OutputLayer()->Activations();    
        int biggest_element = 0;
        double biggest_value = -1.;
        for(int neuron = 0; neuron < topology.back(); neuron++) {
            if(biggest_value < result.at(neuron)) {
                biggest_element = neuron;
                biggest_value = result.at(neuron);
            }
        }

        if(biggest_element == correct_biggest_element) { 
            n_correct++;
        }
        else {
            cout<<fixed;
            cout<<" -> incorrect! biggest value was "<<correct_biggest_value<<" in neuron "<<correct_biggest_element;
            cout<<" , but my guess was that it was element "<<biggest_element<<" which was "<<test_data.at(biggest_element)<<" ... sorry!"<<endl;
            //cout<<" TEST INPUT:"<<endl;
            //network->InputLayer()->ColumnVector()->Print();
            //cout<<" NETWORK OUTPUT:"<<endl;
            //network->OutputLayer()->ColumnVector()->Print();
            //cout<<" NETWORK OUTPUT RAW VALUES: "<<endl;
            //network->OutputLayer()->ColumnVectorRaw()->Print();
        }
    }
    cout<<" -> test results: "<<n_correct<<"/"<<n_tests<<" ("<<100.*n_correct/n_tests<<"\%) correct"<<endl;


    // TRAINING THE NETWORK TO COUNT
    /*
    
    vector<int> topology;
    topology.push_back(10);
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

    */

    return 0;
}

