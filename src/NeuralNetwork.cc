
#include "NeuralNetwork.hh"

NeuralNetwork::NeuralNetwork(vector<int> topology, string neuronType) {
    fTopology = topology;
    fNeuronType = neuronType;
    BuildNetwork();
}

void NeuralNetwork::BuildNetwork() {
    cout<<"Building neural network, type: "<<fNeuronType<<endl;
    for(int i=0; i<nLayers()-1; i++) {
        Layer * l = new Layer(fTopology.at(i), fNeuronType);
        fLayers.push_back(l);
        Matrix * m = new Matrix(fTopology.at(i),fTopology.at(i+1));
        fMatrices.push_back(m);
        m = new Matrix(fTopology.at(i),fTopology.at(i+1));
        fBiasMatrices.push_back(m);
    }
    Layer * l = new Layer(fTopology.at(fTopology.size()-1), fNeuronType);
    fLayers.push_back(l);
}

void NeuralNetwork::ForwardPropagate(bool verbose) {
    for(int i=0; i<nLayers()-1; i++) {
        fLayers.at(i+1)->ActivationsRaw((*fLayers.at(i)->RowVector())*(*fMatrices.at(i)));
        if(verbose) {
            cout<<" Layer "<<i<<":"<<endl;
            fLayers.at(i)->RowVector()->Print();
            cout<<" Matrix from layer "<<i<<" -> "<<i+1<<endl;
            fMatrices.at(i)->Print();
        }
    }
    cout<<" Output Layer: "<<endl;
    if(verbose) OutputLayer()->RowVector()->Print();
}


void NeuralNetwork::InputLayer(Layer * input) { 
    if(input->nNeurons() != fTopology.at(0)) {
        cerr<<"Error: input->nNeurons()="<<input->nNeurons()<<" != fTopology->at(0)="<<fTopology.at(0)<<endl;
        assert(false);
        return;
    }
    fLayers.at(0) = input; 
}

void NeuralNetwork::InputLayer(vector<double> values) {
    if(values.size() != fTopology.at(0)) {
        cerr<<"Error: values.size()="<<values.size()<<" != fTopology.at(0)="<<fTopology.at(0)<<endl;
        assert(false);
        return;
    }
    for(int i=0; i<fTopology.at(0); i++) {
        fLayers.at(0)->ActivationRaw(i, values.at(i));
    }    
}

void NeuralNetwork::OutputLayer(Layer * output) { 
    if(output->nNeurons() != fTopology.at(fLayers.size()-1)) {
        cerr<<"Error: output->nNeurons()="<<output->nNeurons()<<" != fTopology->at("<<fLayers.size()-1<<")="<<fTopology.at(fLayers.size()-1)<<endl;
        assert(false);
        return;
    }
    fLayers.at(fLayers.size()-1) = output; 
}

void NeuralNetwork::OutputLayer(vector<double> values) {
    if(values.size() != fTopology.at(fLayers.size()-1)) {
        cerr<<"Error: values.size()="<<values.size()<<" != fTopology.at("<<fLayers.size()-1<<")="<<fTopology.at(fLayers.size()-1)<<endl;
        assert(false);
        return;
    }
    for(int i=0; i<fTopology.at(0); i++) {
        fLayers.at(fLayers.size()-1)->ActivationRaw(i, values.at(i));
    }    
}

