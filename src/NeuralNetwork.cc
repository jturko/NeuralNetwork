
#include <cmath>

#include "NeuralNetwork.hh"

NeuralNetwork::NeuralNetwork(vector<int> topology, string neuronType) {
    fTargetLayer = NULL;
    fTargetLayerSet = false;
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
        Matrix * bm = new Matrix(1,fTopology.at(i+1));
        fBiasMatrices.push_back(bm);
    }
    Layer * l = new Layer(fTopology.at(fTopology.size()-1), fNeuronType);
    fLayers.push_back(l);
}

void NeuralNetwork::ForwardPropagate(bool verbose) {
    for(int i=0; i<nLayers()-1; i++) {
        fLayers.at(i+1)->ActivationsRaw(((*fLayers.at(i)->RowVector())*(*fMatrices.at(i)))+(*fBiasMatrices.at(i)));
        if(verbose) {
            if(i==0) cout<<endl<<"----> First Layer: "<<endl;
            else cout<<endl<<"----> Layer "<<i<<":"<<endl;
            cout<<" -> Raw values:"<<endl;
            fLayers.at(i)->RowVectorRaw()->Print();
            cout<<" -> Activated values:"<<endl;
            fLayers.at(i)->RowVector()->Print();
            cout<<" -> Matrix from layer "<<i<<" -> "<<i+1<<endl;
            fMatrices.at(i)->Print();
            cout<<" -> Bias matrix from layer "<<i<<" -> "<<i+1<<endl;
            fBiasMatrices.at(i)->Print();
        }
    }
    cout<<endl<<"----> Output Layer: "<<endl;
    if(verbose) { 
        cout<<" -> Raw values:"<<endl;
        OutputLayer()->RowVectorRaw()->Print();
        cout<<" -> Activated values:"<<endl;
        OutputLayer()->RowVector()->Print();
    }
   
    // if target output set, calculate cost
    if(fTargetLayerSet) {
        CalculateCost();
        if(verbose) cout<<" -> Cost f'n: "<<fCost<<endl;
    }
}


void NeuralNetwork::InputLayer(Layer * input) { 
    if(input->nNeurons() != fTopology.at(0)) {
        cerr<<"input->nNeurons()="<<input->nNeurons()<<" != fTopology->at(0)="<<fTopology.at(0)<<endl;
        assert(false);
        return;
    }
    if(input->NeuronType() != this->NeuronType()) {
        cerr<<"input->NeuronType()="<<input->NeuronType()<<" != this->NeuronType()="<<this->NeuronType()<<endl;
        assert(false);
        return;
    }
    fLayers.at(0) = input; 
}

void NeuralNetwork::InputLayer(vector<double> values) {
    if(values.size() != fTopology.at(0)) {
        cerr<<"values.size()="<<values.size()<<" != fTopology.at(0)="<<fTopology.at(0)<<endl;
        assert(false);
        return;
    }
    for(int i=0; i<fTopology.at(0); i++) {
        fLayers.at(0)->ActivationRaw(i, values.at(i));
    }    
}

void NeuralNetwork::OutputLayer(Layer * output) { 
    if(output->nNeurons() != fTopology.at(fLayers.size()-1)) {
        cerr<<"output->nNeurons()="<<output->nNeurons()<<" != fTopology->at("<<fLayers.size()-1<<")="<<fTopology.at(fLayers.size()-1)<<endl;
        assert(false);
        return;
    }
    if(output->NeuronType() != this->NeuronType()) {
        cerr<<"output->NeuronType()="<<output->NeuronType()<<" != this->NeuronType()="<<this->NeuronType()<<endl;
        assert(false);
        return;
    }
    fLayers.at(fLayers.size()-1) = output; 
}

void NeuralNetwork::OutputLayer(vector<double> values) {
    if(values.size() != fTopology.at(fLayers.size()-1)) {
        cerr<<"values.size()="<<values.size()<<" != fTopology.at("<<fLayers.size()-1<<")="<<fTopology.at(fLayers.size()-1)<<endl;
        assert(false);
        return;
    }
    for(int i=0; i<fTopology.at(0); i++) {
        fLayers.at(fLayers.size()-1)->ActivationRaw(i, values.at(i));
    }    
}

void NeuralNetwork::TargetLayer(Layer * target) { 
    if(target->nNeurons() != fTopology.at(fLayers.size()-1)) {
        cerr<<"target->nNeurons()="<<target->nNeurons()<<" != fTopology->at("<<fLayers.size()-1<<")="<<fTopology.at(fLayers.size()-1)<<endl;
        assert(false);
        return;
    }
    if(target->NeuronType() != this->NeuronType()) {
        cerr<<"target->NeuronType()="<<target->NeuronType()<<" != this->NeuronType()="<<this->NeuronType()<<endl;
        assert(false);
        return;
    }
    fTargetLayer = target; 
    fTargetLayerSet = true;
}

void NeuralNetwork::TargetLayer(vector<double> values) {
    if(values.size() != fTopology.at(fLayers.size()-1)) {
        cerr<<"values.size()="<<values.size()<<" != fTopology.at("<<fLayers.size()-1<<")="<<fTopology.at(fLayers.size()-1)<<endl;
        assert(false);
        return;
    }
    for(int i=0; i<fTopology.at(0); i++) {
        fTargetLayer->ActivationRaw(i, values.at(i));
    }    
    fTargetLayerSet = true;
}

double NeuralNetwork::CalculateCost() {
    fTargetLayerSet = false;
    fCost = 0.;
    for(int i=0; i<fTopology.at(fLayers.size()-1); i++) { // quadratic cost
        fCost += pow(TargetLayer()->Neurons().at(i)->Activation() - OutputLayer()->Neurons().at(i)->Activation(), 2.);
    }
    return fCost;
}


