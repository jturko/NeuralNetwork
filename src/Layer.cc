
#include "Layer.hh"
#include "SigmoidNeuron.hh"
#include "ReLUNeuron.hh"

Layer::Layer(int nNeurons, string neuronType) {
    fnNeurons = nNeurons;
    for(int i=0; i<nNeurons; i++) {
        fNeuronType = neuronType;
        Neuron * neuron = NULL;
        if(neuronType=="SigmoidNeuron") neuron = new SigmoidNeuron;
        else if(neuronType=="ReLUNeuron") neuron = new ReLUNeuron;
        else {
            cout<<"Unknown neuron type: "<<neuronType<<endl;
            assert(false);
            return;
        }
        fNeurons.push_back(neuron);
    }
}

Matrix * Layer::RowVector() {
    Matrix * m = new Matrix(1, fnNeurons);
    for(int i=0; i<fnNeurons; i++) m->Element(0, i, fNeurons.at(i)->Activation());

    return m;
}

Matrix * Layer::ColumnVector() {
    Matrix * m = new Matrix(fnNeurons, 1);
    for(int i=0; i<fnNeurons; i++) m->Element(i, 0, fNeurons.at(i)->Activation());

    return m;
}

Matrix * Layer::RowVectorRaw() {
    Matrix * m = new Matrix(1, fnNeurons);
    for(int i=0; i<fnNeurons; i++) m->Element(0, i, fNeurons.at(i)->ActivationRaw());

    return m;
}

Matrix * Layer::ColumnVectorRaw() {
    Matrix * m = new Matrix(fnNeurons, 1);
    for(int i=0; i<fnNeurons; i++) m->Element(i, 0, fNeurons.at(i)->ActivationRaw());

    return m;
}

void Layer::ActivationRaw(int neuron, double value) {
    fNeurons.at(neuron)->ActivationRaw(value);
}

void Layer::ActivationsRaw(Matrix m) {
    if(m.nRows() != 1 || m.nCols() != fnNeurons) {
        cerr<<"Wrong dimensions for input matrix, cannot set as layer activations"<<endl;
        assert(false);
        return;
    }
    
    for(int i=0; i<fnNeurons; i++) {
        fNeurons.at(i)->ActivationRaw(m.Element(0,i));
    }
}   

vector<double> Layer::Activations() {
    vector<double> output;
    for(int i=0; i<fnNeurons; i++) output.push_back(fNeurons.at(i)->Activation());
    return output;
}

vector<double> Layer::ActivationsRaw() {
    vector<double> output;
    for(int i=0; i<fnNeurons; i++) output.push_back(fNeurons.at(i)->ActivationRaw());
    return output;
}
