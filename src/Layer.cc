
#include "Layer.hh"

Layer::Layer(int nNeurons) {
    fnNeurons = nNeurons;
    for(int i=0; i<nNeurons; i++) fNeurons.push_back(new Neuron);
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

void Layer::SetActivationRaw(int neuron, double value) {
    fNeurons.at(neuron)->ActivationRaw(value);
}

void Layer::SetActivationsRaw(Matrix m) {
    if(m.nRows() != 1 || m.nCols() != fnNeurons) {
        cerr<<"Wrong dimensions for input matrix, cannot set as layer activations"<<endl;
        return;
    }
    
    for(int i=0; i<fnNeurons; i++) {
        fNeurons.at(i)->ActivationRaw(m.Element(0,i));
    }
}   

