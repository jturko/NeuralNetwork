
#include "Layer.hh"
#include "SigmoidNeuron.hh"
#include "ReLUNeuron.hh"

Layer::Layer(int nNeurons, string type) {
    fnNeurons = nNeurons;
    for(int i=0; i<nNeurons; i++) {
        Neuron * neuron = NULL;
        if(type=="SigmoidNeuron") neuron = new SigmoidNeuron;
        else if(type=="ReLUNeuron") neuron = new ReLUNeuron;
        else {
            cout<<"Unknown neuron type: "<<type<<endl;
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

