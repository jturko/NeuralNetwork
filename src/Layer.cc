
#include "Layer.hh"

Layer::Layer(int n) {
    fNumNeurons = n;
    for(int i=0; i<n; i++) fNeurons.push_back(new Neuron);
}

Matrix * Layer::RowVector() {
    Matrix * m = new Matrix(1, fNumNeurons);
    for(int i=0; i<fNumNeurons; i++) m->Element(0, i, fNeurons.at(i)->Activation());

    return m;
}

Matrix * Layer::ColumnVector() {
    Matrix * m = new Matrix(fNumNeurons, 1);
    for(int i=0; i<fNumNeurons; i++) m->Element(i, 0, fNeurons.at(i)->Activation());

    return m;
}

