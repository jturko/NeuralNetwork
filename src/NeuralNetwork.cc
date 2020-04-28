
#include <cmath>

#include "Utils.hh"
#include "NeuralNetwork.hh"

NeuralNetwork::NeuralNetwork(vector<int> topology, string neuronType) {
    fTargetLayer = NULL;
    fCostDerivatives = NULL;
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
        Matrix * m = new Matrix(fTopology.at(i+1),fTopology.at(i), true);
        fMatrices.push_back(m);
        Matrix * bm = new Matrix(fTopology.at(i+1),1, true);
        fBiasMatrices.push_back(bm);
    }
    Layer * l = new Layer(fTopology.at(fTopology.size()-1), fNeuronType);
    fLayers.push_back(l);
}

void NeuralNetwork::ForwardPropagate(bool verbose) {
    if(verbose) cout<<endl<<" ---> starting forward propagation..."<<endl;
    for(int i=0; i<nLayers()-1; i++) {
        Matrix * weighted_input = Utils::MatrixAdd(Utils::DotProduct(fMatrices.at(i), fLayers.at(i)->ColumnVector()), fBiasMatrices.at(i));
        fLayers.at(i+1)->WeightedInputs(weighted_input);
        if(verbose) {
            if(i==0) cout<<endl<<"----> Input Layer: "<<endl;
            else cout<<endl<<"----> Layer "<<i<<":"<<endl;
            cout<<" -> Raw values:"<<endl;
            fLayers.at(i)->ColumnVectorRaw()->Print();
            cout<<" -> Activated values:"<<endl;
            fLayers.at(i)->ColumnVector()->Print();
            cout<<" -> Matrix from layer "<<i<<" -> "<<i+1<<endl;
            fMatrices.at(i)->Print();
            cout<<" -> Bias matrix from layer "<<i<<" -> "<<i+1<<endl;
            fBiasMatrices.at(i)->Print();
        }
    }
    cout<<"----> Output Layer: "<<endl;
    if(verbose) { 
        cout<<" -> Raw values:"<<endl;
        OutputLayer()->ColumnVectorRaw()->Print();
        cout<<" -> Activated values:"<<endl;
        OutputLayer()->ColumnVector()->Print();
    }
   
    // if target output set, calculate cost
    if(fTargetLayerSet) {
        CalculateCost();
        if(verbose) cout<<" -> Cost f'n: "<<fCost<<endl;
    }
    if(verbose) cout<<" ---> ending forward propagation..."<<endl;
}

double NeuralNetwork::CalculateCost() {
    fTargetLayerSet = false;
    fCost = 0.;
    if(fCostDerivatives) { delete fCostDerivatives; fCostDerivatives = NULL; }
    fCostDerivatives = new Matrix(OutputLayer()->nNeurons(),1);
    for(int i=0; i<fTopology.at(fLayers.size()-1); i++) { // quadratic cost
        fCost += pow(TargetLayer()->Neurons().at(i)->Activation() - OutputLayer()->Neurons().at(i)->Activation(), 2.);
        fCostDerivatives->Element(i, 0, 2.*(OutputLayer()->Neurons().at(i)->Activation() - TargetLayer()->Neurons().at(i)->Activation()) );
    }
    return fCost;
}

void NeuralNetwork::BackwardPropagate(bool verbose) {
    if(verbose) cout<<endl<<" ---> starting backward propagation..."<<endl;
    
    // calculate error in output layer
    // delta^L = Hadamard( grad(cost(a^L)) , d(sigma)/d(z^L) )
    //  - L = output layer,
    //  - sigma = sigmoid f'n, 
    //  - z^L = weighted inout = m^L*a^{L-1} + b^L
    // gradient of cost f'n is w.r.t layer activation a^L,
    // for quadratic cost f'n, grad(cost(a^L)) = 2*(a^L - y)
    Matrix * output_error = Utils::HadamardProduct( fCostDerivatives, OutputLayer()->ColumnVectorDerivative() );
    if(verbose) {
        cout<<" -> cost gradient:"<<endl;
        fCostDerivatives->Print();
        cout<<" -> output layer sigmoid derivatives:"<<endl;
        OutputLayer()->ColumnVectorDerivative()->Print();
        cout<<" -> Hadamard prod.:"<<endl;
        output_error->Print();
    }
    
    if(verbose) cout<<" ---> ending backward propagation..."<<endl;
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
        fLayers.at(0)->WeightedInput(i, values.at(i));
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
        fLayers.at(fLayers.size()-1)->WeightedInput(i, values.at(i));
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
        fTargetLayer->WeightedInput(i, values.at(i));
    }    
    fTargetLayerSet = true;
}


