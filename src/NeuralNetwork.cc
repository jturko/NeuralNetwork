
#include <cmath>

#include "Utils.hh"
#include "NeuralNetwork.hh"

NeuralNetwork::NeuralNetwork(vector<int> topology, string neuronType) {
    fTopology = topology;
    fNeuronType = neuronType;
    
    fVerbose = false;
    fBatchSize = 1;
    fLearningRate = 0.1; // small arbitrary value
    fTargetLayer = NULL;
    fCostDerivatives = NULL;
    fTargetLayerSet = false;
    
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

void NeuralNetwork::ForwardPropagate() {
    if(fVerbose) cout<<endl<<" ---> starting forward propagation..."<<endl;
    for(int i=0; i<nLayers()-1; i++) {
        Matrix * weighted_input = Utils::MatrixAdd(Utils::DotProduct(fMatrices.at(i), fLayers.at(i)->ColumnVector()), fBiasMatrices.at(i));
        fLayers.at(i+1)->WeightedInputs(weighted_input);
        if(fVerbose) {
            if(i==0) cout<<endl<<" ---> INPUT LAYER: "<<endl;
            else cout<<endl<<" ---> LAYER "<<i<<":"<<endl;
            cout<<"  -> Weighted input values:"<<endl;
            fLayers.at(i)->ColumnVectorRaw()->Print();
            cout<<"  -> Activated values:"<<endl;
            fLayers.at(i)->ColumnVector()->Print();
            cout<<"  -> Matrix from layer "<<i<<" -> "<<i+1<<endl;
            fMatrices.at(i)->Print();
            cout<<"  -> Bias matrix from layer "<<i<<" -> "<<i+1<<endl;
            fBiasMatrices.at(i)->Print();
        }
    }
    if(fVerbose) { 
        cout<<endl<<" ---> OUTPUT LAYER: "<<endl;
        cout<<"  -> Weighted input values:"<<endl;
        OutputLayer()->ColumnVectorRaw()->Print();
        cout<<"  -> Activated values:"<<endl;
        OutputLayer()->ColumnVector()->Print();
    }
   
    if(fTargetLayerSet) {
        CalculateCost();
        if(fVerbose) cout<<"  -> Cost f'n: "<<fCost<<endl;
    }
    if(fVerbose) cout<<"---> ending forward propagation..."<<endl;
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

void NeuralNetwork::BackwardPropagate() {
    // this backward propagation calculates the error matrix for each layer
    // the calculation of the actual cost function derivatives from this 
    // is done separately
    
    if(fVerbose) cout<<endl<<"---> starting backward propagation..."<<endl;
    fErrorMatrices.clear();
    fErrorMatrices.resize(fMatrices.size());    

    // calculate error in output layer
    // delta^L = Hadamard( grad(cost(a^L)) , d(sigma)/d(z^L) )
    //  - L = output layer,
    //  - sigma = sigmoid f'n, 
    //  - z^L = weighted inout = m^L*a^{L-1} + b^L
    // gradient of cost f'n is w.r.t layer activation a^L,
    // for quadratic cost f'n, grad(cost(a^L)) = 2*(a^L - y)
    fErrorMatrices.back() = Utils::HadamardProduct( fCostDerivatives, OutputLayer()->ColumnVectorDerivative() );
    if(fVerbose) {
        cout<<endl<<" ---> OUTPUT LAYER "<<endl;
        cout<<" -> cost gradient:"<<endl;
        fCostDerivatives->Print();
        cout<<" -> output layer sigmoid derivatives:"<<endl;
        OutputLayer()->ColumnVectorDerivative()->Print();
        cout<<" -> output layer error matrix:"<<endl;
        fErrorMatrices.back()->Print();
    }
    
    // calculate the error for each layer before
    // delta^{l} = Hadamard( w^{l+1} * delta^{l+1} , d(sigma)/d(z^l) )
    //  - l = layer of interest
    //  - w^l = weight matrix for layer l
    for(int i=fTopology.size()-3; i>=0; i--) {
        fErrorMatrices.at(i) = Utils::HadamardProduct( Utils::DotProduct( fMatrices.at(i+1)->Transpose(), fErrorMatrices.at(i+1) ) , fLayers.at(i+1)->ColumnVectorDerivative() );
        if(fVerbose) {
            cout<<endl<<" ---> LAYER "<<i<<endl;
            cout<<" -> transpose matrix for layer "<<i+1<<":"<<endl;
            fMatrices.at(i+1)->Transpose()->Print();
            cout<<" -> error for layer "<<i+1<<":"<<endl;
            fErrorMatrices.at(i+1)->Print();
            cout<<" -> layer "<<i<<" sigmoid derivatives:"<<endl;
            fLayers.at(i+1)->ColumnVectorDerivative()->Print();
            cout<<" -> layer "<<i<<" error matrix:"<<endl;
            fErrorMatrices.at(i)->Print();  
        }
    }

    if(fVerbose) cout<<"---> ending backward propagation..."<<endl;
}

void NeuralNetwork::AddToGradient() {
    // go through each weight and bias gradient and add the values calculated by BackwardPropagate
    
    if(fVerbose) cout<<" ---> Adding to gradient..."<<endl;

    if(fGradientMatrices.size() == 0 || fBiasGradientMatrices.size()==0) {
        for(int layer=0; layer<nLayers()-1; layer++) {
            fGradientMatrices.push_back(new Matrix(fTopology.at(layer+1), fTopology.at(layer)));
            fBiasGradientMatrices.push_back(new Matrix(fTopology.at(layer+1), 1));
        }
    }

    double current_val, update;
    for(int layer = 0; layer < fTopology.size()-1; layer++) {
        // update the bias grads.
        for(int neuron = 0; neuron < fTopology.at(layer+1); neuron++) {
            current_val = fBiasGradientMatrices.at(layer)->Element(neuron,0);
            update = fErrorMatrices.at(layer)->Element(neuron,0);           
            if(fVerbose) cout<<" -> layer: "<<layer<<", bias to neuron: "<<neuron<<" in layer: "<<layer+1<<", current_val: "<<current_val<<", update: "<<update;
            fBiasGradientMatrices.at(layer)->Element(neuron, 0, current_val + update);
            if(fVerbose) cout<<", new bias matrix grad.: "<<fBiasGradientMatrices.at(layer)->Element(neuron, 0)<<endl;
        }
        // update the matrix grads.
        for(int neuron = 0; neuron < fTopology.at(layer); neuron++) { 
            for(int weight = 0; weight < fTopology.at(layer+1); weight++) {
                current_val = fGradientMatrices.at(layer)->Element(weight, neuron);
                update = fErrorMatrices.at(layer)->Element(weight, 0) * fLayers.at(layer)->Neurons().at(neuron)->Activation();
                if(fVerbose) cout<<" -> layer: "<<layer<<", neuron: "<<neuron<<", weight to neuron: "<<weight<<"in layer:"<<layer+1<<", current_val: "<<current_val<<", update: "<<update;
                fGradientMatrices.at(layer)->Element(weight, neuron, current_val + update);
                if(fVerbose) cout<<", new matrix grad.: "<<fGradientMatrices.at(layer)->Element(weight, neuron)<<endl;
            }
        }
    }
}

void NeuralNetwork::UpdateNetwork() {
    // calculate the gradient by averaging the values added to it, 
    // then we update the weight matrices and biases accordingly
    
    if(fVerbose) cout<<" ---> updating the network..."<<endl;
    
    double current_val, update;
    for(int layer = 0; layer < fTopology.size()-1; layer++) {
        // update the biases
        for(int neuron = 0; neuron < fTopology.at(layer+1); neuron++) {
            current_val = fBiasMatrices.at(layer)->Element(neuron, 0);
            update = fLearningRate * fBiasGradientMatrices.at(layer)->Element(neuron, 0) / double(fBatchSize);
            if(fVerbose) cout<<" -> layer: "<<layer<<", bias to neuron: "<<neuron<<" in layer: "<<layer+1<<", current_val: "<<current_val<<", update: "<<update;
            fBiasMatrices.at(layer)->Element(neuron, 0, current_val - update);
            if(fVerbose) cout<<", new bias matrix element: "<<fBiasGradientMatrices.at(layer)->Element(neuron, 0)<<endl;
        }
        // update the matrices
        for(int neuron = 0; neuron < fTopology.at(layer); neuron++) { 
            for(int weight = 0; weight < fTopology.at(layer+1); weight++) {
                current_val = fMatrices.at(layer)->Element(weight, neuron);
                update = fLearningRate * fGradientMatrices.at(layer)->Element(weight, neuron) / double(fBatchSize);
                if(fVerbose) cout<<" -> layer: "<<layer<<", neuron: "<<neuron<<", weight to neuron: "<<weight<<"in layer:"<<layer+1<<", current_val: "<<current_val<<", update: "<<update;
                fMatrices.at(layer)->Element(weight, neuron, current_val - update);
                if(fVerbose) cout<<", new matrix element: "<<fGradientMatrices.at(layer)->Element(weight, neuron)<<endl;
            }
        }
    }
    
    fGradientMatrices.clear();
    fBiasGradientMatrices.clear();
}

void NeuralNetwork::SGD(vector <pair <vector<double>,vector<double> > > training_data, int batch_size, double learning_rate) {
    // stochastic gradient decent
    // - the first arguement is the training data, a vector of pairs of double vectors
    // - the size of the main vector is the total number of training examples
    // - each pair is a training example, with the first element in the pair being
    //   the input layer activations, and the second element being the target output
    // - the batch size is how many examples to use in each gradient computation
    // - the learning rate is the factor that we multiply the gradient with before
    //   modifying the elements of the matrices and biases

    fLearningRate = learning_rate;
    fBatchSize = batch_size;    

    int n_batches = training_data.size()/fBatchSize;
    pair< vector<double>,vector<double> > current_example;    
    Layer *input, *output;

    for(int batch=0; batch<n_batches; batch++) {
        fGradientMatrices.clear();
        fBiasGradientMatrices.clear();
        for(int layer=0; layer<nLayers(); layer++) {
            fGradientMatrices.push_back(new Matrix(fTopology.at(layer+1), fTopology.at(layer)));
            fBiasGradientMatrices.push_back(new Matrix(fTopology.at(layer+1), 1));
        }

        for(int example=0; example<batch_size; example++) {
            current_example = training_data.at(batch+example);
            
            // check that both elements of the pair have the correct number of neurons
            if(current_example.first.size()  != InputLayer()->nNeurons() || 
               current_example.second.size() != OutputLayer()->nNeurons() ) {
                cerr<<"training example doesn't have the correct number of neurons!"<<endl;
                cerr<<"input layer n_neurons, example: "<<current_example.first.size()<<" vs layer: "<<InputLayer()->nNeurons()<<endl;
                cerr<<"output layer n_neurons, example: "<<current_example.second.size()<<" vs layer: "<<OutputLayer()->nNeurons()<<endl;
                assert(false);
            }

            // calculate gradient for the current example
            this->InputLayer(current_example.first);
            this->TargetLayer(current_example.second);
            this->ForwardPropagate();
            this->BackwardPropagate();
            this->AddToGradient();

            // backward propagate calculates all the errors for us, but the 
            // gradients for the example are calculated in UpdateNetwork()
        }
            
        // now that the gradients hav been calculated, we update the network with the
        // calculated gradient averaged over the batch size
        this->UpdateNetwork();
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


