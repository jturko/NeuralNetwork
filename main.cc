
#include <iostream>

#include "Neuron.hh"
#include "Matrix.hh"
#include "Layer.hh"

using namespace std;

int main(int argc, char * argv[]) 
{
    srand((unsigned)time(NULL)); rand();
    cout<<"Running NeuralNetwork..."<<endl;
   
    Matrix m1(1,3,true);
    Matrix m2(3,1,true);
    Matrix m3 = m1*m2;
    m1.Print(); 
    m2.Print();
    m3.Print();

    Layer l1(3);
    l1.Neurons().at(0)->ActivationRaw(10);
    l1.Neurons().at(1)->ActivationRaw(20);
    l1.Neurons().at(2)->ActivationRaw(30);
    Matrix m4 = (*l1.RowVector());
    m4.Print();
    Matrix m5 = (*l1.ColumnVector());
    m5.Print();

    return 0;
}

