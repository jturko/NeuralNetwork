
#include <iostream>

#include "Neuron.hh"
#include "Matrix.hh"
#include "Layer.hh"

using namespace std;

int main(int argc, char * argv[]) 
{
    srand((unsigned)time(NULL)); rand();
    cout<<"Running NeuralNetwork..."<<endl;
   
    Matrix m1(2,3);
    m1.Print();
    Matrix m2(3,4);
    m2.Print();
    Matrix m3 = m1*m2;
    m3.Print();

    return 0;
}

