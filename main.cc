
#include <iostream>

#include "Neuron.hh"
#include "Matrix.hh"

using namespace std;

int main(int argc, char * argv[]) 
{
    srand((unsigned)time(NULL)); rand();
    cout<<"Running NeuralNetwork..."<<endl;
    
    Matrix m1(1,3,true);
    Matrix m2(3,1,true);
    m1.Print(); 
    m2.Print();

    Matrix m3 = m1*m2;
    m3.Print();

    return 0;
}

