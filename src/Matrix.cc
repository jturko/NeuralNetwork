
#include "Matrix.hh"
#include "Utils.hh"

using namespace std;

Matrix::Matrix(int dx, int dy, bool random) {
    fX = dx;
    fY = dy;
    for(int i=0; i<dx; i++) {
        vector<double> tmp;
        for(int j=0; j<dy; j++) {
            if(random) tmp.push_back(Utils::Rndm());
            else tmp.push_back(0.0);
        }
        fElements.push_back(tmp);
    }
}

void Matrix::Element(int x, int y, double value) {
    fElements.at(x).at(y) = value;
}

double Matrix::Element(int x, int y) {
    return fElements.at(x).at(y);
}

Matrix Matrix::operator* (Matrix rhs) {
    if(Y() != rhs.X()) cerr<<"Incorrect matrix dims: lhs->Y() = "<<Y()<<"\t rhs->X() = "<<rhs.X()<<endl;
    
    double tmpElement;
    Matrix * m = new Matrix(X(),rhs.Y());
    for(int i=0; i<X(); i++) {
        for(int j=0; j<rhs.Y(); j++) {
            tmpElement = 0.0;
            for(int k=0; k<Y(); k++) tmpElement += Element(j,k)*rhs.Element(k,i);
            m->Element(j,i,tmpElement);
        }
    }
    
    return *m;
}

void Matrix::Print() {
    cout.precision(6);
    cout<<fixed;
    cout<<"---------------------------------"<<endl;
    cout<<"Matrix: "<<Y()<<" x "<<X()<<endl;
    for(int i=0; i<fX; i++) {
        for(int j=0; j<fY; j++) { 
            cout<<fElements.at(i).at(j)<<"\t";   
        }
        cout<<endl;
    }
    cout<<"---------------------------------"<<endl;
}

