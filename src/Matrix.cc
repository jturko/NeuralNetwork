
#include "Matrix.hh"
#include "Utils.hh"

using namespace std;

Matrix::Matrix(int nRows, int nCols, bool random) {
    fnRows = nRows;
    fnCols = nCols;
    for(int i=0; i<nRows; i++) {
        vector<double> tmp;
        for(int j=0; j<nCols; j++) {
            if(random) tmp.push_back(Utils::Rndm());
            else tmp.push_back(0.0);
        }
        fElements.push_back(tmp);
    }
}

void Matrix::Element(int row, int col, double value) {
    fElements.at(row).at(col) = value;
}

double Matrix::Element(int row, int col) {
    return fElements.at(row).at(col);
}

Matrix Matrix::operator* (Matrix rhs) {
    if(nCols() != rhs.nRows()) { 
        cerr<<"Incorrect matrix dims: lhs->nRows() = "<<nRows()<<"\t rhs->nCols() = "<<rhs.nCols()<<endl;
        return Matrix(0,0);
    }
    
    double tmpElement;
    Matrix * m = new Matrix(nRows(),rhs.nCols());
    for(int i=0; i<nRows(); i++) {
        for(int j=0; j<rhs.nCols(); j++) {
            cout<<"Calculating element: "<<i<<" x "<<j<<endl;
            tmpElement = 0.0;
            for(int k=0; k<nCols(); k++) { 
                if(k != 0) cout<<" + ";
                cout<<"("<<Element(i,k)<<flush<<" * "<<rhs.Element(k,j)<<flush<<")";
                tmpElement += Element(i,k)*rhs.Element(k,j);
            }
            cout<<" = "<<tmpElement<<endl;
            m->Element(i,j,tmpElement);
        }
    }
    
    return *m;
}

void Matrix::Print() {
    cout.precision(4);
    cout<<fixed;
    cout<<"---------------------------------"<<endl;
    cout<<"Matrix: "<<nRows()<<" x "<<nCols()<<endl;
    for(int i=0; i<nRows(); i++) {
        for(int j=0; j<nCols(); j++) { 
            cout<<fElements.at(i).at(j)<<"\t";   
        }
        cout<<endl;
    }
    cout<<"---------------------------------"<<endl;
}

