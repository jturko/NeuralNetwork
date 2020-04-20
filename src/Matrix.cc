
#include "Matrix.hh"
#include "Utils.hh"

using namespace std;

Matrix::Matrix(int nRows, int nCols, bool random) {
    fnRows = nRows;
    fnCols = nCols;
    for(int i=0; i<nRows; i++) {
        vector<double> tmp;
        for(int j=0; j<nCols; j++) {
            if(random) tmp.push_back(Utils::Rndm(-1.,1.));
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
        cerr<<"Incorrect matrix dims: lhs->nCols() = "<<nCols()<<"\t rhs->nRows() = "<<rhs.nRows()<<endl;
        assert(false);
        return Matrix(0,0);
    }
    
    double tmpElement;
    Matrix * m = new Matrix(nRows(),rhs.nCols());
    for(int i=0; i<nRows(); i++) {
        for(int j=0; j<rhs.nCols(); j++) {
            tmpElement = 0.0;
            for(int k=0; k<nCols(); k++) tmpElement += Element(i,k)*rhs.Element(k,j);
            m->Element(i,j,tmpElement);
        }
    }
    
    return *m;
}

Matrix Matrix::operator+ (Matrix rhs) {
    if(nRows() != rhs.nRows() || nCols() != rhs.nCols()) { 
        cerr<<"Incorrect matrix dims: lhs->nRows() = "<<nRows()<<"\t rhs->nRows() = "<<rhs.nRows()<<endl;
        cerr<<"                       lhs->nCols() = "<<nCols()<<"\t rhs->nCols() = "<<rhs.nCols()<<endl;
        assert(false);
        return Matrix(0,0);
    }
    
    Matrix * m = new Matrix(nRows(),nCols());
    for(int i=0; i<nRows(); i++) {
        for(int j=0; j<nCols(); j++) {
            m->Element(i,j,Element(i,j)+rhs.Element(i,j));
        }
    }
    
    return *m;
}

void Matrix::Print() {
    cout.precision(4);
    cout<<fixed;
    cout<<"--------- Matrix: "<<nRows()<<" x "<<nCols()<<" ---------"<<endl;
    for(int i=0; i<nRows(); i++) {
        for(int j=0; j<nCols(); j++) { 
            cout<<fElements.at(i).at(j)<<"\t";   
        }
        cout<<endl;
    }
    cout<<"---------------------------------"<<endl;
}

