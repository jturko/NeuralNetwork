
#include "Matrix.hh"
#include "Utils.hh"

using namespace std;

Matrix::Matrix(int nRows, int nCols, bool random) {
    fnRows = nRows;
    fnCols = nCols;
    for(int i=0; i<nRows; i++) {
        vector<double> tmp;
        for(int j=0; j<nCols; j++) {
            if(random) tmp.push_back(Utils::RndmGaus(0.,1.));
            else tmp.push_back(0.0);
        }
        fElements.push_back(tmp);
    }
}

void Matrix::Element(int row, int col, double value) {
    fElements.at(row).at(col) = value;
}

double Matrix::Element(int row, int col) {
    if(row < 0 || row >= nRows() || col < 0 || col >= nCols()) {
        cerr<<"Attempted to access matrix element that is out of range!"<<endl;
        cerr<<"Element: "<<row<<" x "<<col<<endl;
        cerr<<"Matrix dims: "<<nRows()<<" x "<<nCols()<<endl;
        assert(false);
    }
    return fElements.at(row).at(col);
}

Matrix Matrix::operator* (Matrix rhs) {
    if(nCols() != rhs.nRows()) { 
        cerr<<"Error in Matrix::opertor* (Matrix ):"<<endl;
        cerr<<"incorrect matrix dims: lhs: "<<nRows()<<" x "<<nCols()<<endl;
        cerr<<"                       rhs: "<<rhs.nRows()<<" x "<<rhs.nCols()<<endl;
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
        cerr<<"Error in Matrix::opertor+ (Matrix ):"<<endl;
        cerr<<"incorrect matrix dims: lhs: "<<nRows()<<" x "<<nCols()<<endl;
        cerr<<"                       rhs: "<<rhs.nRows()<<" x "<<rhs.nCols()<<endl;
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

Matrix Matrix::operator- (Matrix rhs) {
    if(nRows() != rhs.nRows() || nCols() != rhs.nCols()) { 
        cerr<<"Error in Matrix::opertor- (Matrix ):"<<endl;
        cerr<<"incorrect matrix dims: lhs: "<<nRows()<<" x "<<nCols()<<endl;
        cerr<<"                       rhs: "<<rhs.nRows()<<" x "<<rhs.nCols()<<endl;
        assert(false);
        return Matrix(0,0);
    }
    
    Matrix * m = new Matrix(nRows(),nCols());
    for(int i=0; i<nRows(); i++) {
        for(int j=0; j<nCols(); j++) {
            m->Element(i,j,Element(i,j)-rhs.Element(i,j));
        }
    }
    
    return *m;
}

Matrix * Matrix::Transpose() {
    Matrix * m = new Matrix(nCols(),nRows());
    for(int r=0; r<nRows(); r++) {
        for(int c=0; c<nCols(); c++) {
            m->Element(c,r,Element(r,c));
        }
    }
    
    return m;
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
    //cout<<"---------------------------------"<<endl;
}

