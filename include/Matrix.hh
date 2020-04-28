
#ifndef MATRIX_HH
#define MATRIX_HH

#include <iostream>
#include <vector>

using namespace std;

class Matrix
{
  public:
    Matrix(int rows, int cols, bool random=false);

    Matrix operator* (Matrix rhs);
    Matrix operator+ (Matrix rhs);
    Matrix operator- (Matrix rhs);

    int nRows() { return fnRows; }
    void nRows(int nRows) { fnRows = nRows; }

    int nCols() { return fnCols; }
    void nCols(int nCols) { fnCols = nCols; }

    double Element(int row, int col);
    void Element(int row, int col, double value);

    Matrix * Transpose();

    void Print();

  private:
    int fnRows;
    int fnCols;
    vector< vector<double> > fElements;

};

#endif

