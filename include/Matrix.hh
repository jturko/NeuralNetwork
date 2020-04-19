
#ifndef MATRIX_HH
#define MATRIX_HH

#include <iostream>
#include <vector>

using namespace std;

class Matrix
{
  public:
    Matrix(int dx, int dy, bool random=true);

    Matrix operator* (Matrix rhs);

    int X() { return fX; }
    void X(int x) { fX = x; }

    int Y() { return fY; }
    void Y(int y) { fY = y; }

    double Element(int x, int y);
    void Element(int x, int y, double value);

    void Print();

  private:
    int fX;
    int fY;
    vector< vector<double> > fElements;

};

#endif

