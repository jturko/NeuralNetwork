
#ifndef UTILS_HH
#define UTILS_HH

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <random>

#include "Matrix.hh"

using namespace std;
    
namespace Utils {
    inline double Rndm() { return rand()/double(RAND_MAX); }
    inline double Rndm(double low, double high) { return (low+(high-low)*Rndm()); }
    inline double RndmGaus(double mean, double stddev) {//Box muller method
        static double n2 = 0.0;
        static int n2_cached = 0;
        if (!n2_cached)
        {
            double x, y, r;
            do
            {
                x = 2.0*rand()/RAND_MAX - 1;
                y = 2.0*rand()/RAND_MAX - 1;
    
                r = x*x + y*y;
            }
            while (r == 0.0 || r > 1.0);
            {
                double d = sqrt(-2.0*log(r)/r);
                double n1 = x*d;
                n2 = y*d;
                double result = n1*stddev + mean;
                n2_cached = 1;
                return result;
            }
        }
        else
        {
            n2_cached = 0;
            return n2*stddev + mean;
        }
    }

    inline Matrix * HadamardProduct(Matrix * m1, Matrix * m2) {
        if(m1->nRows() != m2->nRows() || m1->nCols() != m2->nCols()) {
            cerr<<"Wrong dimensions for Hadamard product - m1: "<<m1->nRows()<<"x"<<m1->nCols()<<endl;
            cerr<<"                                        m2: "<<m2->nRows()<<"x"<<m2->nCols()<<endl;
            assert(false);
            return NULL;
        }    
        
        Matrix * m = new Matrix(m1->nRows(), m1->nCols());
        for(int i=0; i<m1->nRows(); i++) {
            for(int j=0; j<m1->nCols(); j++) {
                m->Element(i,j,m1->Element(i,j)*m2->Element(i,j));
            }
        }
        
        return m;
    }
    
    inline Matrix * DotProduct(Matrix * m1, Matrix * m2) {
        if(m1->nCols() != m2->nRows()) {
            cerr<<"incorrect matrix dims: lhs: "<<m1->nRows()<<" x "<<m1->nCols()<<endl;
            cerr<<"                       rhs: "<<m2->nRows()<<" x "<<m2->nCols()<<endl;
            assert(false);
            return NULL;
        }

        Matrix * m = new Matrix(m1->nRows(),m2->nCols());
        for(int i=0; i<m1->nRows(); i++) {
            for(int j=0; j<m2->nCols(); j++) {
                double tmpElement = 0.0;
                for(int k=0; k<m1->nCols(); k++) tmpElement += m1->Element(i,k)*m2->Element(k,j);
                m->Element(i,j,tmpElement);
            }
        }
        
        return m;
    }

    inline Matrix * MatrixAdd(Matrix * m1, Matrix * m2) {
        if(m1->nCols() != m2->nCols() || m1->nRows() != m2->nRows()) {
            cerr<<"incorrect matrix dims: lhs: "<<m1->nRows()<<" x "<<m1->nCols()<<endl;
            cerr<<"                       rhs: "<<m2->nRows()<<" x "<<m2->nCols()<<endl;
            assert(false);
            return NULL;
        }

        Matrix * m = new Matrix(m1->nRows(),m1->nCols());
        for(int i=0; i<m1->nRows(); i++) {
            for(int j=0; j<m1->nCols(); j++) {
                m->Element(i,j,m1->Element(i,j)+m2->Element(i,j));
            }
        }

        return m;
        
    }
    
    inline Matrix * MatrixSubtract(Matrix * m1, Matrix * m2) {
        if(m1->nCols() != m2->nCols() || m1->nRows() != m2->nRows()) {
            cerr<<"incorrect matrix dims: lhs: "<<m1->nRows()<<" x "<<m1->nCols()<<endl;
            cerr<<"                       rhs: "<<m2->nRows()<<" x "<<m2->nCols()<<endl;
            assert(false);
            return NULL;
        }

        Matrix * m = new Matrix(m1->nRows(),m1->nCols());
        for(int i=0; i<m1->nRows(); i++) {
            for(int j=0; j<m1->nCols(); j++) {
                m->Element(i,j,m1->Element(i,j)-m2->Element(i,j));
            }
        }

        return m;
        
    }

    // code for reading MNIST data taken from: 
    // https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
    inline int reverseInt (int i) {
        unsigned char c1, c2, c3, c4;
    
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
    
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    }
    inline void read_mnist()
    {
        ifstream file ("/Users/JTurko/c++/NeuralNetwork/train-images-idx3-ubyte.gz");
        if (file.is_open())
        {
            int magic_number=0;
            int number_of_images=0;
            int n_rows=0;
            int n_cols=0;
            file.read((char*)&magic_number,sizeof(magic_number)); 
            magic_number= reverseInt(magic_number);
            file.read((char*)&number_of_images,sizeof(number_of_images));
            number_of_images= reverseInt(number_of_images);
            file.read((char*)&n_rows,sizeof(n_rows));
            n_rows= reverseInt(n_rows);
            file.read((char*)&n_cols,sizeof(n_cols));
            n_cols= reverseInt(n_cols);
            for(int i=0;i<number_of_images;++i)
            {
                for(int r=0;r<n_rows;++r)
                {
                    for(int c=0;c<n_cols;++c)
                    {
                        unsigned char temp=0;
                        file.read((char*)&temp,sizeof(temp));
    
                    }
                }
            }
        }
    }
    // end of code taken from:
    // https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
}

#endif

