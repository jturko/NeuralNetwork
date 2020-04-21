
#ifndef UTILS_HH
#define UTILS_HH

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

namespace Utils {
    inline double Rndm() { return std::rand()/double(RAND_MAX); }
    inline double Rndm(double low, double high) { return (low+(high-low)*Rndm()); }

    // code for reading MNIST data taken from: 
    // https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
    inline int reverseInt (int i) 
    {
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

