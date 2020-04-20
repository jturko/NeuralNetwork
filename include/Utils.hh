
#ifndef UTILS_HH
#define UTILS_HH

#include<cmath>

namespace Utils {
    inline double Rndm() { return std::rand()/double(RAND_MAX); }
    inline double Rndm(double low, double high) { return (low+(high-low)*Rndm()); }
}

#endif

