
#ifndef UTILS_HH
#define UTILS_HH

#include<cmath>

namespace Utils {
    double Rndm() { return std::rand()/double(RAND_MAX); }
    double Rndm(double low, double high) { return (low+(high-low)*random()); }
}

#endif

