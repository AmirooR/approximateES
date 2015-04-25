#pragma once
#include <boost/shared_array.hpp>

typedef boost::shared_array<short> short_array;

class EnergyMinimizer
{
    public:
   virtual short_array minimize(short_array input, float lambda, float& energy, float &m, float& b) = 0;

   virtual ~EnergyMinimizer(){}
};


