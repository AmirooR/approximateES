#pragma once
#include <boost/shared_array.hpp>

typedef boost::shared_array<short> short_array;

class EnergyMinimizer
{
    public:
   virtual short_array minimize(short_array input, double lambda, double& energy, double &m, double& b) = 0;

   virtual ~EnergyMinimizer(){}
};


