#pragma once

#include "Short.hpp"

class EnergyMinimizer
{
    public:
   virtual short_array minimize(short_array input, float lambda, float& energy, float &m, float& b) = 0;
};


