#include "../approximateES.hpp"

class TestEnergyMinimizer: public EnergyMinimizer
{
    //size_t N;
    public:
    
    TestEnergyMinimizer()
    {
    }
    
    virtual short_array minimize(short_array input, float lambda, float& energy, float &m, float& b)
    {
        short_array x( new Short[1] );
        if( lambda <= -0.5 )
        {
            m = 1;
            b = 1;
            energy = m*lambda + b;
            x[0] = Short(0);
        }
        else if( lambda <= 0.5)
        {
            m = 0;
            b = 0.5;
            energy = m*lambda + b;
            x[0] = Short(1);
        }
        else if( lambda <= 1)
        {
            m = -1;
            b = 1;
            energy = m*lambda + b;
            x[0] = Short(2);
        }
        return x;
    }
};
int main()
{
    TestEnergyMinimizer* e = new TestEnergyMinimizer;
    ApproximateES aes(1, -1, 1, e, NULL, 100);
    aes.loop();
    delete e;
}
