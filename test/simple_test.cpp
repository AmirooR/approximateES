#include "../approximateES.hpp"

int main()
{
    TestEnergyMinimizer* e = new TestEnergyMinimizer;
    ApproximateES aes(1, -1, 1, e, NULL, 100);
    aes.loop();
    delete e;
}
