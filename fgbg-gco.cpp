#include "approximateES.hpp"
#include <opencv2/opencv.hpp>
#include "GCoptimization.h"
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

class FgBgGCEnergyMinimizer: public EnergyMinimizer
{
    Mat img1;
    size_t number_of_vars; 
    int width, height, num_labels;
    GCoptimizationGridGraph *gc;
    float c, d;
    float bg;
    float fg;
    Mat img1F;

    public:
    FgBgGCEnergyMinimizer(const char* input_img1, float fg = 0.0f, float bg = 1.0f, float c = 1.0f, float d = 1.0f):fg(fg), bg(bg), c(c), d(d), gc(NULL)
    {
        img1 = imread(input_img1, 0);
        img1.convertTo( img1F, CV_32FC1);
        img1F = img1F / 255.0f;
        width = img1.cols;
        height = img1.rows;
        num_labels = 2;
        number_of_vars = (size_t) width*height;
        gc = new GCoptimizationGridGraph(width, height, num_labels); 
    }
    
    size_t getNumberOfVariables()
    {
        return number_of_vars;
    }

    virtual short_array minimize(short_array input, float lambda, float& energy, float &m, float& b)
    {
        short_array output(new short[number_of_vars]);
        try
        {
            cout<<"Lambda: "<<lambda<<endl;
            for( int y = 0; y < height; y++)
            {
                for(int x = 0; x < width; x++)
                {
                    gc->setLabel(y*width+x, (int) input[y*width+x] );// initialize labeling by input
                    for(int l = 0; l < num_labels; l++)
                    {
                        float data_ = fg;
                        if(l == 1) data_ = bg;
                        float cost = (img1F.at<float>(y,x) - data_)*(img1F.at<float>(y,x) - data_);
                        cout<<"cost @("<<x<<", "<<y<<") label: "<<l<<" is "<<cost<<endl;
                        gc->setDataCost(y*width + x, l, (c+lambda*d)*cost);
                    }
                }
            }
            cout<<"Smoothness: "<<endl;

            for(int l1 = 0; l1 < num_labels; l1++)
                for(int l2 = 0; l2 < num_labels; l2++)
                {
                    gc->setSmoothCost(l1,l2, fabs(l1 - l2));
                }

            cout<<"Before optimization: energy is "<<gc->compute_energy()<<endl;
            gc->expansion(5);
            cout<<"After optimization: energy is "<<gc->compute_energy()<<endl;
                        
            for(size_t i = 0; i < number_of_vars; i++)
            {
                output[i] = (short)gc->whatLabel(i);
            }

            m = d * gc->giveDataEnergy() / (c+lambda*d);
            b = c * gc->giveDataEnergy() / (c+lambda*d) + gc->giveSmoothEnergy();
            energy = m * lambda + b;
            cout<<"M = "<<m<<", B = "<<b<<endl;
        }
        catch( GCException e)
        {
            e.Report();
        }

        return output;
    }

        
    virtual ~FgBgGCEnergyMinimizer()
    {
        if(gc)
            delete gc;
    }
};



int main()
{
    
    FgBgGCEnergyMinimizer* e = new FgBgGCEnergyMinimizer("grays.png", 0.5f, 0.0f, 1.0f, 1.0f);
    ApproximateES aes(e->getNumberOfVariables(), 0.0,100.0, e, NULL, 200);
    aes.loop();
    delete e;
    return 0;
}
