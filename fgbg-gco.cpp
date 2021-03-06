#include "approximateES.hpp"
#include <opencv2/opencv.hpp>
#include "GCoptimization.h"
#include <vector>
#include <cmath>
#include <string>
#include <sstream>

#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
                ( std::ostringstream() << std::dec << x ) ).str()
using namespace std;
using namespace cv;

class FgBgGCEnergyMinimizer: public EnergyMinimizer
{
    Mat img1;
    size_t number_of_vars; 
    int width, height, num_labels;
    GCoptimizationGridGraph *gc;
    double c, d;
    double bg;
    double fg;
    Mat img1F;

    public:
    FgBgGCEnergyMinimizer(const char* input_img1, double fg = 0.0, double bg = 1.0, double c = 1.0, double d = 1.0):fg(fg), bg(bg), c(c), d(d), gc(NULL)
    {
        img1 = imread(input_img1, 0);
        img1.convertTo( img1F, CV_64FC1);
        img1F = img1F / 255.0;
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

    virtual short_array minimize(short_array input, double lambda, double& energy, double &m, double& b)
    {
        short_array output(new short[number_of_vars]);
        try
        {
            cout<<"Lambda: "<<lambda<<endl;
            for( int y = 0; y < height; y++)
            {
                for(int x = 0; x < width; x++)
                {
                    //gc->setLabel(y*width+x, (int) input[y*width+x] );// initialize labeling by input
                    gc->setLabel(y*width+x, 0);
                    for(int l = 0; l < num_labels; l++)
                    {
                        //double cost = fabs(img1F.at<double>(y,x) - data_);//*(img1F.at<double>(y,x) - data_);
                        double cost = fabs(img1F.at<double>(y,x)-fg);//fabs(img1F.at<double>(y,x)-fg);
                        if( l == 1 )
                        {
                            cost = fabs(img1F.at<double>(y,x) - (10.0/7.0) );//fabs(img1F.at<double>(y,x)-bg);
                            double cost2 = fabs(img1F.at<double>(y,x) + 2.0/7.0);
                            cost = cost < cost2 ? cost : cost2;
                            cost = cost * cost;
                        }
                        //cost = cost * cost;
                        //cout<<"cost @("<<x<<", "<<y<<") = "<< img1F.at<double>(y,x)<<" label: "<<l<<" is "<<cost<<endl;
                        gc->setDataCost(y*width + x, l, (c+lambda*d)*cost);
                    }
                }
            }

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

            cout<<"Data term: "<<gc->giveDataEnergy() << endl;
            cout<<"Smooth term: "<<gc->giveSmoothEnergy() <<endl;

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

    int getWidth(){return width;}
    int getHeight(){return height;}

        
    virtual ~FgBgGCEnergyMinimizer()
    {
        if(gc)
            delete gc;
    }
};



int main()
{
    
    FgBgGCEnergyMinimizer* e = new FgBgGCEnergyMinimizer("grays.png", 4.0/7.0, 0.784, 0.0,1.0);
    ApproximateES aes(e->getNumberOfVariables(), 0.001, 1.0, e, NULL, 200);
    aes.loop();
    vector<short_array> labelings = aes.getLabelings();
    for(size_t i = 0; i < labelings.size(); i++)
    {
        Mat m = Mat::zeros(e->getHeight(), e->getWidth(), CV_8UC3);
        for(int y = 0; y < e->getHeight(); y++)
        {
            for(int x = 0; x < e->getWidth(); x++)
            {
                if( labelings[i][ y*(e->getWidth()) + x] == 0 ) //fg
                {
                    m.at<Vec3b>(y,x) = Vec3b(255,255,255);
                }
            }
        }

        string out("_output.png");
        string s = SSTR( i ) + out;
        imwrite(s.c_str(), m);
    }

    delete e;
    return 0;
}
