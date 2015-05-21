#include "../approximateES.hpp"
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

struct ForSmoothFn{
    int width;
    int height;
    double lambda1;
    double lambda2;
    double beta;
    Mat img;
};

double smoothFn(int p1, int p2, int l1, int l2, void* data)
{
    if( l1 == l2)
        return 0;
    ForSmoothFn *myData = (ForSmoothFn *)data;

    int r1 = p1/myData->width;
    int r2 = p2/myData->width;
    int c1 = p1%myData->width;
    int c2 = p2%myData->width;

    double color_diff = fabs(myData->img.at<double>(r1,c1) - myData->img.at<double>(r2,c2));
    
    return (myData->lambda1 + myData->lambda2*exp(-myData->beta*color_diff*color_diff));
}

class FgBgSVSGCEnergyMinimizer: public EnergyMinimizer
{
    Mat img1;
    size_t number_of_vars; 
    int width, height, num_labels;
    GCoptimizationGridGraph *gc;
    double c, d;
    double bg;
    double fg;
    Mat img1F;
    double lambda1;
    double lambda2;
    double beta;
    int m_counter;

  protected:
    
    void computeBeta()
    {
        beta = 0.0;
        int count = 0;
        Mat hm1 = img1F;
        Mat hm2;
        pow( hm1.rowRange(2,height) - hm1.rowRange(1,height-1), 2, hm2);
        count += hm2.rows * hm2.cols;
        beta += sum(hm2)[0];

        pow( hm1.colRange(2,width) - hm1.colRange(1,width-1), 2, hm2);
        count += hm2.rows * hm2.cols;
        beta += sum(hm2)[0];
        double beta_inv = 2.0*(beta/count);
        if(beta_inv == 0)
        {
            beta_inv = 1e-10;//epsilon
        }
        beta = 1.0 / beta_inv;
        cout<<"Beta is: "<<beta<<endl;
    }

  public:
    FgBgSVSGCEnergyMinimizer(const char* input_img1, double fg = 1.0, double bg = 0.0, double lambda1 = 0.0, double lambda2 = 10.0, double c = 0.0, double d = 1.0):
        fg(fg), 
        bg(bg), 
        lambda1(lambda1), 
        lambda2(lambda2), 
        c(c), 
        d(d),
        beta(1.0),
        m_counter(0), 
        gc(NULL)
    {
        img1 = imread(input_img1, 0);
        img1.convertTo( img1F, CV_64FC1);
        img1F = img1F / 255.0;
        width = img1.cols;
        height = img1.rows;
        num_labels = 2;
        number_of_vars = (size_t) width*height;
        gc = new GCoptimizationGridGraph(width, height, num_labels);
        computeBeta();
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
            if(m_counter % 100 == 0 ) 
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
                            //cost = fabs(img1F.at<double>(y,x) - (10.0/7.0) );//
                            cost = fabs(img1F.at<double>(y,x)-bg);
                            /*double cost2 = fabs(img1F.at<double>(y,x) + 2.0/7.0);
                            cost = cost < cost2 ? cost : cost2;
                            cost = cost * cost;*/
                        }
                        //cost = cost * cost;
                        //cout<<"cost @("<<x<<", "<<y<<") = "<< img1F.at<double>(y,x)<<" label: "<<l<<" is "<<cost<<endl;
                        gc->setDataCost(y*width + x, l, (c+lambda*d)*cost);
                    }
                }
            }

            /*for(int l1 = 0; l1 < num_labels; l1++)
                for(int l2 = 0; l2 < num_labels; l2++)
                {
                    gc->setSmoothCost(l1,l2, fabs(l1 - l2));
                }*/
            ForSmoothFn toFn;
            toFn.img = img1F;
            toFn.lambda1 = lambda1;
            toFn.lambda2 = lambda2;
            toFn.width = width;
            toFn.height = height;
            toFn.beta = beta;
            gc->setSmoothCost(&smoothFn, &toFn);

            if(m_counter % 100 == 0 ) 
                cout<<"Before optimization: energy is "<<gc->compute_energy()<<endl;
            gc->expansion(1);
            if(m_counter % 100 == 0 )
                cout<<"After optimization: energy is "<<gc->compute_energy()<<endl;
                        
            for(size_t i = 0; i < number_of_vars; i++)
            {
                output[i] = (short)gc->whatLabel(i);
            }

            if(m_counter % 100 == 0 ) 
            {
                cout<<"Data term: "<<gc->giveDataEnergy() << endl;
                cout<<"Smooth term: "<<gc->giveSmoothEnergy() <<endl;
            }

            m = d * gc->giveDataEnergy() / (c+lambda*d);
            b = c * gc->giveDataEnergy() / (c+lambda*d) + gc->giveSmoothEnergy();
            energy = m * lambda + b;
            if(m_counter % 100 == 0 ) 
                cout<<"M = "<<m<<", B = "<<b<<endl;
        }
        catch( GCException e)
        {
            e.Report();
        }

        m_counter++;
        return output;
    }

    int getWidth(){return width;}
    int getHeight(){return height;}

        
    virtual ~FgBgSVSGCEnergyMinimizer()
    {
        if(gc)
            delete gc;
    }
};



int main()
{
    FgBgSVSGCEnergyMinimizer* e = new FgBgSVSGCEnergyMinimizer("grays.jpg", /*fg*/1.0, /*bg*/0.0, /*lambda1*/0.0, /*lambda2*/10.0, /*c*/0.0,/*d*/1.0 );
    
    ApproximateES aes(/* number of vars */ e->getNumberOfVariables(),/*lambda_min */ 0.001,/* lambda_max*/ 100.0, /* energy_minimizer */e,/* x0 */ NULL, /*max_iter */10000,/*verbosity*/ 0);
   
    aes.loop();
    vector<short_array> labelings = aes.getLabelings();
    aes.writeLambdas("lambdas.txt");
    
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
