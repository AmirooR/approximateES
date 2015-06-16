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
    
    return (myData->lambda1 + myData->lambda2*exp(-myData->beta*color_diff*color_diff*3.0));
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
    double *unaries;

  protected:
    
    void computeBeta()
    {
        beta = 0.0;
        int count = 0;
        Mat hm1 = img1F;
        Mat hm2;
        pow( hm1.rowRange(1,height) - hm1.rowRange(0,height-1), 2, hm2);
        count += hm2.rows * hm2.cols;
        beta += sum(hm2)[0];

        pow( hm1.colRange(1,width) - hm1.colRange(0,width-1), 2, hm2);
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
        gc(NULL),
        unaries(NULL)
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
        unaries = new double[width*height*num_labels];
        set_unaries();
    }
    
    size_t getNumberOfVariables()
    {
        return number_of_vars;
    }

    void set_unaries()
    {
        for( int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
            {
                for(int l = 0; l < num_labels; l++)
                {
                    double cost = fabs(img1F.at<double>(y,x)-fg);
                    if( l == 1 )
                    {
                        cost = fabs(img1F.at<double>(y,x)-bg);
                    }
                    unaries[ (y*width+x)*num_labels + l] = cost;
                }
            }
        }
    }

    virtual short_array minimize(short_array input, double lambda, double& energy, double &m, double& b)
    {
        short_array output(new short[number_of_vars]);
        try
        {
            //if(m_counter % 100 == 0 ) 
                cout<<"Lambda: "<<lambda<<endl;
            for( int y = 0; y < height; y++)
            {
                for(int x = 0; x < width; x++)
                {
                    gc->setLabel(y*width+x, 0);
                    double hv1 = unaries[(y*width+x)*num_labels+1] - unaries[(y*width+x)*num_labels];
                    for(int l = 0; l < num_labels; l++)
                    {
                        gc->setDataCost(y*width + x, l, (hv1+lambda)*l /*(c+lambda*d)*unaries[ (y*width+x)*num_labels + l]*/);
                    }
                }
            }

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

            double sum_unaries = 0.0;
            int num_one = 0;            
            for(size_t i = 0; i < number_of_vars; i++)
            {
                output[i] = (short)gc->whatLabel(i);
                if( output[i] == 1 )
                {
                    sum_unaries += (unaries[i*num_labels + 1] - unaries[i*num_labels]);
                    num_one++;
                }
            }

            if(m_counter % 100 == 0 ) 
            {
                cout<<"Data term: "<<gc->giveDataEnergy() << endl;
                cout<<"Sum unaries: "<<sum_unaries<<" +N(1)*lambda = "<<num_one*lambda+sum_unaries << endl;
                cout<<"Smooth term: "<<gc->giveSmoothEnergy() <<endl;
            }

            /*if( (c + lambda * d ) == 0 )
            {
                m = 0;
                b = 
            }
            else
            {
                m = d * gc->giveDataEnergy() / (c+lambda*d);
                b = c * gc->giveDataEnergy() / (c+lambda*d) + gc->giveSmoothEnergy();
                energy = m * lambda + b;
            }*/

            m = num_one;//sum_unaries;
            b = gc->giveSmoothEnergy()+sum_unaries; //gc->giveSmoothEnergy();
//            if(m_counter % 100 == 0 ) 
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

        if(unaries)
            delete[] unaries;
    }
};



int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        cout<<"Usage: "<<argv[0]<<" image output_dir"<<endl;
        return 1;
    }
    FgBgSVSGCEnergyMinimizer* e = new FgBgSVSGCEnergyMinimizer(argv[1], /*fg*/1.0, /*bg*/0.0, /*lambda1*/0.0, /*lambda2*/10.0, /*c*/0.0,/*d*/1.0 );
    
    ApproximateES aes(/* number of vars */ e->getNumberOfVariables(),/*lambda_min */ -1000.0,/* lambda_max*/ 1000.0, /* energy_minimizer */e,/* x0 */ NULL, /*max_iter */10000,/*verbosity*/ 10);
   
    aes.loop();
    vector<short_array> labelings = aes.getLabelings();
    string out_dir(argv[2]);
    string lambdas_file = out_dir + string("lambdas.txt");
    aes.writeLambdas(lambdas_file.c_str());
    
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
        string s = out_dir + SSTR( i ) + out;
        imwrite(s.c_str(), m);
    }

    delete e;
    return 0;
}
