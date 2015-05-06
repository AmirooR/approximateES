#include "approximateES.hpp"
#include <opencv2/opencv.hpp>
#include "GCoptimization.h"
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

class StereoGCEnergyMinimizer: public EnergyMinimizer
{
    vector<Mat> diffs;
    Mat img1;
    Mat img2;
    size_t max_disp;
    size_t number_of_vars; 
    int width, height, num_labels;
    double smooth_mult;
    GCoptimizationGridGraph *gc;
    double c, d;

    public:
    StereoGCEnergyMinimizer(const char* input_img1, const char* input_img2, size_t max_disp = 8, double c = 1.0, double d = 1.0, double smooth_mult = 5.0):max_disp(max_disp), c(c), d(d), smooth_mult(smooth_mult), gc(NULL)
    {
        img1 = imread(input_img1);
        img2 = imread(input_img2);

        Mat img1F, img2F;
        img1.convertTo( img1F, CV_64FC3);
        img2.convertTo( img2F, CV_64FC3);
        img1F = img1F / 255.0;
        img2F = img2F / 255.0;
        stereo_unaries( img1F, img2F);
        width = diffs[0].cols;
        height = diffs[0].rows;
        num_labels = diffs.size();
        number_of_vars = (size_t) width*height;
        gc = new GCoptimizationGridGraph(width, height, num_labels); 
        cout<<"Constructed ... "<<endl;
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
                    //cout<<"Setting label ("<<x<<", "<<y<<"): "<< input[y*width+x] << endl;
                    gc->setLabel(y*width+x, (int) input[y*width+x] );// initialize labeling by input
                    for(int l = 0; l < num_labels; l++)
                    {
                        gc->setDataCost(y*width + x, l, (c+lambda*d)*diffs[l].at<double>(y,x));
                    }
                }
            }
            cout<<"Smoothness: "<<endl;

            for(int l1 = 0; l1 < num_labels; l1++)
                for(int l2 = 0; l2 < num_labels; l2++)
                {
                    gc->setSmoothCost(l1,l2, smooth_mult*fabs(l1 - l2));
                }

            cout<<"Before optimization: energy is "<<gc->compute_energy()<<endl;
            gc->expansion(10);
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

        
    void stereo_unaries(Mat& img1, Mat& img2)
    {
        for(size_t i = 0; i < max_disp; i++)
        {
            Mat diff, diff2, diff3;
            if( i == 0)
            {
                diff = img1 - img2;
            }
            else
            {
                diff = img1.colRange(2*i, img1.cols) - img2.colRange(0, img2.cols - 2*i);
            }

            diff  = diff.mul(diff);
            diff2 = diff.reshape(1, diff.cols*diff.rows);
            reduce(diff2, diff3, 1, CV_REDUCE_SUM);
            diff3 = diff3.reshape(1, diff.rows);

            if( i != max_disp - 1)
            {
                diff3 = diff3.colRange( max_disp - i - 1, diff3.cols - max_disp + i + 1 );
            }

            diffs.push_back( diff3);

        }

    }

    virtual ~StereoGCEnergyMinimizer()
    {
        if(gc)
            delete gc;
    }
};



int main()
{
    StereoGCEnergyMinimizer* e = new StereoGCEnergyMinimizer("scene1.row3.col1.ppm", "scene1.row3.col2.ppm",  8, 100.0, 1.0, 5.0);
    ApproximateES aes(e->getNumberOfVariables(), -10.0,10.0, e, NULL, 200);
    aes.loop();
    delete e;
    return 0;
}
