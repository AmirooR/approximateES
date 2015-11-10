#include "approximateES.hpp"
#include <opencv2/opencv.hpp>
#include "GCoptimization.h"
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <cstdlib>

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
    Vec3d color_diff3 = myData->img.at<Vec3d>(r1,c1) - myData->img.at<Vec3d>(r2,c2);
    double color_diff = pow(fabs(color_diff3[0]),2) + pow(fabs(color_diff3[1]),2) + pow(fabs(color_diff3[2]),2);
    
    return (myData->lambda1 + myData->lambda2*exp(-myData->beta*color_diff));
}
class StereoGCEnergyMinimizer: public EnergyMinimizer
{
    vector<Mat> diffs;
    Mat img1;
    Mat img2;
    Mat img1F;
    Mat img2F;
    size_t max_disp;
    size_t number_of_vars; 
    int width, height, num_labels;
    double smooth_mult;
    double lambda1, lambda2;
    double beta;
    GCoptimizationGridGraph *gc;
    double c, d;

    public:
    StereoGCEnergyMinimizer(const char* input_img1, const char* input_img2, size_t max_disp = 8, double c = 1.0, double d = 1.0, double smooth_mult = 5.0, double lambda1 = 0.0,double lambda2 = 10.0):
        max_disp(max_disp), 
        c(c), 
        d(d), 
        smooth_mult(smooth_mult), 
        lambda1(lambda1),
        lambda2(lambda2),
        gc(NULL)
    {
        img1 = imread(input_img1);
        img2 = imread(input_img2);

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
        computeBeta(); 
        cout<<"Constructed ... "<<endl;
    }
    
    protected:
    
    void computeBeta()
    {
        beta = 0.0;
        int count = 0;
        Mat hm1 = img1F;
        Mat hm2;
        pow( hm1.rowRange(1,height) - hm1.rowRange(0,height-1), 2, hm2);
        count += hm2.rows * hm2.cols*3;
        Scalar ss = sum(hm2);
        beta += ss[0]+ss[1]+ss[2];

        pow( hm1.colRange(1,width) - hm1.colRange(0,width-1), 2, hm2);
        count += hm2.rows * hm2.cols*3;
        ss = sum(hm2);
        beta += ss[0]+ss[1]+ss[2];
        double beta_inv = 2.0*(beta/count);
        if(beta_inv == 0)
        {
            beta_inv = 1e-10;//epsilon
        }
        beta = 1.0 / beta_inv;
        cout<<"Beta is: "<<beta<<endl;
    }

    public:
    int getWidth(){return width;}
    int getHeight(){return height;}
    
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

            /*for(int l1 = 0; l1 < num_labels; l1++)
                for(int l2 = 0; l2 < num_labels; l2++)
                {
                    gc->setSmoothCost(l1,l2, smooth_mult*fabs(l1 - l2));
                }*/
            ForSmoothFn toFn;
            toFn.img = img1F;
            toFn.lambda1 = lambda1;
            toFn.lambda2 = lambda2;
            toFn.width = width;
            toFn.height = height;
            toFn.beta = beta;
            gc->setSmoothCost(&smoothFn, &toFn);


            cout<<"Before optimization: energy is "<<gc->compute_energy()<<endl;
            gc->expansion(10);
            cout<<"After optimization: energy is "<<gc->compute_energy()<<endl;

            double sum_unaries = 0.0;
                        
            for(size_t i = 0; i < number_of_vars; i++)
            {
                output[i] = (short)gc->whatLabel(i);
                sum_unaries += diffs[output[i]].at<double>(i/width,i%width);
            }

            m = d * sum_unaries;//gc->giveDataEnergy() / (c+lambda*d);
            b = c * sum_unaries + gc->giveSmoothEnergy();//gc->giveDataEnergy() / (c+lambda*d) + gc->giveSmoothEnergy();

            energy = m * lambda + b;
            cout<<"M = "<<m<<", B = "<<b<<endl;
        }
        catch( GCException e)
        {
            e.Report();
        }

        return output;
    }

        
    void stereo_unaries_prev(Mat& img1, Mat& img2)
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

    void stereo_unaries_prev2(Mat& img1_f, Mat& img2_f)
    {
        for(size_t i = 0; i < max_disp; i++)
        {
            Mat diff, diff2, diff3;
            if( i == 0)
            {
                diff = img1_f - img2_f;
            }
            else
            {
                diff = img1_f.colRange(i, img1_f.cols) - img2_f.colRange(0, img2_f.cols - i);
            }

            diff  = diff.mul(diff);
            diff2 = diff.reshape(1, diff.cols*diff.rows);
            reduce(diff2, diff3, 1, CV_REDUCE_SUM);
            diff3 = diff3.reshape(1, diff.rows);

            if( i != max_disp - 1)
            {
                diff3 = diff3.colRange( max_disp - i - 1, diff3.cols);
            }

            diffs.push_back( diff3);

        }

    }
    
    void stereo_unaries(Mat& img1_f, Mat& img2_f)
    {
        for(size_t d=0; d < max_disp; d++)
        {
            Mat diff(img1_f.rows, img1_f.cols, CV_64F, Scalar(9000)); 

            for(int y = 0; y < img1_f.rows; y++)
            {
                for(int x = 0; x < img1_f.cols; x++)
                {
                    int xl = x;
                    int xr = xl - d;
                    if(xr >= 0 && xr < img1_f.cols)
                    {
                        Vec3d p_diff = img1_f.at<Vec3d>(y,xl) - img2_f.at<Vec3d>(y,xr);
                        diff.at<double>(y,x) = abs(p_diff[0]) + abs(p_diff[1]) + abs(p_diff[2]);
                    }
                }
            }
            diffs.push_back( diff);
        }
        /*for(size_t i = 0; i < max_disp; i++)
        {
            Mat diff, diff2, diff3;


            if( i == 0)
            {
                diff = img1_f - img2_f;
            }
            else
            {
                diff = img1_f.colRange(i, img1_f.cols) - img2_f.colRange(0, img2_f.cols - i);
            }

            diff  = diff.mul(diff);
            diff2 = diff.reshape(1, diff.cols*diff.rows);
            reduce(diff2, diff3, 1, CV_REDUCE_SUM);
            diff3 = diff3.reshape(1, diff.rows);

            if( i != max_disp - 1)
            {
                diff3 = diff3.colRange( max_disp - i - 1, diff3.cols);
            }

            diffs.push_back( diff3);

        }*/

    }
    
    virtual ~StereoGCEnergyMinimizer()
    {
        if(gc)
            delete gc;
    }
};



int main(int argc, char* argv[])
{
    if(argc < 7)
    {
        printf("Usage: %s left-image right-image output_folder/ num_disparity scale lambda\n", argv[0]);
        return 1;
    }
    int num_disp = atoi(argv[4]);
    int scale = atoi(argv[5]);
    StereoGCEnergyMinimizer* e = new StereoGCEnergyMinimizer(
            /* left */ argv[1], 
            /* right*/ argv[2],
            /* num disparities*/  num_disp, 
            /* c */0,
            /* d */ 1.0,
            /*smooth_mult*/ 1.0, 
            /*lambda1*/ 0.1,
            /*lambda2*/ 1.0);
   // ApproximateES aes(e->getNumberOfVariables(), 
   //         /*lambda_min*/ 0.0,
   //         /*lambda_max*/15.1, 
   //         /*minimizer*/e,
    //        /*x_0*/ NULL, 
     //       /* iterations */300, 
      //      /* verbosity */10);
    //aes.loop();
    double _e, _m, _b;
    double lambda = atof(argv[6]);
    short_array in(new short[e->getNumberOfVariables()]);
    for(size_t i=0; i < e->getNumberOfVariables(); i++)
        in[i] = 0;
    short_array x1 = e->minimize(in, lambda, _e, _m, _b);
    string out_dir(argv[3]);
    string out("_output.png");
    string s = out_dir + string(argv[6]) + out;
    string out_file = out_dir + string(argv[6]) + string(".txt");
    FILE *fp = fopen(out_file.c_str(), "w");
    fprintf(fp, "(lambda = %f) m = %f, b = %f, e = %f\n", lambda, _m, _b, _e);
    fclose(fp);
    
    Mat m = Mat::zeros(e->getHeight(), e->getWidth()/*+num_disp-1*/, CV_8UC3);
    for(int y = 0; y < e->getHeight(); y++)
    {
        for(int x = 0; x < e->getWidth(); x++)
        {
            int l_color = x1[ y*(e->getWidth()) + x]*scale;
            m.at<Vec3b>(y,x/*+num_disp-1*/) = Vec3b(l_color,l_color,l_color);
        }
    }
    imwrite(s.c_str(), m);  
    
    delete e;
    return 0;
}
