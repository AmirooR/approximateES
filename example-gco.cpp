#include "approximateES.hpp"
#include <opencv2/opencv.hpp>
#include "GCoptimization.h"
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

size_t max_disp = 8;
float a = 100.0f;
float b = 100.0f;
float lambda_min = -1.0f;
float lambda_max = 1.0f;

vector<Mat> stereo_unaries(Mat& img1, Mat& img2)
{
//    assert(img1.type() == img2.type() == CV_32FC3 );
    vector<Mat> differences;
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

        differences.push_back( diff3);

    }

    return differences;
}



void potts_example()
{
    Mat img1 = imread("scene1.row3.col2.ppm");
    Mat img2 = imread("scene1.row3.col1.ppm");
    Mat img1F, img2F;
    img1.convertTo( img1F, CV_32FC3);
    img2.convertTo( img2F, CV_32FC3);
    img1F = img1F / 255.0f;
    img2F = img2F / 255.0f;
    vector<Mat> diffs = stereo_unaries( img1F, img2F);
    int width = diffs[0].cols;
    int height = diffs[0].rows;
    int num_labels = diffs.size();
    try
    {
        for(float lambda = lambda_min; lambda <= lambda_max; lambda += 0.1)
        {
            cout<<"Lambda: "<<lambda<<endl;
            GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width, height, num_labels); 
            for( int y = 0; y < height; y++)
            {
                for(int x = 0; x < width; x++)
                {
                    for(int l = 0; l < num_labels; l++)
                    {
                        gc->setDataCost(y*width + x, l, (a+lambda*b)*diffs[l].at<float>(y,x));
                    }
                }
            }

            for(int l1 = 0; l1 < num_labels; l1++)
                for(int l2 = 0; l2 < num_labels; l2++)
                {
                    gc->setSmoothCost(l1,l2,5.0f*fabs(l1 - l2));
                }

            cout<<"Before optimization: energy is "<<gc->compute_energy()<<endl;
            gc->expansion(2);
            cout<<"After optimization: energy is "<<gc->compute_energy()<<endl;

            delete gc;
        }
    }
    catch( GCException e)
    {
        e.Report();
    }

}

int main()
{
    potts_example();
    return 0;
}
