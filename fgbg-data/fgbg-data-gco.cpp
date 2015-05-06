#include "../approximateES.hpp"
#include <opencv2/opencv.hpp>
#include "GCoptimization.h"
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <cstdio>
#include "util.h"

#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
                ( std::ostringstream() << std::dec << x ) ).str()
using namespace std;
using namespace cv;

// Store the colors we read, so that we can write them again.
int nColors = 0;
int colors[255];
unsigned int getColor( const unsigned char * c ){
	return c[0] + 256*c[1] + 256*256*c[2];
}
void putColor( unsigned char * c, unsigned int cc ){
	c[0] = cc&0xff; c[1] = (cc>>8)&0xff; c[2] = (cc>>16)&0xff;
}
// Produce a color image from a bunch of labels
unsigned char * colorize( const short * map, int W, int H ){
	unsigned char * r = new unsigned char[ W*H*3 ];
	for( int k=0; k<W*H; k++ ){
		int c = colors[ map[k] ];
		putColor( r+3*k, c );
	}
	return r;
}

// Certainty that the groundtruth is correct
//double a = 1.0, b = 1.0, lambda = 0.0;


// Simple classifier that is 50% certain that the annotation is correct
double * classify( const unsigned char * im, int W, int H, int M, double GT_PROB){
	const double u_energy = -log( 1.0f / M );
	const double n_energy = -log( (1.0f - GT_PROB) / (M-1) );
	const double p_energy = -log( GT_PROB );
	double * res = new double[W*H*M];
	for( int k=0; k<W*H; k++ ){
		// Map the color to a label
		int c = getColor( im + 3*k );
		int i;
		for( i=0;i<nColors && c!=colors[i]; i++ );
		if (c && i==nColors){
			if (i<M)
				colors[nColors++] = c;
			else
				c=0;
		}
		
		// Set the energy
		double * r = res + k*M;
		if (c){
			for( int j=0; j<M; j++ )
				r[j] = n_energy;
			r[i] = p_energy;
		}
		else{
			for( int j=0; j<M; j++ )
				r[j] = u_energy;
		}
	}
	return res;
}


class FgBgDataGCEnergyMinimizer: public EnergyMinimizer
{
    size_t number_of_vars; 
    int width, height, num_labels;
    GCoptimizationGridGraph *gc;
    double c, d;

    unsigned char* im;
    unsigned char* anno;
    double* unary;
    double GT_PROB;

    public:
    FgBgDataGCEnergyMinimizer(const char* input_im, const char* input_anno,double GT_PROB=0.5, double c = 1.0f, double d = 1.0f): GT_PROB(GT_PROB),c(c), d(d), gc(NULL)
    {
        // Number of labels
        // Load the color image and some crude annotations (which are used in a simple classifier)
        int W, H, GW, GH;
        num_labels = 21;
        unsigned char * im = readPPM( input_im, W, H );
        if (!im){
            printf("Failed to load image!\n");
        }
        unsigned char * anno = readPPM( input_anno, GW, GH );
        if (!anno){
            printf("Failed to load annotations!\n");
        }
        if (W!=GW || H!=GH){
            printf("Annotation size doesn't match image!\n");
        }

        width = W;
        height = H;

        /////////// Put your own unary classifier here! ///////////
        unary = classify( anno, W, H, num_labels, GT_PROB);

        number_of_vars = (size_t) width*height;
        gc = new GCoptimizationGridGraph(width, height, num_labels); 
    }

    int getNumLabels()
    {
        return num_labels;
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
                        double cost = unary[ ((y*width)+x)*num_labels + l];
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

        
    virtual ~FgBgDataGCEnergyMinimizer()
    {
        if(gc)
            delete gc;
        delete[] im;
        delete[] anno;
        delete[] unary;
    }
};



int main(int argc, char* argv[])
{
 	if (argc<3){
		printf("Usage: %s image annotations output\n", argv[0] );
		return 1;
	}
	   
    FgBgDataGCEnergyMinimizer* e = new FgBgDataGCEnergyMinimizer(argv[1], argv[2], .7f, 0.0f, 1.0f);
    ApproximateES aes(e->getNumberOfVariables(), 0.001, 10.0, e, NULL, 200);
    aes.loop();
    vector<short_array> labelings = aes.getLabelings();
    for(size_t i = 0; i < labelings.size(); i++)
    {
        unsigned char *res = colorize( labelings[i].get(), e->getWidth(), e->getHeight() );
        string out("_data_output.ppm");
        string s = SSTR( i ) + out;
        writePPM( s.c_str(), e->getWidth(), e->getHeight(), res);
        delete[] res;
    }

    cout<<nColors<<endl;

    delete e;
    return 0;
}
