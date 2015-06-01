/*
    Copyright (c) 2011, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "densecrf.h"
#include <cstdio>
#include <cmath>
#include "util.h"
#include <iostream>
#include "../../approximateES.hpp"
#include <vector>
#include <string>
#include <sstream>
#include <stdlib.h>


using namespace std;

#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
                        ( std::ostringstream() << std::dec << x ) ).str()


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
const float GT_PROB = 0.5;
float a = 0.0, b = 1.0, lambda = 0.9;


// Simple classifier that is 50% certain that the annotation is correct
float * classify( const unsigned char * im, int W, int H, int M , short* map){
	const float u_energy = -log( 1.0f / M );
	const float n_energy = -log( (1.0f - GT_PROB) / (M-1) );
	const float p_energy = -log( GT_PROB );
	float * res = new float[W*H*M];
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
		float * r = res + k*M;
		if (c){
			for( int j=0; j<M; j++ )
				r[j] = n_energy;
			r[i] = p_energy;
            map[k] = (short)i;
		}
		else{
			for( int j=0; j<M; j++ )
				r[j] = u_energy;
            map[k] = (short)(-1);
		}
	}
	return res;
}


class DenseEnergyMinimizer: public EnergyMinimizer
{
    int W;
    int H;
    int M;
    int N;
    unsigned char *im;
    unsigned char *anno;
    short *map;
    float *unary;
    float *u_result;
    float *p_result;
    DenseCRF2D* crf;
    float u_sum;
    float p_sum;
  public:
    DenseEnergyMinimizer(const char *im_path, const char *anno_path, int M):
        M(M),
        u_sum(0),
        p_sum(0),
        crf(NULL),
        map(NULL),
        unary(NULL),
        u_result(NULL),
        p_result(NULL),
        im(NULL),
        anno(NULL)
    {
        int GH, GW;
        im = readPPM( im_path, W, H );
        anno = readPPM( anno_path, GW, GH );
        if (!im){
            printf("Failed to load image!\n");
            exit(1);
        }

        if (!anno){
            printf("Failed to load annotations!\n");
            exit(1);
        }

        if (W!=GW || H!=GH){
            printf("Annotation size doesn't match image!\n");
            exit(1);
        }
        N = W*H;
        short * map = new short[N];
        float * unary = classify( anno, W, H, M , map);

        crf = new DenseCRF2D(W,H,M);
        crf->setUnaryEnergy( unary );
        crf->addPairwiseGaussian( 3, 3, 3 );
        crf->addPairwiseBilateral( 60, 60, 20, 20, 20, im, 10 );
        u_result = new float[N];
        p_result = new float[N];

        crf->unaryEnergy( map, u_result);
        crf->pairwiseEnergy(map, p_result, -1);
        for(int i = 0; i < N; ++i)
        {
            u_sum += u_result[i];
            p_sum += p_result[i];
        }
        std::cout<< "Unary sum: "<<u_sum <<endl;
        std::cout<< "Pairwise sum: "<<p_sum <<endl;
        std::cout<< "Total energy: "<< (p_sum + u_sum) << endl;

    }

    virtual short_array minimize(short_array input, double lambda, double& energy, double &m, double& b)
    {
        short_array output(new short[N]);
        float* l_unary = new float[N*M];
        for(int i = 0; i < N*M; i++)
            l_unary[i] = (float)lambda*unary[i];
        crf->setUnaryEnergy(l_unary);
        float b_energy = lambda * u_sum + p_sum;
        cout<<"Before optimization: energy is "<<b_energy<<endl;
        crf->map(3, map);
        crf->unaryEnergy( map, u_result);
        crf->pairwiseEnergy(map, p_result, -1);
        float n_u_sum = 0.0f, n_p_sum = 0.0f;
        for(int i = 0; i < N; ++i)
        {
            n_u_sum += u_result[i];
            n_p_sum += p_result[i];
            output[i] = map[i];
        }
        cout<< "Unary sum: "<<n_u_sum <<" = lambda*u = "<<lambda<<"* "<< u_sum<<" = "<<lambda*u_sum<<endl;
        cout<< "Pairwise sum: "<<n_p_sum <<endl;
        cout<< "After optimization: energy is "<< (n_p_sum + n_u_sum) << endl;
        m = u_sum;
        b = n_p_sum;
        energy = m * lambda + b;
        delete[] l_unary;
    }
    
    int getWidth(){return W;}
    int getHeight(){return H;}

    size_t getNumberOfVariables()
    {
        return (size_t)N;
    }

    virtual ~DenseEnergyMinimizer()
    {
        if(crf)
            delete crf;
        
        if(map)
            delete[] map;
        if(unary)
            delete[] unary;
        if(u_result)
            delete[] u_result;
        if(p_result)
            delete[] p_result;
        if(im)
            delete[] im;
        if(anno)
            delete[] anno;

    }

};
int main( int argc, char* argv[]){
	if (argc<4){
		printf("Usage: %s image annotations outputdir\n", argv[0] );
		return 1;
	}
    const int M = 3;
    DenseEnergyMinimizer *e = new DenseEnergyMinimizer(argv[1],argv[2],M);
			// Setup the CRF model
	ApproximateES aes(/* number of vars */ e->getNumberOfVariables(),/*lambda_min */ 0.0,/* lambda_max*/ 0.1, /* energy_minimizer */e,/* x0 */ NULL, /*max_iter */10000,/*verbosity*/ 10);
    aes.loop();
    vector<short_array> labelings = aes.getLabelings();
    string out_dir(argv[3]);
    string lambdas_file = out_dir + string("lambdas.txt");
    aes.writeLambdas(lambdas_file.c_str());
    
    for(size_t i = 0; i < labelings.size(); i++)
    {
        /*Mat m = Mat::zeros(e->getHeight(), e->getWidth(), CV_8UC3);
        for(int y = 0; y < e->getHeight(); y++)
        {
            for(int x = 0; x < e->getWidth(); x++)
            {
                if( labelings[i][ y*(e->getWidth()) + x] == 0 ) //fg
                {
                    m.at<Vec3b>(y,x) = Vec3b(255,255,255);
                }
            }
        }*/
        string out("_output.ppm");
        string s = out_dir + SSTR( i ) + out;
        unsigned char *res = colorize( labelings[i].get(), e->getWidth(), e->getHeight());
        writePPM( s.c_str(), e->getWidth(), e->getHeight(), res);
        delete[] res;
        //imwrite(s.c_str(), m);
    }

    delete e;
}
