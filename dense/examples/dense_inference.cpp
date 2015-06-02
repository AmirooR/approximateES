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
#include <cstdlib>
#include <cstring>

#define NO_NORMALIZATION 0
#define MEAN_NORMALIZATION 1
#define PIXEL_NORMALIZATION 2
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
unsigned char * colorize( short* map, int W, int H ){
	unsigned char * r = new unsigned char[ W*H*3 ];
	for( int k=0; k<W*H; k++ ){
		int c = colors[ map[k] ];
		putColor( r+3*k, c );
	}
	return r;
}

// Certainty that the groundtruth is correct
const float GT_PROB = 0.75;


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
            map[k] = (short)(0);
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
    float *l_unary;
    float *u_result;
    float *p_result;
    float *init_x;
    DenseCRF2D* crf;
    double u_sum;
    double p_sum;
    bool approximate_pairwise;
    bool do_initialization;
    int do_normalization;
    double gsx,gsy,gw;
    double bsx,bsy,bsr,bsg,bsb,bw;
    double *norms;
    double mean_norm;

  public:
    DenseEnergyMinimizer(const char *im_path, const char *anno_path, int M, 
            int do_normalization = 0,
            bool do_initialization = true,
            bool approximate_pairwise=false,
            double gsx = 3.f, double gsy = 3.f, double gw=3.f,
            double bsx = 60.f, double bsy = 60.f, double bsr=20.f, double bsg=20.f, double bsb=20.f, double bw=10.f
            ):
        M(M),
        u_sum(0),
        p_sum(0),
        crf(NULL),
        map(NULL),
        unary(NULL),
        l_unary(NULL),
        u_result(NULL),
        init_x(NULL),
        p_result(NULL),
        im(NULL),
        anno(NULL),
        norms(NULL),
        approximate_pairwise(approximate_pairwise),
        do_initialization(do_initialization),
        do_normalization(do_normalization),
        gsx(gsx), gsy(gsy), gw(gw),
        bsx(bsx), bsy(bsy), bsr(bsr), bsg(bsg), bsb(bsb), bw(bw)
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
        map = new short[N];
        unary = classify( anno, W, H, M , map);
        init_x = new float[N*M];
        memcpy( init_x, unary, N*M*sizeof(float) );
        l_unary = new float[N*M];
        norms = new double[N];

        crf = new DenseCRF2D(W,H,M);
        crf->setUnaryEnergy( unary );
        crf->setInitX( init_x);
        crf->addPairwiseGaussian( gsx, gsy, gw );
        crf->addPairwiseBilateral( bsx, bsy, bsr, bsg, bsb, im, bw );
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
        if( do_normalization > 0 )
        {
            cout<<"Computing norms"<<endl;
            compute_norms();
        }

    }

    short* get_map(){return map;}

    void make_log_probability_x(short *input_map)
    {
        const float n_energy = -log( (1.0f - GT_PROB) / (M-1) );
        const float p_energy = -log( GT_PROB );
        for( int k=0; k<W*H; k++ )
        {
            // Set the energy
            float * r = init_x + k*M;
            for( int j=0; j<M; j++ )
                r[j] = n_energy;
            r[input_map[k]] = p_energy;
        }
    }

    virtual short_array minimize(short_array input, double lambda, double& energy, double &m, double& b)
    {
        cout<<"Minimizing Lambda = "<<lambda<<endl;
        short_array output(new short[N]);
        if( do_initialization)
        {
            make_log_probability_x(input.get());
            crf->setInitX(init_x);
        }
        for(int i = 0; i < N*M; i++)
            l_unary[i] = lambda*unary[i];
        crf->setUnaryEnergy(l_unary);
        if(approximate_pairwise)
        {
            double b_energy = lambda * u_sum + p_sum;
            cout<<"Before optimization: approximate energy is "<<b_energy<<endl;
            crf->unaryEnergy( map, u_result);
            crf->pairwiseEnergy(map, p_result, -1);
            float n_u_sum = 0.0f, n_p_sum = 0.0f;
            for(int i = 0; i < N; ++i)
            {
                n_u_sum += u_result[i];
                n_p_sum += p_result[i];
            }
            cout<< "Unary sum: "<<n_u_sum <<" = lambda*u = "<<lambda<<"* "<< u_sum<<" = "<<lambda*u_sum<<endl;
            cout<< "Pairwise sum: "<<n_p_sum <<endl;
            crf->map(3, map);
            cout<<"AFTER ***"<<endl;

            crf->unaryEnergy( map, u_result);
            crf->pairwiseEnergy(map, p_result, -1);
            n_u_sum = 0.0f, n_p_sum = 0.0f;
            u_sum = 0;
            for(int i = 0; i < N; ++i)
            {
                n_u_sum += u_result[i];
                n_p_sum += p_result[i];
                output[i] = map[i];
                u_sum += unary[ i*M+map[i]];
            }
            cout<< "Unary sum: "<<n_u_sum <<" = lambda*u = "<<lambda<<"* "<< u_sum<<" = "<<lambda*u_sum<<endl;
            cout<< "Pairwise sum: "<<n_p_sum <<endl;
            cout<< "After optimization: energy is "<< (n_p_sum + n_u_sum) << endl;
            m = u_sum;
            b = n_p_sum;
            energy = m * lambda + b;
        }
        else//computing exact pairwise
        {
            crf->map(3, map);
            cout<<"AFTER ***"<<endl;
            crf->unaryEnergy( map, u_result);
            //crf->pairwiseEnergy(map, p_result, -1);
            double n_p_sum = compute_pairwise_energy();
            double n_u_sum = 0.0;
            u_sum = 0;
            for(int i = 0; i < N; ++i)
            {
                n_u_sum += u_result[i];
                //n_p_sum += p_result[i];
                output[i] = map[i];
                u_sum += unary[ i*M+map[i]];
            }
            cout<< "Unary sum: "<<n_u_sum <<" = lambda*u = "<<lambda<<"* "<< u_sum<<" = "<<lambda*u_sum<<endl;
            cout<< "Pairwise sum: "<<n_p_sum <<endl;
            cout<< "After optimization: energy is "<< (n_p_sum + n_u_sum) << endl;
            m = u_sum;
            b = n_p_sum;
            energy = m * lambda + b;
        }
        return output;
    }
    
    int getWidth(){return W;}
    int getHeight(){return H;}

    size_t getNumberOfVariables()
    {
        return (size_t)N;
    }

    void compute_norms()
    {
        mean_norm = 0.0;
        for(int k=0; k < N; k++)
        {
            double this_sum = 0.0;
#pragma omp parallel for reduction(+:this_sum)                    
            for(int  k2=0; k2 < N;  k2++)
            {
                double d_e = 0;
                int dx = (k%W) - (k2%W);//i - i2;
                int dy = (k/W) - (k2/W);//j - j2;
                int dr = im[k*3+0]-im[k2*3+0];
                int dg = im[k*3+1]-im[k2*3+1];
                int db = im[k*3+2]-im[k2*3+2];
                d_e = bw*exp(-0.5 * ( (dx*dx)/(bsx*bsx) + (dy*dy)/(bsy*bsy) + 
                            (dr*dr)/(bsr*bsr) + (dg*dg)/(bsg*bsg) + (db*db)/(bsb*bsb) ) );
                d_e += gw*exp(-0.5 * ( (dx*dx)/(gsx*gsx) + (dy*dy)/(gsy*gsy) ) );
                this_sum += d_e;
            }
            norms[k] = this_sum;
            mean_norm += this_sum;
        }
        mean_norm = mean_norm/N;
    }

    double compute_pairwise_energy()
    {
        double sum_e = 0.0f;
        for(int k=0; k < N; k++)
        {
#pragma omp parallel for reduction(-:sum_e)                    
            for(int  k2=0; k2 < k;  k2++)
            {
                double d_e = 0;
                if( map[k] == map[k2] )
                {
                    /*int j = k/W;
                    int i = k%W;
                    int j2 = k2/W;
                    int i2 = k2%W;*/
                    int dx = (k%W) - (k2%W);//i - i2;
                    int dy = (k/W) - (k2/W);//j - j2;
                    int dr = im[k*3+0]-im[k2*3+0];
                    int dg = im[k*3+1]-im[k2*3+1];
                    int db = im[k*3+2]-im[k2*3+2];
                    d_e = bw*exp(-0.5 * ( (dx*dx)/(bsx*bsx) + (dy*dy)/(bsy*bsy) + 
                                (dr*dr)/(bsr*bsr) + (dg*dg)/(bsg*bsg) + (db*db)/(bsb*bsb) ) );
                    d_e += gw*exp(-0.5 * ( (dx*dx)/(gsx*gsx) + (dy*dy)/(gsy*gsy) ) );
                    if(do_normalization>0)
                    {
                        if(do_normalization == PIXEL_NORMALIZATION)
                            d_e /= norms[k];
                        else if(do_normalization == MEAN_NORMALIZATION)
                            d_e /= mean_norm;
                    }
                }
                sum_e -= d_e;
            }
        }
        return sum_e;
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
        if(l_unary)
            delete[] l_unary;
        if(init_x)
            delete[] init_x;
        if(norms)
            delete[] norms;

    }

};
int main( int argc, char* argv[]){
	if (argc<4){
		printf("Usage: %s image annotations outputdir\n", argv[0] );
		return 1;
	}
    const int M = 3;
    DenseEnergyMinimizer *e = new DenseEnergyMinimizer(argv[1],argv[2],/*number of labels*/M,
            /* do normalization */ MEAN_NORMALIZATION ,//PIXEL_NORMALIZATION, NO_NORMALIZATION,
            /* do initialization */ true, 
            /* approximate pairwise */false);
    
	ApproximateES aes(/* number of vars */ e->getNumberOfVariables(),/*lambda_min */ 0.0,/* lambda_max*/ 100.0, /* energy_minimizer */e,/* x0 */ e->get_map(), /*max_iter */10000,/*verbosity*/ 10);
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
