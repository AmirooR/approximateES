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
    short *prev_map;
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
    bool use_prev_computation;
    int do_normalization;
    double gsx,gsy,gw;
    double bsx,bsy,bsr,bsg,bsb,bw;
    double *norms;
    double mean_norm;
    int num_computations;
    double prev_p_e;

    public:
    DenseEnergyMinimizer(const char *im_path, const char *anno_path, int M, 
            int do_normalization = 0,
            bool do_initialization = true,
            bool approximate_pairwise=false,
            bool use_prev_computation=true,
            double gsx = 3.f, double gsy = 3.f, double gw=3.f,
            double bsx = 60.f, double bsy = 60.f, double bsr=20.f, double bsg=20.f, double bsb=20.f, double bw=10.f
            ):
        M(M),
        u_sum(0),
        p_sum(0),
        crf(NULL),
        map(NULL),
        prev_map(NULL),
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
        use_prev_computation(use_prev_computation),
        gsx(gsx), gsy(gsy), gw(gw),
        bsx(bsx), bsy(bsy), bsr(bsr), bsg(bsg), bsb(bsb), bw(bw),
        num_computations(0),
        prev_p_e(0)
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
        prev_map = new short[N];
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
            l_unary[i] = lambda+unary[i];
        crf->setUnaryEnergy(l_unary);
        if(approximate_pairwise)
        {
            double b_energy = N*lambda + u_sum + p_sum;
            cout<<"Before optimization: approximate energy is "<<b_energy<<endl;
            crf->unaryEnergy( map, u_result);
            crf->pairwiseEnergy(map, p_result, -1);
            float n_u_sum = 0.0f, n_p_sum = 0.0f;
            for(int i = 0; i < N; ++i)
            {
                n_u_sum += u_result[i];
                n_p_sum += p_result[i];
            }
            cout<< "Unary sum: "<<n_u_sum <<" = N*lambda+u = "<<N<<"* "<<lambda<<"+ "<< u_sum<<" = "<<N*lambda+u_sum<<endl;
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
            cout<< "Unary sum: "<<n_u_sum <<" = N*lambda+u = "<<N*lambda<<"+ "<< u_sum<<" = "<<N*lambda+u_sum<<endl;
            cout<< "Pairwise sum: "<<n_p_sum <<endl;
            cout<< "After optimization: energy is "<< (n_p_sum + n_u_sum) << endl;
            m = N;
            b = u_sum + n_p_sum;
            energy = m * lambda + b;
        }
        else//computing exact pairwise
        {
            crf->map(3, map);
            cout<<"AFTER ***"<<endl;
            crf->unaryEnergy( map, u_result);
            /* for test 
               if(num_computations>0)
               test_computations();
               end for test */
            double n_p_sum = compute_pairwise_energy(prev_p_e);
            if(use_prev_computation)
            {
                /*  for test 
                    use_prev_computation = false;
                    double my_np_sum = compute_pairwise_energy(prev_p_e);
                    cout<<KBBLU<< "pairwise: "<<my_np_sum<<", using prev: "<<n_p_sum<<RESET<<endl;
                    use_prev_computation = true;
                    end of for test*/

                memcpy(prev_map, map, N*sizeof(short));
                prev_p_e = n_p_sum;

            }

            double n_u_sum = 0.0;
            u_sum = 0;
            for(int i = 0; i < N; ++i)
            {
                n_u_sum += u_result[i];
                //n_p_sum += p_result[i];
                output[i] = map[i];
                u_sum += unary[ i*M+map[i]];
            }
            cout<< "Unary sum: "<<n_u_sum <<" = N*lambda+u = "<<N*lambda<<"+ "<< u_sum<<" = "<<N*lambda+u_sum<<endl;
            cout<< "Pairwise sum: "<<n_p_sum <<endl;
            cout<< "After optimization: energy is "<< (n_p_sum + n_u_sum) << endl;
            m = N;
            b = u_sum+n_p_sum;
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

    /* computes kernel[k,k2] without normalization */
    inline double compute_k_j_no_norm(int k, int k2)
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
        double d_e = bw*exp(-0.5 * ( (dx*dx)/(bsx*bsx) + (dy*dy)/(bsy*bsy) + 
                    (dr*dr)/(bsr*bsr) + (dg*dg)/(bsg*bsg) + (db*db)/(bsb*bsb) ) );
        d_e += gw*exp(-0.5 * ( (dx*dx)/(gsx*gsx) + (dy*dy)/(gsy*gsy) ) );
        return d_e;
    }

    /* computes the norms:
     *  - mean_norm
     *  - pixel-wise norm in norms[i]
     *    for pixel i
     * */
    void compute_norms()
    {
        mean_norm = 0.0;
        for(int k=0; k < N; k++)
        {
            double this_sum = 0.0;
#pragma omp parallel for reduction(+:this_sum)                    
            for(int  k2=0; k2 < N;  k2++)
            {                
                this_sum += compute_k_j_no_norm(k,k2);
            }
            norms[k] = this_sum;
            mean_norm += this_sum;
        }
        mean_norm = mean_norm/(N);
    }

    /* compute kernel[i,j] and normalize it */
    inline double compute_k_i_j(int i, int j)
    {
        double d_e = compute_k_j_no_norm(i,j);
        if(do_normalization>0)
        {
            if(do_normalization == PIXEL_NORMALIZATION)
                d_e /= norms[i];
            else if(do_normalization == MEAN_NORMALIZATION)
                d_e /= mean_norm;
        }
        return d_e;
    }

    /* computing pairwise using previous energy
     * @prev_pe: previous energy
     * NOTE: @prev_map , @map are previous map, and current map labelling
     * and should be set before calling this funtion.
     * */
    double compute_pairwise_using_prev(double prev_pe)
    {
        double sum_e = prev_pe;
        for(int k=0; k < N; k++)
        {
            if( map[k] != prev_map[k] )
            {
#pragma omp parallel for reduction(+:sum_e)                    
                for(int k2=0; k2 < N; k2++)
                {
                    double d_e = 0.0;
                    if( prev_map[k] == prev_map[k2] && map[k] != map[k2] && k!=k2)
                    {
                        if(! (prev_map[k2] != map[k2] && k > k2))
                            d_e += compute_k_i_j(k,k2);
                    }
                    else if(map[k] == map[k2] && prev_map[k] != prev_map[k2] && k!=k2)
                    {
                        if(! (prev_map[k2] != map[k2] && k > k2))
                            d_e -= compute_k_i_j(k,k2);
                    }
                    sum_e += d_e;
                }
            }
        }
        return sum_e;
    }


    double compute_pairwise_energy(double prev_pe)
    {
        if(num_computations > 0 && use_prev_computation)
            return compute_pairwise_using_prev(prev_pe);
        double sum_e = 0.0f;
        for(int k=0; k < N; k++)
        {
#pragma omp parallel for reduction(-:sum_e)                    
            for(int  k2=0; k2 < k;  k2++)
            {
                double d_e = 0;
                if( map[k] == map[k2] )
                {
                    d_e = compute_k_i_j(k,k2); 
                }
                sum_e -= d_e;
            }
        }
        num_computations++;
        //prev_p_e = sum_e;
        return sum_e;
    }
//DEBUG
    void test_computations()
    {
        check_symmetry();
        //exact
        cout<<KBYEL<<" ** TESTING ** "<<RESET<<endl;
        cout<<endl;
        cout<<endl;
        double prev_exact, new_exact;
        int prev_exact_num, new_exact_num;
        compute_pairwise_exact( prev_map, prev_exact_num, prev_exact);
        compute_pairwise_exact( map, new_exact_num, new_exact);
        printf("%sprev_exact: %f # %d\n",KRED,prev_exact, prev_exact_num);
        printf("%snew_exact:  %f # %d\n",KRED,new_exact, new_exact_num);

        double similar_e;int similar_num;
        similar(prev_map, map, similar_num, similar_e);
        printf("%ssimilar:    %f # %d\n",KRED,similar_e, similar_num);

        double subtract_e;int subtract_num;
        subtract(prev_map, map, subtract_num , subtract_e);
        printf("%ssubtract:   %f # %d\n",KRED,subtract_e, subtract_num);

        double add_e;int add_num;
        add(prev_map, map, add_num, add_e);
        printf("%sadd:        %f # %d\n\n",KRED, add_e, add_num);
        printf("sim+sub(%f) == prev(%f)\n", similar_e+subtract_e, prev_exact);
        printf("sim+add(%f) == new (%f)\n\n", similar_e+add_e, new_exact);

        printf("#sim+#sub(%d) == #prev(%d)\n", similar_num+subtract_num, prev_exact_num);
        printf("#sim+#add(%d) == #new (%d)\n\n", similar_num+add_num, new_exact_num);

        double t_current_sum, t_add_sum, t_sub_sum;
        int t_current_num, t_add_num, t_sub_num;
        cout<<KBGRN<<"PREV_P_E: "<< -prev_p_e<<RESET<<endl;
        using_prev(prev_map, map, prev_exact, t_add_sum, t_add_num, t_sub_sum, t_sub_num, t_current_sum, t_current_num);

        printf("%snew_fast_e(%f) == new(%f)\n",KCYN, t_current_sum, new_exact);
        printf("%sfast_add(%f)   == add_e(%f)\n",KCYN, t_add_sum, add_e);
        printf("%sfast_sub(%f)   == sub_e(%f)\n\n",KCYN, t_sub_sum, subtract_e);

        printf("%s#new_fast_e(%d) != #new(%d)\n",KMAG, t_current_num, new_exact_num);
        printf("%s#fast_add(%d)   == #add_e(%d)\n",KMAG, t_add_num, add_num);
        printf("%s#fast_sub(%d)   == #sub_e(%d)\n",KMAG, t_sub_num, subtract_num);

        cout<<RESET<<endl;
        cout<<endl;
        cout<<endl;
    }

    void check_symmetry()
    {
        for(int i=0; i < N; i++)
            for(int j=0; j<i; j++)
            {
                double d1 = compute_k_i_j(i,j);
                double d2 = compute_k_i_j(j,i);
                if( d1 != d2 )
                    cout<<KBGRN<<"ERROR IN SYMMETRY ("<<i<<", "<<j<<")"<<RESET<<endl;
            }

    }

    void compute_pairwise_exact(short* t_map, int& num,double& sum)
    {
        sum = 0;
        num = 0;
        for(int i=0; i < N; i++)
            for(int j=0; j<i; j++)
                if( t_map[i] == t_map[j] )
                {
                    sum += compute_k_i_j(i,j);
                    num++;
                }
    }

    void similar(short* t_prev_map, short* t_new_map, int& num, double& sum)
    {
        sum = 0;
        num = 0;
        for(int i=0; i < N; i++)
            for(int j=0; j<i; j++)
                if( t_prev_map[i] == t_prev_map[j]  && t_new_map[i] == t_new_map[j] )
                {
                    sum += compute_k_i_j(i,j);
                    num++;
                }

    }

    void subtract(short* t_prev_map, short* t_new_map, int& num, double& sum)
    {
        sum = 0;
        num = 0;
        for(int i=0; i < N; i++)
            for(int j=0; j<i; j++)
                if( t_prev_map[i] == t_prev_map[j]  && t_new_map[i] != t_new_map[j] )
                {
                    sum += compute_k_i_j(i,j);
                    num++;
                }
    }


    void add(short* t_prev_map, short* t_new_map, int& num, double& sum)
    {
        sum = 0;
        num = 0;
        for(int i=0; i < N; i++)
            for(int j=0; j<i; j++)
                if( t_prev_map[i] != t_prev_map[j]  && t_new_map[i] == t_new_map[j] )
                {
                    sum += compute_k_i_j(i,j);
                    num++;
                }
    }

    void using_prev(short* t_prev_map, short* t_new_map, double prev_pe, 
            double& add_sum, int& add_num,
            double& sub_sum, int& sub_num,
            double& cur_sum, int& cur_num)
    {
        add_sum = sub_sum = 0;
        cur_num = sub_num = add_num = 0;
        cur_sum = prev_pe;
        for(int i=0; i < N; i++)
            if( t_prev_map[i] != t_new_map[i] )
                for(int j=0; j<N; j++)
                {

                    if(t_prev_map[i] != t_prev_map[j] && t_new_map[i] == t_new_map[j] && i!=j)
                    {
                        if(! (t_prev_map[j] != t_new_map[j] && i > j) )
                        {
                            double curr = compute_k_i_j(i,j);
                            add_sum += curr;
                            cur_sum += curr;
                            add_num++;
                            cur_num++;
                        }
                    }
                    else if(t_prev_map[i] == t_prev_map[j] && t_new_map[i] != t_new_map[j] && i!=j)
                    {
                        if(! (t_prev_map[j] != t_new_map[j] && i > j) )
                        {
                            double curr = compute_k_i_j(i,j);
                            sub_sum += curr;
                            cur_sum -= curr;
                            sub_num++;
                            cur_num++;
                        }
                    }
                }
    }

//END OF DEBUG


    virtual ~DenseEnergyMinimizer()
    {
        if(crf)
            delete crf;
        if(map)
            delete[] map;
        if(prev_map)
            delete[] prev_map;
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
            /* do normalization */ MEAN_NORMALIZATION,//MEAN_NORMALIZATION ,//PIXEL_NORMALIZATION, NO_NORMALIZATION,
            /* do initialization */ true, 
            /* approximate pairwise */false,
            /* use_prev_computation */true);
    
	ApproximateES aes(/* number of vars */ e->getNumberOfVariables(),/*lambda_min */ -0.000015,/* lambda_max*/ 0.000015, /* energy_minimizer */e,/* x0 */ e->get_map(), /*max_iter */10000,/*verbosity*/ 10);
    aes.loop();
    vector<short_array> labelings = aes.getLabelings();
    string out_dir(argv[3]);
    string lambdas_file = out_dir + string("lambdas.txt");
    aes.writeLambdas(lambdas_file.c_str());
    
    for(size_t i = 0; i < labelings.size(); i++)
    {
        string out("_output.ppm");
        string s = out_dir + SSTR( i ) + out;
        unsigned char *res = colorize( labelings[i].get(), e->getWidth(), e->getHeight());
        writePPM( s.c_str(), e->getWidth(), e->getHeight(), res);
        delete[] res;
        //imwrite(s.c_str(), m);
    }

    delete e;
}
