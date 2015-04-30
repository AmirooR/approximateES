#pragma once

#include "keep-me-concave/keep_me_concave.hpp"
#include <queue>
#include "EnergyMinimizer.hpp"

typedef struct Undefined
{
    float lambda;
    short_array x_l;
    short_array x_r;
    Undefined(float _lambda, short_array _x_l, short_array _x_r):lambda(_lambda),x_l(_x_l),x_r(_x_r)
    {
    }
}Undefined;

// I think there is no need to store \Lambda^\star
/*typedef struct Defined
{
    float lambda;
    short_array x;
    Defined(float _lambda, short_array _x):lambda(_lambda),x(_x)
    {
    }
}Defined;*/


class ApproximateES
{
    vector<short_array> labelings; // labelings aligned with KeepMeConcave::segments
    KeepMeConcave kmc;
    float lambda_min, lambda_max;
    size_t N; // number of variables
    queue<Undefined> Lambda;
    //vector<Defined> LambdaStar;
    EnergyMinimizer* minimizer;

    size_t max_iter;

    public:
    ApproximateES(size_t _N, float _lambda_min, float _lambda_max, EnergyMinimizer* _m ,short* _x0 = NULL, size_t _max_iter = 10000):kmc(_lambda_min, _lambda_max), lambda_min(_lambda_min), lambda_max(_lambda_max), N(_N), minimizer(_m), max_iter(_max_iter) 
    {
        short_array x0( new short[N] );
        for(size_t i = 0; i < N; i++) // copy
        {
            if(_x0 != NULL )
                x0[i] = _x0[i];
            else x0[i] = 0;
        }
        Undefined u1(lambda_min, x0, x0);
        Undefined u2(lambda_max, x0, x0);
        Lambda.push(u1);
        Lambda.push(u2);

        labelings.push_back( x0 );
        labelings.push_back( x0 );
    }

    bool compare(const short_array& s1, const short_array& s2, float lambda)
    {
        if(lambda == lambda_min || lambda == lambda_max)
            return false;
        for(size_t i = 0; i < N; i++)
        {
            if( s1[i] != s2[i] )
                return false;
        }
        return true;

    }

    void loop()
    {
        size_t iter = 0;
        while( (!Lambda.empty()) && (iter < max_iter) )
        {
            cout<<"Iteration: "<<iter<<endl;
            Undefined u = Lambda.front();
            Lambda.pop();
            float energy2, m2, b2,min_m, min_b, min_energy;
            short_array min_x = minimizer->minimize( u.x_l, u.lambda, min_energy, min_m, min_b);
            short_array x2 = minimizer->minimize( u.x_r, u.lambda, energy2, m2, b2);
                        
            if( energy2 < min_energy)
            {
                min_m = m2;
                min_b = b2;
                min_energy = energy2;
                min_x = x2;
            }

            //if( !compare( min_x, u.x_l, u.lambda) && !compare(min_x, u.x_r, u.lambda) )
            {
                LineSegment l( min_m, min_b, lambda_min, lambda_max, false );
                kmc.addLineSegment(l);
                cout<<"* Adding line segment: "<<l<<endl;
                int num_intersections = kmc.num_intersections;
                if( num_intersections > 1 && (kmc.intersecting_lambda[0] != kmc.intersecting_lambda[num_intersections -1] ) )
                { // had intersections
                    cout.precision(8);
                    cout<<"\t Had intersections at "<< kmc.intersecting_lambda[0]<<", "<<kmc.intersecting_lambda[num_intersections-1]<<endl;                    
                    cout<<"\t\t Energies are: "<< fixed << kmc.intersecting_energies[0]<<", "<<fixed<<kmc.intersecting_energies[num_intersections-1]<<endl;
                    float lambda_l = kmc.intersecting_lambda[0];
                    float lambda_r = kmc.intersecting_lambda[num_intersections - 1];
                    int index_l = kmc.intersecting_indexes[0];
                    int index_r = kmc.intersecting_indexes[num_intersections - 1];
                    if(index_r - index_l > 1 )
                        labelings.erase( labelings.begin() + index_l + 1, labelings.begin() + index_r);
                    Undefined u1( lambda_l, labelings[index_l], min_x );
                    Undefined u2( lambda_r, labelings[index_r], min_x );
                    Lambda.push(u1);
                    Lambda.push(u2);

                    labelings.insert( labelings.begin() + index_l + 1, min_x);
                    
                }
            }
            //else
            //{
            //}

            iter++;
        }
        if(iter == max_iter)
            cout<<"Maximum number of iterations reached"<<endl;
        cout<<"Done: "<<endl;
        cout<<kmc<<endl;
        cout<<"labelings:"<<endl;
        for(size_t i = 0; i < labelings.size(); i++)
        {
            cout<<labelings[i][0]<<" ";
        }
        cout<<endl;
    }

    vector<short_array> getLabelings()
    {
        return labelings;
    }

    ~ApproximateES()
    {
    }

};
