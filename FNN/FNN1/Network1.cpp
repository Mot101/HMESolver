#include "Network1.hpp"
#include <ctime>
#include <iostream>
#include <algorithm> // max_element
#include <cmath>     // exp
#include <cstdlib>   // exit

using namespace std;
static std::vector<float> softmax_vec(const std::vector<float> &z);

// -------------------- constructor --------------------//
_NN_::_NN_(const std::vector<int> & l)
{
        npl=l ; 
        // safety checks 
        if(npl.size()<2){
            std::cerr << "NN Error : needs at least 2 layers \n" ; 
            std::exit(1); 
        }
        for(size_t i=0; i<npl.size();i++){
            if(npl[i]<=0){
                std::cerr << "NN Error Layer " << i << " has invalid size " << npl[i] << std::endl ; 
                std::exit(1); 
            }
        }
        srand((unsigned)time(0)); 
        //we allocate containers per layer 
        weights.assign(npl.size(),{}); 
        bias.assign(npl.size(),{}); 
        outputs.assign(npl.size(),{});  
        backprop.assign(npl.size(),{}); 

        for (int i=0 ; i<npl.size();i++)
        {
            bias[i].assign(npl[i],0.0f); 
            outputs[i].assign(npl[i],0.0f); 
            backprop[i].assign(npl[i],0.0f);
        }
        for (int i=1 ; i<npl.size();i++)
        {
            weights[i].resize(npl[i]); 
            for (int j=0 ; j<npl[i];j++){
                weights[i][j].resize(npl[i-1]); // connections with previous layer
                bias[i][j]=((float)rand()/RAND_MAX-0.5f)*2.0f;
                for(int k=0 ; k<l[i-1];k++){
                    weights[i][j][k]=((float)rand()/RAND_MAX-0.5f)*2.0f ; 
                }
            }
        }       
}

//-----------------forward propagation-------------------//
std::vector<float> _NN_::forward(const std::vector<float>& input)
{
        //safety tests : 
        if((int)input.size()!= npl[0]){
            std::cerr << "Forward Error : Input Size " << input.size() << "expected " << npl[0] << std::endl ; 
            std::exit(1);
        }
        outputs[0]=input ; 
        // storing the pre actications 
        std::vector<float> z_out ; 
        for (int i=1 ; i<(int)npl.size();i++)
        {
            std::vector<float> z(npl[i],0.0f);
            for(int j=0;j<npl[i];j++)
            {
                float sum=bias[i][j]; 
                for(int k=0;k<npl[i-1];k++)
                {
                    sum+=weights[i][j][k]*outputs[i-1][k]; 
                }
                z[j]=sum ; 
            }
            // hidden layers : sigmoid 
            if(i<(int)npl.size()-1){
                for(int j=0 ; j<npl[i];j++){
                    outputs[i][j]=sigmoid(z[j]);
                }
            }
            else{
                //output layer : softmax
                outputs[i]=softmax_vec(z); 
            }
        }
        return outputs.back();
}
// -------------------- training (softmax + cross-entropy) --------------------
void _NN_::training(const std::vector<float>& input, const std::vector<float>& target ,float learning_rate)
{   
    //safety tests : 
        if((int)input.size()!= npl[0]){
            std::cerr << "Training Error : Input Size " << input.size() << "expected " << npl[0] << std::endl ; 
            std::exit(1);
        }
        if((int)target.size()!= npl.back()){
            std::cerr << "Training Error : Target Size " << target.size() << "expected " << npl.back() << std::endl ; 
            std::exit(1);
        }
    forward(input); 

    //we compute backpropagation ft
    for (int j=0 ; j< npl.back();j++){
        backprop.back()[j]=(target[j]-outputs.back()[j]);
    }

    //backpropagation 
    for(int i =npl.size()-2 ;i>0 ; i--){
        for(int j =0;j<npl[i];j++){
            float sum = 0.0f ;
            for (int k =0;k<npl[i+1];k++){
                sum+=weights[i+1][k][j]*backprop[i+1][k];
            } 
            backprop[i][j]=sum*sigmoid_derivative(outputs[i][j]);
        }
    }

    //update weights and biases 
    for (int i = 1 ; i < npl.size();i++){
        for (int j =0;j<npl[i];j++){
            bias[i][j]+=learning_rate*backprop[i][j];
            for(int k=0 ; k<npl[i-1];k++){
                weights[i][j][k]+=learning_rate*outputs[i-1][k]*backprop[i][j];
            }
        }
    }
}



// -------------------- sigmoid --------------------//
float _NN_::sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

float _NN_::sigmoid_derivative(float a)
{
    // a is sigmoid output
    return a * (1.0f - a);
}
//--------------softmax------------///
static std::vector<float> softmax_vec(const std::vector<float> &z){
    std::vector<float> p(z.size()); 
    if(z.empty()){
        return p ; 
    }
    float m = *std::max_element(z.begin(),z.end());
    float sum=0.0f ; 
    for(size_t i =0 ; i<z.size();i++){
        p[i]=std::exp(z[i]-m); 
        sum+=p[i];
    }
    if (sum<=0.0f){
        sum =1.0f ;
    } 
    for(size_t i =0 ; i<z.size();i++){
        p[i]/=sum ; 
    } 
    return p ; 
}

