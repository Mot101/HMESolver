#pragma once // replace the ifndef , define and endif 
#include <cmath>
#include <string>
#include<vector>
#include <algorithm>
class _NN_
{
    private :
    std::vector<int> npl ; //nbr of neurones per layer 
    std::vector<std::vector<std::vector<float>>> weights; 
    //weights[layer][neuron][previous neuron]
    std::vector<std::vector<float>> bias ; //bias[layer][neuron]; 
    std::vector<std::vector<float>> outputs ; //output per layer 
    std::vector<std::vector<float>> backprop ; //backpropagation per layer 

    public : 
    //constructor
    _NN_(const std::vector<int> &);
    //forward propagation 
    std::vector<float> forward(const std::vector<float>&);
    // training 
    void training(const std::vector<float>& , const std::vector<float>&, float); 
    
    //Activation ft 
    float sigmoid(float x); 
    float sigmoid_derivative(float x);
    static std::vector<float> softmax_vec(const std::vector<float> &z);

};
