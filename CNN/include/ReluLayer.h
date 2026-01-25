#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include <bits/stdc++.h>
#include "base_layer.h"
using namespace std;

class ReluLayer : public BaseLayer {
private:
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    int input_channels;
    vector<float> last_input;
    vector<float> last_output;
    vector<float> d_input;
public:
    ReluLayer(int input_channels, int input_height, int input_width);
    vector<int> get_output_size() const override;
    const vector<float>& forward(const vector<float>& input, bool train);
    const vector<float>& backward(const vector<float>& d_out) ;
    
};

#endif