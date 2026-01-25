#ifndef CONVOLUTIONLAYER_H
#define CONVOLUTIONLAYER_H

#include <bits/stdc++.h>
#include "base_layer.h"
using namespace std;

class ConvolutionLayer : public BaseLayer {
    private:
            int input_height;
            int input_width;
            int num_filters;
            int filter_size;
            int stride;
            int padding;
            int output_height;
            int output_width;
            int init_channels;
            int padded_height;
            int padded_width;
            vector<float> filters;
            vector<float> biases; 
            vector<float> d_filters;
            vector<float> d_biases;
            vector<float> last_output;
            vector<float> d_padded_input;
            vector<float> d_input_grad;
            vector<float> last_input;

    public:
            ConvolutionLayer(int num_filters, int filter_size, int stride, int padding, int input_height, int input_width, int input_channels);
            vector<int> get_output_size() const;
            const vector<float>& forward(const vector<float>& input, bool train=true);
            const vector<float>& backward(const vector<float>& d_out);
            void update_filters(float learning_rate, float l2_reg=0.0f);
            void update_biases(float learning_rate, float l2_reg=0.0f);
            vector<float> save_filters() const;
            vector<float> save_biases() const;
            void load_filters(const vector<float>& new_filters);
            void load_biases(const vector<float>& new_biases);
            void zero_gradients();
};

#endif