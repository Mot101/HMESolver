#ifndef FC_H
#define FC_H

#include <bits/stdc++.h>
#include "base_layer.h"
using namespace std;

class FCLayer : public BaseLayer {
    private:
        int input_height;
        int input_width;
        int output_size;
        int input_channels;
        int flat;
        vector<float> z_elements;
        vector<float> weights; 
        vector<float> biases; 
        vector<float> d_weights; 
        vector<float> d_biases; 
        vector<float> last_output; 
        vector<float> last_input;
        vector<float> d_input;
    public:
        FCLayer(int input_channels, int input_height, int input_width, int output_size);
        vector<int> get_output_size() const;
        const vector<float>& forward(const vector<float>& input, bool train=true);
        const vector<float>& backward(const vector<float>& d_out);
        void update_weights(float learning_rate, float l2_reg=0.0f);
        void update_biases(float learning_rate, float l2_reg=0.0f);
        vector<float> save_weights() const;
        vector<float> save_biases() const;
        void load_weights(const vector<float>& new_weights);
        void load_biases(const vector<float>& new_biases);
        void zero_gradients();
};

#endif