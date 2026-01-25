#ifndef POOL_H
#define POOL_H

#include <bits/stdc++.h>
#include "base_layer.h"
using namespace std;

class PoolLayer : public BaseLayer {
    private:
        int input_height;
        int input_width;
        int input_channels;
        int pool_height;
        int pool_width;
        int output_height;
        int output_width;
        int stride;
        vector<float> last_output;
        vector<float> last_input;
        vector<float> max_mask;
        vector<float> d_input;
    public:
        PoolLayer(int input_channels, int input_height, int input_width, int pool_size, int stride);
        vector<int> get_output_size() const;
        const vector<float>& forward(const vector<float>& input, bool train=true);
        const vector<float>& backward(const vector<float>& d_out);
        
};

#endif
