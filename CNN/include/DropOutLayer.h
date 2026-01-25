#ifndef DROPOUTLAYER_H
#define DROPOUTLAYER_H

#include <bits/stdc++.h>
#include "base_layer.h"
using namespace std;

class DropOutLayer : public BaseLayer {
    private:
        float drop_probability;
        int input_height;
        int input_width;
        int output_height;
        int output_width;
        int input_channels;
        bool train;
        vector<int> mask_drop; 
        vector<float> last_output;
        vector<float> d_input;
    public:
        DropOutLayer(float drop_probability, int input_channels, int input_height, int input_width, bool train=true);
        vector<int> get_output_size() const;
        const vector<float>& forward(const vector<float>& input, bool train=true);
        const vector<float>& backward(const vector<float>& d_out);

};

#endif