#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include "base_layer.h"
#include <bits/stdc++.h>
using namespace std;

class SoftmaxLayer : public BaseLayer {
    private:
        int input_size;
        vector<float> last_input;
        vector<float> last_output;
        vector<float> d_input;
        const vector<float>* targets = nullptr;
        float last_loss;
    public:
        SoftmaxLayer(int input_size);
        vector<float>& forward(const vector<float>& input, bool train=true);
        vector<float>& backward(const vector<float>& d_out);
        vector<int> get_output_size() const;
        void set_targets(const vector<float>* target);
        float get_loss() const;

    
};

#endif

