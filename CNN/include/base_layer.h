#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include <bits/stdc++.h>
using namespace std;

class BaseLayer {
    public:
        BaseLayer() = default;
        virtual vector<int> get_output_size() const = 0;
        virtual const vector<float>& forward(const vector<float>& input, bool train) = 0;
        virtual const vector<float>& backward(const vector<float>& d_out) = 0;
        virtual ~BaseLayer() = default;
        virtual void load_weights(const vector<float>& new_weights) {};
        virtual void load_biases(const vector<float>& new_biases) {};
        virtual void load_filters(const vector<float>& new_filters) {};
        virtual void update_weights(float learning_rate, float l2_reg) {};
        virtual void update_biases(float learning_rate, float l2_reg) {};
        virtual void update_filters(float learning_rate, float l2_reg) {};
        virtual vector<float> save_weights() const { return {}; }
        virtual vector<float> save_biases() const { return {}; }
        virtual vector<float> save_filters() const { return {}; }
        virtual void zero_gradients() { };
        virtual float get_loss() const { return 0.0f; }
        virtual void set_targets(const vector<float>* targets) {  };

        
    };

#endif