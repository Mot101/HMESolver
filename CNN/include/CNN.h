#ifndef CNN_H
#define CNN_H

#include <bits/stdc++.h>
#include "base_layer.h"
using namespace std;

class CNN_model{
    public:
        vector<BaseLayer*> layers;
        CNN_model() = default;
        const vector<float>& forward(const vector<float>& input, bool train=true);
        const vector<float>& backward(const vector<float>& d_out);
        void add_layer(BaseLayer* layer);
        float train_step(const vector<vector<float>>& inputs,const vector<vector<float>>& targets,const vector<int>& indices,int begin, int end,float learning_rate, float l2_reg, bool train_flag);
        void train(const vector<vector<float>>& inputs, const vector<vector<float>>& targets, int epochs, int batch_size, float learning_rate, float l2_reg, const vector<vector<float>>& test_inputs = {}, const vector<vector<float>>& test_targets = {}, bool shuffle_flag = true, bool metrics = false, bool train=true);
        const vector<float>& predict(const vector<float>& input, bool train=true);
        vector<int> get_output_size() const;
        float evaluate(const vector<vector<float>>& images, const vector<vector<float>>& labels, bool train=false);
        vector<string> predict_classes(const vector<vector<float>>& images, const vector<string>& classes,bool train=false);
        ~CNN_model();
        vector<pair<string,string>> predict_pairs(const vector<vector<float>>& images, const vector<vector<float>>& true_labels, const vector<string>& classes,bool train=false);
        float calculate_test_loss(const vector<vector<float>>& inputs, const vector<vector<float>>& targets, bool train_flag);

};

#endif