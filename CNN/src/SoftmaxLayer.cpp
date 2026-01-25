#include "SoftmaxLayer.h"
using namespace std;

SoftmaxLayer::SoftmaxLayer(int input_size) {
    this->input_size = input_size;
    last_input.resize(input_size, 0.0f);
    last_output.resize(input_size, 0.0f);
    d_input.resize(input_size, 0.0f);
    last_loss = 0.0f;
}

vector<float>& SoftmaxLayer::forward(const vector<float>& input, bool train) {
    if ((int)input.size() != input_size) {
        throw runtime_error("SoftmaxLayer::forward: input size mismatch.");
    }
    last_input = input;

    float max_input = *max_element(input.begin(), input.end());
    float sum_exp = 0.0f;
    for (int i = 0; i < input_size; ++i) {
        last_output[i] = exp(input[i] - max_input);
        sum_exp += last_output[i];
    }
    if (!isfinite(sum_exp) || sum_exp <= 0.0f) {
        fill(last_output.begin(), last_output.end(), 1.0f / (float)input_size);
        cout << "Warning: SoftmaxLayer::forward: sum_exp is non-finite or non-positive. Returning uniform distribution." << endl;
        last_loss = 0.0f;
        return last_output;
    }
    for (int i = 0; i < input_size; ++i) {
        last_output[i] /= sum_exp;
    }
    // calculate loss
    last_loss = 0.0f;
    if (targets != nullptr) {
        if ((int)targets->size() != input_size) {
            throw runtime_error("SoftmaxLayer::forward: targets size mismatch.");
        }
        const float eps = 1e-12f;
        int label = -1;
        for (int i = 0; i < input_size; i++) {
            if ((*targets)[i] == 1.0f) {
                label = i;
                break;
            }
        }
        if (label == -1) {
            throw runtime_error("SoftmaxLayer::forward: invalid target vector, no class marked as 1.");
        }
        if (!isfinite(last_output[label]) || last_output[label] < eps) {
            last_loss = -log(eps);
        } else {
            last_loss = -log(last_output[label]);
        }
    }   
    return last_output;
}

vector<float>& SoftmaxLayer::backward(const vector<float>& d_out) {
    
    if (targets == nullptr) {
        throw runtime_error("SoftmaxLayer::backward: targets not set.");
    }
    if ((int)targets->size() != input_size) {
        throw runtime_error("SoftmaxLayer::backward: targets size mismatch.");
    }
    for (int i = 0; i < input_size; ++i) {
        d_input[i] = last_output[i] - (*targets)[i];
    }
    return d_input;
}

float SoftmaxLayer::get_loss() const {
    if (targets == nullptr) {
        throw runtime_error("SoftmaxLayer::get_loss: targets not set.");
    }
    if (last_loss < 0.0f) {
        cout << "Warning: SoftmaxLayer::get_loss called before forward pass or loss is non-positive." << endl;
        return 0.0f;
    }
    return last_loss;
}

void SoftmaxLayer::set_targets(const vector<float>* targets) {
    this->targets = targets;
}
vector<int> SoftmaxLayer::get_output_size() const {
    return {input_size,1,1};
}