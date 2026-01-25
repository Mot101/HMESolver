#include "FC.h"

using namespace std;

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

FCLayer::FCLayer(int input_channels,int input_height, int input_width, int output_size) {
    this->input_height = input_height;
    this->input_width = input_width;
    this->output_size = output_size;
    this->input_channels = input_channels;
    this->flat = input_channels * input_height * input_width;

    weights.resize(output_size * input_height * input_width * input_channels);
    biases.resize(output_size);
    d_weights.resize(output_size * input_height * input_width * input_channels, 0.0f);
    d_biases.resize(output_size, 0.0f);
    z_elements.resize(output_size, 0.0f);
    d_input.resize(input_channels * input_height * input_width, 0.0f);
    last_output.resize(output_size, 0.0f);
    last_input.resize(input_channels * input_height * input_width, 0.0f);

    float var_param = sqrt(2.0f / (input_height * input_width * input_channels));
    normal_distribution<float> distribution(0.0f, var_param);
    default_random_engine generator;
    for (int i = 0; i < output_size * input_height * input_width * input_channels; ++i) {
        weights[i] = distribution(generator);
    }
    for (int i = 0; i < output_size; ++i) {
        biases[i] = 0.0f;
    }
    
}


const vector<float>& FCLayer::forward(const vector<float>& input, bool train){
    if ((int)input.size() != flat){
        throw runtime_error("FCLayer::forward: input size mismatch.");
    }
    last_input = input;

    for (int i = 0; i < output_size; ++i) {
        float sum = biases[i];
        for (int j = 0; j < flat; ++j) sum += weights[i * flat + j] * last_input[j];
        last_output[i] = sum;
    }
   
    return last_output;
}

void FCLayer::zero_gradients() {
    fill(d_weights.begin(), d_weights.end(), 0.0f);
    fill(d_biases.begin(), d_biases.end(), 0.0f);
}

const vector<float>& FCLayer::backward(const vector<float>& d_out){
    if ((int)d_out.size() != output_size){
        throw runtime_error("FCLayer::backward: d_out size mismatch.");
    }
    fill(d_input.begin(), d_input.end(), 0.0f);

    for (int i = 0; i < output_size; ++i) {
        float g = d_out[i];
        d_biases[i] += g;
        int base = i * flat;
        for (int j = 0; j < flat; ++j) {
            d_weights[base + j] += last_input[j] * g;
            d_input[j]          += weights[base + j] * g;
        }
    }
    return d_input;
}



void FCLayer::update_weights(float learning_rate, float l2_reg) {
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_channels * input_height * input_width; ++j)
            weights[i*input_channels * input_height * input_width + j] -= learning_rate * (d_weights[i*input_channels * input_height * input_width + j] + l2_reg * weights[i*input_channels * input_height * input_width + j]);
    }
}

void FCLayer::update_biases(float learning_rate, float l2_reg) {
    for (int i = 0; i < output_size; ++i) {
        biases[i] -= learning_rate * (d_biases[i]);
    }
}

vector<int> FCLayer::get_output_size() const {
    return vector<int>{output_size, 1, 1};
}

vector<float> FCLayer::save_weights() const {
    return weights;
}
vector<float> FCLayer::save_biases() const {
    return biases;
}
void FCLayer::load_weights(const vector<float>& new_weights) {
    weights = new_weights;
}
void FCLayer::load_biases(const vector<float>& new_biases) {
    biases = new_biases;
}



