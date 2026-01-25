#include "ReluLayer.h"

using namespace std;

ReluLayer::ReluLayer(int input_channels, int input_height, int input_width) {
    this->input_height = input_height;
    this->input_width = input_width;
    this->output_height = input_height;
    this->output_width = input_width;
    this->input_channels = input_channels;
    d_input.resize(input_channels * input_height * input_width, 0.0f);
    last_input.resize(input_channels * input_height * input_width, 0.0f);
}

const vector<float>& ReluLayer::forward(const vector<float>& input, bool train){
    if (input.size() != (size_t)(input_channels * input_height * input_width)){
        throw runtime_error("ReluLayer::forward: input size mismatch.");
    }
    last_input = input;
    last_output = last_input;

    for (int c = 0; c < input_channels; c++) {
        for (int h = 0; h < input_height; h++) {
            for (int w = 0; w < input_width; w++) {
                const int i = c * input_height * input_width + h * input_width + w;
                last_output[i] = max(0.0f, last_input[i]);
            }
        }
    }
    return last_output;
}

const vector<float>& ReluLayer::backward(const vector<float>& d_out) {
    fill(d_input.begin(), d_input.end(), 0.0f);    
    for (int c = 0; c < input_channels; c++) {
        for (int h = 0; h < input_height; h++) {
            for (int w = 0; w < input_width; w++) {
                if (last_input[c * input_height * input_width + h * input_width + w] <= 0) {
                    d_input[c * input_height * input_width + h * input_width + w] = 0.0f;
                }
                else {
                    d_input[c * input_height * input_width + h * input_width + w] = d_out[c * input_height * input_width + h * input_width + w];
                }
            }
        }
    }
    return d_input;
}


vector<int> ReluLayer::get_output_size() const {
    return vector<int>{input_channels, output_height, output_width};
}