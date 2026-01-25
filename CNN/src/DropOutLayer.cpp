#include "DropOutLayer.h"

using namespace std;

DropOutLayer::DropOutLayer(float drop_probability,int input_channels, int input_height, int input_width,  bool train) {
    this->drop_probability = drop_probability;
    this->input_height = input_height;
    this->input_width = input_width;
    this->output_height = input_height;
    this->output_width = input_width;
    this->input_channels = input_channels;
    this->train = train;
    last_output.resize(input_channels * input_height * input_width, 0.0f);
    mask_drop.resize(input_channels * input_height * input_width, 1);
    d_input.resize(input_channels * input_height * input_width, 0.0f);
}

const vector<float>& DropOutLayer::forward(const vector<float>& input, bool train){
    this->train = train;
    if (!train){
        return input; 
    }
    if (input.size() != input_channels * input_height * input_width){
        throw runtime_error("DropOutLayer::forward: input size mismatch.");
    }
    for(int c = 0; c < input_channels; c++){
        for(int h = 0; h < input_height; h++){
            for (int w = 0; w < input_width; w++){
                float rand_val = ((float)rand()) / RAND_MAX;
                if (rand_val < drop_probability){
                    mask_drop[c * input_height * input_width + h * input_width + w] = 0;
                    last_output[c * input_height * input_width + h * input_width + w] = 0.0;
                }
                else{
                    mask_drop[c * input_height * input_width + h * input_width + w] = 1;
                    last_output[c * input_height * input_width + h * input_width + w] = input[c * input_height * input_width + h * input_width + w] / (1.0 - drop_probability); 
                }
            }
        }
    }
    return last_output;
}

const vector<float>& DropOutLayer::backward(const vector<float>& d_out){
    if (!train){
        return d_out; 
    }
    if (d_out.size() != input_channels * input_height * input_width){
        throw runtime_error("DropOutLayer::backward: d_out size mismatch.");
    }
    for(int c = 0; c < input_channels; c++){
        for(int h = 0; h < input_height; h++){
            for (int w = 0; w < input_width; w++){
                if (mask_drop[c * input_height * input_width + h * input_width + w] == 1){
                    d_input[c * input_height * input_width + h * input_width + w] =d_out[c * input_height * input_width + h * input_width + w] * mask_drop[c * input_height * input_width + h * input_width + w] / (1.0 - drop_probability); 
                }
                else{
                    d_input[c * input_height * input_width + h * input_width + w] = 0.0;
                }
            }
        }
    }
    return d_input;
}

vector<int> DropOutLayer::get_output_size() const {
    return vector<int>{input_channels, output_height, output_width};
}
