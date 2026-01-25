#include "Pool.h"

using namespace std;

PoolLayer::PoolLayer(int input_channels, int input_height, int input_width, int pool_size, int stride) {
    this->input_channels = input_channels;
    this->input_height = input_height;
    this->input_width = input_width;
    this->pool_height = pool_size;
    this->pool_width = pool_size;
    this->stride = stride;
    this->output_height = (input_height - pool_size) / stride + 1;
    this->output_width = (input_width - pool_size) / stride + 1;
    last_output.resize(input_channels * output_height * output_width, 0.0f);
    d_input.resize(input_channels * input_height * input_width, 0.0f);
    max_mask.assign(input_channels * input_height * input_width, 0.0f);
}

const vector<float>& PoolLayer::forward(const vector<float>& input, bool train){
    if (input.size() != (size_t)(input_channels * input_height * input_width)){
        throw runtime_error("PoolLayer::forward: input size mismatch.");
    }

    fill(max_mask.begin(), max_mask.end(), 0.0f);

    for (int c = 0; c < input_channels; c++) {
        for (int h = 0; h < output_height; h++) {
            for (int w = 0; w < output_width; w++) {
                float max_val = -numeric_limits<float>::infinity();
                int max_h = -1;
                int max_w = -1;

                for (int ph = 0; ph < pool_height; ph++) {
                    for (int pw = 0; pw < pool_width; pw++) {
                        int input_h = h * stride + ph;
                        int input_w = w * stride + pw;

                        float v = input[c * input_height * input_width + input_h * input_width + input_w];
                        if (v > max_val) {
                            max_val = v;
                            max_h = input_h;
                            max_w = input_w;
                        }
                    }
                }

                max_mask[c * input_height * input_width + max_h * input_width + max_w] = 1.0f;
                last_output[c * output_height * output_width + h * output_width + w] = max_val;
            }
        }
    }
    return last_output;
}

const vector<float>& PoolLayer::backward(const vector<float>& d_out){
    if (d_out.size() != (size_t)(input_channels * output_height * output_width)){
        throw runtime_error("PoolLayer::backward: d_out size mismatch.");
    }
    fill(d_input.begin(), d_input.end(), 0.0f);

    for (int c = 0; c < input_channels; c++){
        for (int h = 0; h < output_height; h++){
            for (int w = 0; w < output_width; w++){
                int max_h = -1, max_w = -1;

                for (int oh = 0; oh < pool_height; oh++){
                    for (int ow = 0; ow < pool_width; ow++){
                        int ih = h * stride + oh;
                        int iw = w * stride + ow;
                        if (max_mask[c * input_height * input_width + ih * input_width + iw] == 1.0f){
                            max_h = ih;
                            max_w = iw;
                        }
                    }
                }

                if (max_h >= 0 && max_w >= 0) {
                    d_input[c * input_height * input_width + max_h * input_width + max_w] +=
                        d_out[c * output_height * output_width + h * output_width + w];
                }
            }
        }
    }
    return d_input;
}

vector<int> PoolLayer::get_output_size() const {
    return vector<int>{input_channels, output_height, output_width};
}
