#include "ConvolutionLayer.h"

using namespace std;

ConvolutionLayer::ConvolutionLayer(int num_filters, int filter_size, int stride, int padding, int input_height, int input_width, int input_channels) {
    this->num_filters = num_filters;
    this->filter_size = filter_size;
    this->stride = stride;
    this->padding = padding;
    this->input_height = input_height;
    this->input_width = input_width;
    this->output_height = (input_height - filter_size + 2 * padding) / stride + 1;
    this->output_width  = (input_width  - filter_size + 2 * padding) / stride + 1;
    this->init_channels = input_channels;
    this->padded_height = input_height + 2 * padding;
    this->padded_width  = input_width  + 2 * padding;
    
    filters.resize(num_filters * init_channels * filter_size * filter_size);
    biases.resize(num_filters);
    last_output.resize(num_filters * output_height * output_width, 0.0f);
    last_input.resize(init_channels * padded_height * padded_width, 0.0f);
    d_filters.resize(filters.size(), 0.0f);
    d_biases.resize(num_filters, 0.0f);
    d_padded_input.resize(init_channels * padded_height * padded_width, 0.0f);
    d_input_grad.resize(init_channels * input_height * input_width, 0.0f);
    
    // He initialization (see more here: https://en.wikipedia.org/wiki/Weight_initialization and https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/)
    // I use this because ReLU activation is used after convolution
    float var_param = sqrt(2.0f / (filter_size * filter_size * init_channels));
    normal_distribution<float> distribution(0.0f, var_param);
    default_random_engine generator;
    for (int f = 0; f < num_filters; f++) {
        for (int c = 0; c < init_channels; c++) {
            for (int fh = 0; fh < filter_size; fh++) {
                for (int fw = 0; fw < filter_size; fw++) {
                    filters[f * init_channels * filter_size * filter_size+ c * filter_size * filter_size+ fh * filter_size + fw] = distribution(generator);
                }
            }
        }
        biases[f] = 0.0f;
    }
}

void ConvolutionLayer::zero_gradients() {
    fill(d_filters.begin(), d_filters.end(), 0.0f);
    fill(d_biases.begin(), d_biases.end(), 0.0f);
}

const vector<float>& ConvolutionLayer::forward(const vector<float>& input, bool train) {

    if ((int)input.size() != init_channels * input_height * input_width) {
        throw runtime_error("ConvolutionLayer::forward: Input size mismatch.");
    }

    const int C = init_channels;
    const int in_hw = input_height * input_width;
    const int out_hw = output_height * output_width;
    const int kk = filter_size * filter_size;

    if (padding == 0) {
        last_input = input;
        const float* X = input.data();

        for (int f = 0; f < num_filters; ++f) {
            const int w_f_base = f * C * kk;
            const float b = biases[f];

            for (int oh = 0; oh < output_height; ++oh) {
                const int ih0 = oh * stride;
                for (int ow = 0; ow < output_width; ++ow) {
                    const int iw0 = ow * stride;

                    float sum = b;
                    for (int c = 0; c < C; ++c) {
                        const int x_c_base = c * in_hw + ih0 * input_width + iw0;
                        const int w_c_base = w_f_base + c * kk;

                        for (int fh = 0; fh < filter_size; ++fh) {
                            const int x_row = x_c_base + fh * input_width;
                            const int w_row = w_c_base + fh * filter_size;
                            for (int fw = 0; fw < filter_size; ++fw) {
                                sum += X[x_row + fw] * filters[w_row + fw];
                            }
                        }
                    }
                    last_output[f * out_hw + oh * output_width + ow] = sum;
                }
            }
        }
       
    }
    else {
        const int pad_hw = padded_height * padded_width;

        fill(last_input.begin(), last_input.end(), 0.0f);
        for (int c = 0; c < C; ++c) {
            const int pad_base = c * pad_hw;
            const int in_base  = c * in_hw;
            for (int h = 0; h < input_height; ++h) {
                const int pad_row = (h + padding) * padded_width + padding;
                const int in_row  = h * input_width;
                for (int w = 0; w < input_width; ++w) {
                    last_input[pad_base + pad_row + w] = input[in_base + in_row + w];
                }
            }
        }

        for (int f = 0; f < num_filters; ++f) {
            const int w_f_base = f * C * kk;
            const float b = biases[f];

            for (int oh = 0; oh < output_height; ++oh) {
                const int ih0 = oh * stride;
                for (int ow = 0; ow < output_width; ++ow) {
                    const int iw0 = ow * stride;

                    float sum = b;
                    for (int c = 0; c < C; ++c) {
                        const int x_c_base = c * pad_hw + ih0 * padded_width + iw0;
                        const int w_c_base = w_f_base + c * kk;

                        for (int fh = 0; fh < filter_size; ++fh) {
                            const int x_row = x_c_base + fh * padded_width;
                            const int w_row = w_c_base + fh * filter_size;
                            for (int fw = 0; fw < filter_size; ++fw) {
                                sum += last_input[x_row + fw] * filters[w_row + fw];
                            }
                        }
                    }

                    last_output[f * out_hw + oh * output_width + ow] = sum;
                }
            }
        }
    }
    return last_output;
}

const vector<float>& ConvolutionLayer::backward(const vector<float>& d_out) {

    const int C = init_channels;
    const int in_hw = input_height * input_width;
    const int out_hw = output_height * output_width;
    const int kk = filter_size * filter_size;
    zero_gradients();
    if ((int)d_out.size() != num_filters * out_hw) {
        throw runtime_error("ConvolutionLayer::backward: d_out size mismatch.");
    }
    fill(d_input_grad.begin(), d_input_grad.end(), 0.0f);

   if (padding == 0) {
        if ((int)last_input.size() != C * in_hw) {
            throw runtime_error("ConvolutionLayer::backward: last_input size mismatch.");
        }
        const float* X = last_input.data();


        for (int f = 0; f < num_filters; ++f) {
            const int w_f_base = f * C * kk;
            const int out_base = f * out_hw;

            for (int oh = 0; oh < output_height; ++oh) {
                const int ih0 = oh * stride;
                for (int ow = 0; ow < output_width; ++ow) {
                    const int iw0 = ow * stride;

                    const float g = d_out[out_base + oh * output_width + ow];
                    d_biases[f] += g;

                    for (int c = 0; c < C; ++c) {
                        const int x_c_base = c * in_hw + ih0 * input_width + iw0;
                        const int w_c_base = w_f_base + c * kk;
                        float* dXc = d_input_grad.data();

                        for (int fh = 0; fh < filter_size; ++fh) {
                            const int x_row = x_c_base + fh * input_width;
                            const int w_row = w_c_base + fh * filter_size;
                            for (int fw = 0; fw < filter_size; ++fw) {
                                const float x = X[x_row + fw];
                                const float w = filters[w_row + fw];
                                d_filters[w_row + fw] += x * g;
                                dXc[x_row + fw]       += w * g;
                            }
                        }
                    }
                }
            }
        }
    }
    else {


        const int pad_h = padded_height;
        const int pad_w = padded_width;
        const int pad_hw = pad_h * pad_w;
        if ((int)last_input.size() != C * pad_hw) {
            throw runtime_error("ConvolutionLayer::backward: last_input (padded) size mismatch.");
        }


        for (int f = 0; f < num_filters; ++f) {
            const int out_base = f * out_hw;
            for (int i = 0; i < out_hw; ++i) {
                d_biases[f] += d_out[out_base + i]  ;
            }
        }

        for (int f = 0; f < num_filters; ++f) {
            const int out_base = f * out_hw;
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    const float g = d_out[out_base + oh * output_width + ow];
                    const int ih0 = oh * stride;
                    const int iw0 = ow * stride;

                    for (int fh = 0; fh < filter_size; ++fh) {
                        for (int fw = 0; fw < filter_size; ++fw) {
                            const int ih = ih0 + fh;
                            const int iw = iw0 + fw;
                            for (int c = 0; c < C; ++c) {
                                d_filters[f * C * kk + c * kk + fh * filter_size + fw] += last_input[c * pad_hw + ih * pad_w + iw] * g;
                            }
                        }
                    }
                }
            }
        }

        fill(d_padded_input.begin(), d_padded_input.end(), 0.0f);

        for (int f = 0; f < num_filters; ++f) {
            const int out_base = f * out_hw;
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    const float g = d_out[out_base + oh * output_width + ow];
                    const int ih0 = oh * stride;
                    const int iw0 = ow * stride;

                    for (int fh = 0; fh < filter_size; ++fh) {
                        for (int fw = 0; fw < filter_size; ++fw) {
                            const int ih = ih0 + fh;
                            const int iw = iw0 + fw;
                            for (int c = 0; c < C; ++c) {
                                d_padded_input[c * pad_hw + ih * pad_w + iw] +=
                                    filters[f * C * kk + c * kk + fh * filter_size + fw] * g;
                            }
                        }
                    }
                }
            }
        }

        for (int c = 0; c < C; ++c) {
            const int pad_base = c * pad_hw;
            const int out_base = c * in_hw;
            for (int h = 0; h < input_height; ++h) {
                const int pad_row = (h + padding) * pad_w + padding;
                const int out_row = h * input_width;
                for (int w = 0; w < input_width; ++w) {
                    d_input_grad[out_base + out_row + w] = d_padded_input[pad_base + pad_row + w];
                }
            }
        }
    }
    return d_input_grad;
}

void ConvolutionLayer::update_filters(float learning_rate, float l2_reg) {
    for (int i = 0; i < (int)filters.size(); i++) {
        filters[i] -= learning_rate * (d_filters[i] + l2_reg * filters[i]);
    }
}

void ConvolutionLayer::update_biases(float learning_rate, float l2_reg) {
    for (int f = 0; f < num_filters; f++) {
        biases[f] -= learning_rate * (d_biases[f] + l2_reg * biases[f]);
    }
}

vector<int> ConvolutionLayer::get_output_size() const {
    return vector<int>{num_filters, output_height, output_width};
}

vector<float> ConvolutionLayer::save_filters() const {
    return filters;
}

vector<float> ConvolutionLayer::save_biases() const {
    return biases;
}

void ConvolutionLayer::load_filters(const vector<float>& new_filters) {
    filters = new_filters;
}

void ConvolutionLayer::load_biases(const vector<float>& new_biases) {
    biases = new_biases;
}
