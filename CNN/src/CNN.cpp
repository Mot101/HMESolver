
#include "CNN.h"

using namespace std;

void CNN_model::add_layer(BaseLayer* layer) {
    layers.push_back(layer);
}

const vector<float>& CNN_model::forward(const vector<float>& input, bool train) {
    const vector<float>* output = &input; // make to avoid copy input
    for (BaseLayer* layer : layers) {
        output = &layer->forward(*output, train);
    }
    return *output;
}

const vector<float>& CNN_model::backward(const vector<float>& d_out) {
    const vector<float>* d_input = &d_out;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        d_input = &(*it)->backward(*d_input);
    }
    return *d_input;
}

float CNN_model::train_step(const vector<vector<float>>& inputs, const vector<vector<float>>& targets, const vector<int>& indices, int begin, int end, float learning_rate, float l2_reg, bool train_flag) {
    // "begin" and "end" are indices in indices array
    float loss = 0.0f;
    for (BaseLayer* layer : layers) {
        layer->zero_gradients();
    }

    vector<float> d_out;
    for (int idx = begin; idx < end; ++idx) {
        int sample_idx = indices[idx];
        const vector<float>& input = inputs[sample_idx];
        const vector<float>& target = targets[sample_idx];
        layers.back()->set_targets(&target);


        const vector<float>& output = forward(input, train_flag);
        // calculate loss 
        loss += layers.back()->get_loss();
        backward(d_out);
    }
    const int batch_size = end - begin;
    const float lr_eff = learning_rate / (batch_size > 0 ? (float)batch_size : 1.0f);
    for (BaseLayer* layer : layers) {
        layer->update_weights(lr_eff, l2_reg);
        layer->update_biases(lr_eff, l2_reg);
        layer->update_filters(lr_eff, l2_reg);
    }
    return loss;
}
float CNN_model::calculate_test_loss(const vector<vector<float>>& inputs, const vector<vector<float>>& targets, bool train_flag) {
    float total_loss = 0.0f;
    const int N = (int)inputs.size();
    for (int n = 0; n < N; ++n) {
        const vector<float>& input = inputs[n];
        const vector<float>& target = targets[n];
        layers.back()->set_targets(&target);

        const vector<float>& output = forward(input, train_flag);
        total_loss += layers.back()->get_loss();
    }
    return total_loss;
}
void CNN_model::train(const vector<vector<float>>& inputs, const vector<vector<float>>& targets, int epochs, int batch_size,float learning_rate, float l2_reg, const vector<vector<float>>& test_inputs, const vector<vector<float>>& test_targets, bool shuffle_flag, bool metrics, bool train_flag) {
    ofstream log_file("training_log.txt");
    if (!log_file) {    
        throw runtime_error("CNN_model::train: Could not open log file.");
    }
    vector<int> indices(inputs.size());
    iota(indices.begin(), indices.end(), 0); // fill with 0,1,2,...,N-1

    static mt19937 rng(42); 
    int check = 0; 
    for (int epoch = 0; epoch < epochs; ++epoch) {
        if (shuffle_flag) {
            shuffle(indices.begin(), indices.end(), rng);
        }
        if (epoch % 4 == 0 && epoch != 0) {
            learning_rate *= 0.5f;
        }

        float total_loss = 0.0f;
        cout << "Starting epoch " << (epoch + 1) << "/" << epochs << endl;
        for (int t = 0; t < (int)inputs.size(); t += batch_size) {
            const int end = min(t + batch_size, (int)inputs.size());
            total_loss += train_step(inputs, targets, indices, t, end, learning_rate, l2_reg, train_flag);
            if ((t/batch_size) % 10000 == 0) cout << "Processed " << end << "/" << inputs.size() << " samples\r" << flush;
        }
        if (metrics){ 
            log_file << "Epoch " << epoch << " evaluation: ";
            log_file << "Train accuracy: ";
            float acc_train = evaluate(inputs, targets, false);
            log_file << acc_train << ", ";
            log_file << "Test accuracy: ";
            float acc = evaluate(test_inputs, test_targets, false);
            log_file << acc << endl;
        }
        cout << endl;
        cout << "Epoch " << (epoch + 1) << "/" << epochs
             << ", Loss: " << (total_loss / (float)inputs.size()) << endl;
        if (metrics) {
            log_file << "Epoch " << epoch << " loss: ";
            log_file << (total_loss / (float)inputs.size()) << ", ";
            float test_loss = calculate_test_loss(test_inputs, test_targets, false);
            log_file << "Test Loss: " << test_loss << endl;
        }
        
    }
}

const vector<float>& CNN_model::predict(const vector<float>& input, bool train) {
    return forward(input, train);
}

CNN_model::~CNN_model() {
    for (BaseLayer* layer : layers) delete layer;
}

vector<int> CNN_model::get_output_size() const {
    if (layers.empty()) return {};
    return layers.back()->get_output_size();
}

float CNN_model::evaluate(const vector<vector<float>>& images, const vector<vector<float>>& labels, bool train_flag) {
    const int N = (int)images.size();
    if (N == 0) return 0.0f;

    int correct = 0;

    for (int n = 0; n < N; ++n) {
        const vector<float>& out = forward(images[n], train_flag);

        int pred = 0;
        float best = out[0];
        for (int i = 1; i < (int)out.size(); i++) {
            if (out[i] > best) {
                best = out[i];
                pred = i;
            }
        }

        int target = 0;
        for (int i = 0; i < (int)labels[n].size(); i++) {
            if (labels[n][i] == 1.0f) { target = i; break; }
        }

        if (pred == target) correct++;
    }

    return (float)correct / (float)N;
}

// method to predict class names for a vector of images (in our task to predict symbols from equation)
vector<string> CNN_model::predict_classes(const vector<vector<float>>& images, const vector<string>& class_names, bool train_flag) {
    vector<string> predicted_classes;
    for (vector<vector<float>>::const_iterator img_it = images.begin(); img_it != images.end(); ++img_it) {
        const vector<float>& out = forward(*img_it, train_flag);
        
        int pred = 0;
        float best = out[0];
        for (int i = 1; i < (int)out.size(); i++) {
            if (out[i] > best) {
                best = out[i];
                pred = i;
            }
        }
        predicted_classes.push_back(class_names[pred]);
    }
    return predicted_classes;
}

//method to predict class names for images and return vector of pairs (true class, predicted class)
vector<pair<string, string>> CNN_model::predict_pairs(const vector<vector<float>>& images, const vector<vector<float>>& true_labels, const vector<string>& class_names, bool train_flag) {
    vector<pair<string, string>> results;
    for (int idx = 0; idx < images.size(); ++idx) {
        const vector<float>& out = forward(images[idx], train_flag);
        
        int pred = 0;
        float best = out[0];
        for (int i = 1; i < (int)out.size(); i++) {
            if (out[i] > best) {
                best = out[i];
                pred = i;
            }
        }

        int true_class = 0;
        for (int i = 0; i < (int)true_labels[idx].size(); i++) {
            if (true_labels[idx][i] == 1.0f) { true_class = i; break; }
        }

        results.emplace_back(class_names[true_class], class_names[pred]);
    }
    return results;
}