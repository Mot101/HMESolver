#include <bits/stdc++.h>
#include "CNN.h"
#include "ConvolutionLayer.h"
#include "ReluLayer.h"
#include "Pool.h"
#include "DropOutLayer.h"
#include "FC.h"
#include "read_mnist.h"
#include "read_jpg.h"
#include "symbols.h"
#include "SoftmaxLayer.h"
using namespace std;
namespace fs = filesystem;


vector<float> load_vector_from_file(const string& filename) {
    ifstream ifs(filename);
    if (!ifs) throw runtime_error("solver: Could not open file: " + filename);

    vector<float> data;
    string line;
    while (getline(ifs, line)) {
        if (line.empty()) continue;
        data.push_back(stof(line));
    }
    return data;
}
vector<string> load_classes_from_file(const string& filename) {
    ifstream ifs(filename);
    if (!ifs) throw runtime_error("solver: Could not open file: " + filename);

    vector<string> classes;
    string line;
    while (getline(ifs, line)) {
        if (line.empty()) continue;
        classes.push_back(line);
    }
    return classes;
}

int main(int argc, char* argv[]) {
    if (argc != 2){
        cout << "Usage: ./cnn_extractor <image_path>" << endl;
        return 1;
    }
    string weights_folder = "weights/";
    string image_path = argv[1]; // path to the input image
    string output_dir = "symbols"; // directory to save extracted symbols

    detect_and_save_symbols(image_path, output_dir); 
    vector<string> classes = load_classes_from_file(weights_folder + "classes.txt");
    cout << "Loading CNN model..." << endl;
    CNN_model cnn;
    cnn.add_layer(new ConvolutionLayer(6, 5, 1, 2, 44, 44,1));
    cnn.layers[0]->load_filters(load_vector_from_file(weights_folder + "conv1_filters.txt"));   
    cnn.layers[0]->load_biases(load_vector_from_file(weights_folder + "conv1_biases.txt"));
    vector<int> out = cnn.get_output_size();
    cnn.add_layer(new ReluLayer(out[0], out[1], out[2]));
    out = cnn.get_output_size();
    cnn.add_layer(new PoolLayer(out[0], out[1], out[2], 2, 2));
    out = cnn.get_output_size();
    cnn.add_layer(new ConvolutionLayer(16, 5, 1, 2, out[1], out[2], out[0]));
    cnn.layers[3]->load_filters(load_vector_from_file(weights_folder + "conv2_filters.txt"));   
    cnn.layers[3]->load_biases(load_vector_from_file(weights_folder + "conv2_biases.txt"));
    out = cnn.get_output_size();
    cnn.add_layer(new ReluLayer(out[0], out[1], out[2]));
    out = cnn.get_output_size();
    cnn.add_layer(new PoolLayer(out[0], out[1], out[2], 2, 2));
    out = cnn.get_output_size();
    cnn.add_layer(new FCLayer(out[0], out[1], out[2], 120));
    cnn.layers[6]->load_weights(load_vector_from_file(weights_folder + "fc1_weights.txt"));
    cnn.layers[6]->load_biases(load_vector_from_file(weights_folder + "fc1_biases.txt"));
    out = cnn.get_output_size();
    cnn.add_layer(new ReluLayer(out[0], out[1], out[2]));
    cnn.add_layer(new FCLayer(out[0], out[1], out[2], 84));
    cnn.layers[8]->load_weights(load_vector_from_file(weights_folder + "fc2_weights.txt"));
    cnn.layers[8]->load_biases(load_vector_from_file(weights_folder + "fc2_biases.txt"));
    out = cnn.get_output_size();
    cnn.add_layer(new ReluLayer(out[0], out[1], out[2]));
    out = cnn.get_output_size();
    cnn.add_layer(new FCLayer(out[0], out[1], out[2], (int)classes.size()));
    cnn.layers[10]->load_weights(load_vector_from_file(weights_folder + "fc3_weights.txt"));
    cnn.layers[10]->load_biases(load_vector_from_file(weights_folder + "fc3_biases.txt"));
    out = cnn.get_output_size();
    cnn.add_layer(new SoftmaxLayer(out[0]));
    cout << "Model loaded successfully." << endl;
    cout << "Classifying symbols..." << endl;
    image_data data;
    data.read_data(output_dir);
    vector<vector<float>> images = data.get_images();
    cout << "Predicted equation: " << endl;
    for(vector<vector<float>>::iterator img_it = images.begin(); img_it != images.end(); img_it++) {
        const vector<float>& out = cnn.predict(*img_it, false);
        
        int pred = 0;
        float best = out[0];
        for (int i = 1; i < (int)out.size(); i++) {
            if (out[i] > best) {
                best = out[i];
                pred = i;
            }
        }
        if (classes[pred] == "dot") cout << "*" << " ";
        else if (classes[pred] == "forward_slash") cout << "/" ;
        else if (classes[pred] == "plus") cout << "+";
        else if (classes[pred] == "minus") cout << "-";
        else cout << classes[pred] ;
    }
    cout << endl;
    return 0;
}