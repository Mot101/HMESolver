#include <bits/stdc++.h>

#include "CNN.h"
#include "ConvolutionLayer.h"
#include "ReluLayer.h"
#include "Pool.h"
#include "DropOutLayer.h"
#include "FC.h"
#include "read_mnist.h"
#include "read_jpg.h"
#include "SoftmaxLayer.h"

using namespace std;

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

void save_weights(const BaseLayer* layer, const string& filename) {
    ofstream ofs(filename);
    if (!ofs) {
        throw runtime_error("train: Could not open file for saving weights.");
    }
    vector<float> v = layer->save_weights();
    for (float x : v) ofs << x << '\n';
    ofs.close();
}

void save_biases(const BaseLayer* layer, const string& filename) {
    ofstream ofs(filename);
    if (!ofs) {
        throw runtime_error("train: Could not open file for saving biases.");
    }
    vector<float> biases = layer->save_biases();
    
    for (float x : biases) ofs << x << '\n';
    ofs.close();
}

void save_filters(const BaseLayer* layer, const string& filename) {
    ofstream ofs(filename);
    if (!ofs) {
        throw runtime_error("train: Could not open file for saving filters.");
    }
    vector<float> filters = layer->save_filters();
    for (float x : filters) ofs << x << '\n';
    ofs.close();
}

void save_classes(const vector<string>& classes, const string& filename) {
    ofstream ofs(filename);
    if (!ofs) {
        throw runtime_error("train: Could not open file for saving classes.");
    }
    for (const string& cls : classes) {
        ofs << cls << endl;
    }
    ofs.close();
}

int main(int argc, char* argv[]) {
    if (argc != 8){
        cout << "Usage: ./cnn_train <train_data_dir> <test_data_dir> <epochs> <batch_size> <learning_rate> <l2_reg> <0/1 for the record training_log.txt>" << endl;
        return 1;
    }
    srand(static_cast<unsigned int>(time(0)));
    string weights_folder = "weights/";
    image_data train_data;
    image_data test_data;
    time_t start, end;
    time(&start);
    cout << "Data reading started." << endl;
    train_data.read_train_data(argv[1]);
    vector<vector<float>> train_images = train_data.train_data(1);
    vector<vector<float>> train_labels = train_data.train_labels(1);
    test_data.read_train_data(argv[2]);
    vector<vector<float>> test_images = test_data.train_data(1);
    vector<vector<float>> test_labels = test_data.train_labels(1);
    time(&end);
    float time_taken = float(end - start);
    cout << "Data reading completed in " << time_taken << " seconds." << endl;
    save_classes(train_data.get_class_names(), weights_folder + "classes.txt");
    cout << "Name of classes saved to classes.txt" << endl;
    time(&start);
    cout << "Creating CNN model..." << endl;
    CNN_model cnn;
    vector<int> out;

    cnn.add_layer(new ConvolutionLayer(6, 5, 1, 2, 44, 44,1));
    
    out = cnn.get_output_size();
    cnn.add_layer(new ReluLayer(out[0], out[1], out[2]));
    out = cnn.get_output_size();
    cnn.add_layer(new PoolLayer(out[0], out[1], out[2], 2, 2));
    out = cnn.get_output_size();
    cnn.add_layer(new ConvolutionLayer(16, 5, 1, 2, out[1], out[2], out[0]));

    out = cnn.get_output_size();
    cnn.add_layer(new ReluLayer(out[0], out[1], out[2]));
    out = cnn.get_output_size();
    cnn.add_layer(new PoolLayer(out[0], out[1], out[2], 2, 2));
    out = cnn.get_output_size();
   
    cnn.add_layer(new FCLayer(out[0], out[1], out[2], 120));
    out = cnn.get_output_size();
    cnn.add_layer(new ReluLayer(out[0], out[1], out[2]));
    out = cnn.get_output_size();
    cnn.add_layer(new FCLayer(out[0], out[1], out[2], 84));
    out = cnn.get_output_size();
    cnn.add_layer(new ReluLayer(out[0], out[1], out[2]));
    out = cnn.get_output_size();
    cnn.add_layer(new FCLayer(out[0], out[1], out[2], (int)train_data.get_class_names().size()));
    out = cnn.get_output_size();
    cnn.add_layer(new SoftmaxLayer(out[0]));
    
    cout << "CNN model created." << endl;
    time(&end);
    time_taken = float(end - start);
    cout << "Time taken for model creation: " << time_taken << " seconds" << endl;
    time(&start);
    cout << "Starting " ;
    if (atoi(argv[7]) == 1) {
        cout << "training with log recording..." << endl;
        cnn.train(train_images, train_labels, atoi(argv[3]), atoi(argv[4]), stof(argv[5]), stof(argv[6]), test_images, test_labels, true, true, true);
    }
    else{
        cout << "training..." << endl;
        cnn.train(train_images, train_labels, atoi(argv[3]), atoi(argv[4]), stof(argv[5]), stof(argv[6]));
    }

    cout << "Training completed." << endl;
    time(&end);
    time_taken = float(end - start);
    cout << "Time taken for training: " << time_taken << " seconds" << endl;
    cout << "Train Accuracy: ";
    float train_accuracy = cnn.evaluate(train_images, train_labels, false);
    cout << train_accuracy * 100 << "%" << endl;

    time(&start);
    cout << "Evaluating on test data..." << endl;
    float accuracy = cnn.evaluate(test_images, test_labels, false);
    
    cout << "Test Accuracy: " << accuracy * 100 << "%" << endl;
    time(&end);
    time_taken = float(end - start);
    cout << "Time taken for evaluation: " << time_taken << " seconds" << endl;

    cout << "Saving model parameters..." << endl;
    
    // first convolutional layer
    save_filters(cnn.layers[0], weights_folder +  "conv1_filters.txt");
    save_biases(cnn.layers[0], weights_folder + "conv1_biases.txt");
    // second convolutional layer
    save_filters(cnn.layers[3], weights_folder + "conv2_filters.txt");
    save_biases(cnn.layers[3], weights_folder + "conv2_biases.txt");
    
    // first fully connected layer
    save_weights(cnn.layers[6], weights_folder + "fc1_weights.txt");
    save_biases(cnn.layers[6], weights_folder + "fc1_biases.txt");
    // second fully connected layer
    save_weights(cnn.layers[8], weights_folder + "fc2_weights.txt");
    save_biases(cnn.layers[8], weights_folder + "fc2_biases.txt");
    // third fully connectedm layer
    save_weights(cnn.layers[10], weights_folder + "fc3_weights.txt");
    save_biases(cnn.layers[10], weights_folder + "fc3_biases.txt");
    
    
    cout << "Model parameters saved." << endl;
    
    

    
    return 0;
}
