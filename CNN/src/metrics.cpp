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
    if (!ifs) throw runtime_error("metrics::load_vector_from_file: Could not open file: " + filename);

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
    if (!ifs) throw runtime_error("metrics::load_classes_from_file: Could not open file: " + filename);

    vector<string> classes;
    string line;
    while (getline(ifs, line)) {
        if (line.empty()) continue;
        classes.push_back(line);
    }
    return classes;
}

int main() {

    image_data data;
    string weights_folder = "weights/";
    cout << "Data reading started." << endl;
    data.read_train_data("validation/");
    vector<vector<float>> images = data.train_data(1);
    vector<vector<float>> labels = data.train_labels(1);
    cout << "Data reading completed." << endl;
    
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

    cnn.add_layer(new SoftmaxLayer((int)classes.size()));
    cout << "Model loaded successfully." << endl;
    
    cout << "Predicting classes for test data..." << endl;
    vector<pair<string, string>> predictions = cnn.predict_pairs(images, labels, data.get_class_names(), false);
    cout << "Predictions:" << endl;

    map<string, int> correct_count;
    map<string, int> total_count;
    for (const auto& p : predictions) {
        total_count[p.first]++;
        if (p.first == p.second) {
            correct_count[p.first]++;
        }
    }
    for (const auto& entry : total_count) {
        const string& class_name = entry.first;
        int total = entry.second;
        int correct = correct_count[class_name];
        float accuracy = (total > 0) ? (static_cast<float>(correct) / total) * 100.0f : 0.0f;
        cout << "Class: " << class_name << ", Accuracy: " << accuracy << "%" << endl;
    }

    // save accuracy per class to file
    ofstream ofs("class_accuracies.csv");
    ofs << "Class,Accuracy" << endl;
    for (const auto& entry : total_count) {
        const string& class_name = entry.first;
        int total = entry.second;
        int correct = correct_count[class_name];
        float accuracy = (total > 0) ? (static_cast<float>(correct) / total) : 0.0f;
        ofs << class_name << "," << accuracy << endl;
    }
    ofs.close();

    // calculate confusion matrix
    map<string, map<string, int>> confusion_matrix;
    for (const auto& p : predictions) {
        confusion_matrix[p.first][p.second]++;
    }

    // save to csv file
    ofstream ofs1("confusion_matrix.csv");
    ofs1 << "True/Predicted";
    for (const auto& cls : classes) {
        ofs1 << "," << cls;
    }
    ofs1 << endl;
    for (const auto& true_cls : classes) {
        ofs1 << true_cls;
        for (const auto& pred_cls : classes) {
            ofs1 << "," << confusion_matrix[true_cls][pred_cls];
        }
        ofs1 << endl;
    }
    ofs1.close();
    cout << "Confusion matrix saved to confusion_matrix.csv" << endl;

    // f1 score per class and macro f1 score
    map<string, int> tp, fp, fn;
    for (const auto& cls : classes) {
        tp[cls] = fp[cls] = fn[cls] = 0;
    }

    for (const auto& p : predictions) {
        const string& t = p.first;
        const string& pred = p.second;
        if (t == pred) {
            tp[t]++;
        } else {
            fn[t]++;
            fp[pred]++;
        }
    }

    ofstream ofs2("class_f1.csv");
    ofs2 << "Class,Precision,Recall,F1" << endl;

    double sum_f1 = 0.0;
    int f1_count = 0;

    for (const auto& cls : classes) {
        const double TP = tp[cls];
        const double FP = fp[cls];
        const double FN = fn[cls];

        const double precision = (TP + FP > 0.0) ? (TP / (TP + FP)) : 0.0;
        const double recall    = (TP + FN > 0.0) ? (TP / (TP + FN)) : 0.0;
        const double f1        = (precision + recall > 0.0) ? (2.0 * precision * recall / (precision + recall)) : 0.0;

        cout << "Class: " << cls
             << ", Precision: " << precision
             << ", Recall: " << recall
             << ", F1: " << f1 << endl;

        ofs2 << cls << "," << precision << "," << recall << "," << f1 << endl;

        sum_f1 += f1;
        f1_count++;
    }
    ofs2.close();

    const double macro_f1 = (f1_count > 0) ? (sum_f1 / f1_count) : 0.0;
    cout << "Macro-F1: " << macro_f1 << endl;

    ofstream ofs3("f1_summary.txt");
    ofs3 << "Macro-F1: " << macro_f1 << "\n";
    ofs3.close();

    return 0;
}