#include "read_mnist.h"

using namespace std;

vector<vector<float>> read_mnist_images(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("read_mnist_images: Could not open file: " + filename);
    }

    int magic_number = 0;
    int number_of_images = 0;
    int rows = 0;
    int cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = ntohl(magic_number);
    if (magic_number != 2051) {
        throw runtime_error("read_mnist_images: Invalid MNIST image file!");
    }

    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = ntohl(number_of_images);
    file.read((char*)&rows, sizeof(rows));
    rows = ntohl(rows);
    file.read((char*)&cols, sizeof(cols));
    cols = ntohl(cols);

    vector<vector<float>> images(number_of_images, vector<float>(1 * rows * cols));

    for (int i = 0; i < number_of_images; ++i) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                unsigned char pixel = 0;
                file.read((char*)&pixel, sizeof(pixel));
                images[i][0 * rows * cols + r * cols + c] = static_cast<float>(pixel) / 255.0f; // 0 because we use grayscale images
            }
        }
    }

    file.close();
    return images;
}

vector<vector<float>> read_mnist_labels(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("read_mnist_labels: Could not open file: " + filename);
    }

    int magic_number = 0;
    int number_of_labels = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = ntohl(magic_number);
    if (magic_number != 2049) {
        throw runtime_error("read_mnist_labels: Invalid MNIST label file!");
    }

    file.read((char*)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = ntohl(number_of_labels);

    vector<vector<float>> labels(number_of_labels, vector<float>(1 * 10, 0.0f));

    for (int i = 0; i < number_of_labels; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        labels[i][0 * 10 + static_cast<int>(label)] = 1.0f; 
    }

    file.close();
    return labels;
}