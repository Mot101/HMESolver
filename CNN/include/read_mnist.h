#ifndef READ_MNIST_H
#define READ_MNIST_H

#include <bits/stdc++.h>
#include <arpa/inet.h>

using namespace std;

vector<vector<float>> read_mnist_images(const string& filename);
vector<vector<float>> read_mnist_labels(const string& filename);

#endif