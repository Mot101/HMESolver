#ifndef READ_JPG_H
#define READ_JPG_H

#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>


using namespace std;

class image_data {
    public:
        vector<vector<float>> images;
        vector<vector<float>> labels;
        vector<string> class_names;
        int num_classes;
        int num_images;
        void read_train_data(const string& class_dir);
        void read_data(const string& class_dir);
        vector<vector<float>> train_data(float train_ratio);
        vector<vector<float>> train_labels(float train_ratio);
        vector<vector<float>> test_data(float train_ratio, float val_ratio=0.0f);
        vector<vector<float>> test_labels(float train_ratio, float val_ratio=0.0f);
        vector<vector<float>> validation_data(float train_ratio, float val_ratio=0.0f);
        vector<vector<float>> validation_labels(float train_ratio, float val_ratio=0.0f);
        vector<vector<float>> get_images();
        vector<string> get_class_names();
};

int get_number_of_picture(const string& s);


#endif