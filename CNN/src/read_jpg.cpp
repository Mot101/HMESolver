#include "read_jpg.h"

namespace fs = std::filesystem;

// function to sort vector of symbols based on the number in their filename
int get_number_of_picture(const string& s){
    string sub_1 = "symbols/symbol_";
    string sub_2 = ".jpg";
    int pos1 = s.find(sub_1);
    int pos2 = s.find(sub_2);
    string number_str = s.substr(pos1 + sub_1.length(), pos2 - (pos1 + sub_1.length()));
    return stoi(number_str);
}

// function to make the background of a photo look the same
void normalize_background(cv::Mat& img){
    int k = min(3, min(img.rows, img.cols));
    k = max(3, k);
    cv::Rect rect1(0,0,k,k);
    cv::Rect rect2(img.cols - k, 0, k, k);
    cv::Rect rect3(0, img.rows - k, k, k);
    cv::Rect rect4(img.cols - k, img.rows - k, k, k);
    float median_val = (cv::mean(img(rect1))[0] + cv::mean(img(rect2))[0] + cv::mean(img(rect3))[0] + cv::mean(img(rect4))[0]) / 4.0f;
    if (median_val > 128.0){
        cv::bitwise_not(img, img); // invert colors
    }
}

// function to resize image to output_size x output_size with padding and saving aspect ratio
void resize_image(cv::Mat& img, int output_size){
    if (img.empty()) {
        throw runtime_error("ResizeImage: Empty image provided for resizing.");
    }
    int k = min(2, min(img.rows, img.cols)/5);
    k = max(1, k);
    float mean_val = (cv::mean(img(cv::Rect(0,0,k,k)))[0] + cv::mean(img(cv::Rect(img.cols - k, 0, k, k)))[0] +
                          cv::mean(img(cv::Rect(0, img.rows - k, k, k)))[0] + 
                          cv::mean(img(cv::Rect(img.cols - k, img.rows - k, k, k)))[0]) / 4.0f;
    float border_value = (mean_val > 128.0f) ? 255.0f : 0.0f;
    float scale = min(output_size / (float)img.cols, output_size / (float)img.rows);
    int new_width = (int)(img.cols * scale);
    int new_height = (int)(img.rows * scale);
    cv::resize(img, img, cv::Size(new_width, new_height), 0, 0, cv::INTER_AREA);
    cv::Mat padded_img(output_size, output_size, img.type(), cv::Scalar(border_value));
    int x = (output_size - new_width) / 2;  
    int y = (output_size - new_height) / 2;
    img.copyTo(padded_img(cv::Rect(x, y, new_width, new_height)));
    img = padded_img;
}

void image_data::read_train_data(const string& class_dir) {

    images.clear();
    labels.clear();
    class_names.clear();

    // directory_iterator does not provide sorted order, so we first collect class names
    for (const auto& entry : fs::directory_iterator(class_dir)) {
        if (entry.is_directory()) {
            string class_name = entry.path().filename().string();
            class_names.push_back(class_name);
        }
    }

    num_classes = class_names.size();
    sort(class_names.begin(), class_names.end());
    cout << "Found " << num_classes << " classes." << endl;
    cout << "Class names: ";
    for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
        cout << class_names[class_idx] << " ";
    }
    cout << endl;
    const int img_size = 44;
    for (int idx = 0; idx < num_classes; idx++) {
        const fs::path class_path = fs::path(class_dir) / class_names[idx];
        int taken = 0;
        for (const auto& img_entry : fs::directory_iterator(class_path)) {
            if (!img_entry.is_regular_file()) continue;
            const std::string img_path = img_entry.path().string();
            cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                throw runtime_error("read_train_data: could not read image: " + img_path);
            }
            normalize_background(img);


            if (img.rows != img_size || img.cols != img_size) {
                resize_image(img, img_size);
            }

            img.convertTo(img, CV_32F, 1.0 / 255.0);
            std::vector<float> img_vec;
            img_vec.reserve(img_size * img_size);

            for (int r = 0; r < img.rows; ++r) {
                const float* row = img.ptr<float>(r);
                img_vec.insert(img_vec.end(), row, row + img.cols);
            }

            std::vector<float> label_vec(num_classes, 0.0f);
            label_vec[idx] = 1.0f;

            images.push_back(move(img_vec));
            labels.push_back(move(label_vec));
        }
    }

    // shuffle data for better training
    vector<int> indices(images.size());
    iota(indices.begin(), indices.end(), 0);
    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);
    vector<vector<float>> shuffled_images;
    vector<vector<float>> shuffled_labels;
    for (int i : indices) {
        shuffled_images.push_back(move(images[i]));
        shuffled_labels.push_back(move(labels[i]));
    }
    images = move(shuffled_images);
    labels = move(shuffled_labels);
}

vector<string> image_data::get_class_names() {
    return class_names;
}

// method to read images from equation after separating symbols
void image_data::read_data(const string& class_dir) {
    images.clear();
    labels.clear();
    class_names.clear();

    const int img_size = 44;
    vector<pair<int,string>> image_paths;
    for (const auto& img_entry : fs::directory_iterator(class_dir)){
        if (!img_entry.is_regular_file()) continue;
        const std::string img_path = img_entry.path().string();
        int img_number = get_number_of_picture(img_path);
        image_paths.push_back({img_number, img_path});
    }
    sort(image_paths.begin(), image_paths.end(), [](const pair<int,string>& a, const pair<int,string>& b) {
        return a.first < b.first;
    });
    for (vector<pair<int,string>>::iterator it = image_paths.begin(); it != image_paths.end(); it++){
        const std::string img_path = it->second;
        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            throw runtime_error("read_data: could not read image: " + img_path);
        }

        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); 
        normalize_background(img);
        if (img.rows != img_size || img.cols != img_size) {
            resize_image(img, img_size);
        }

        img.convertTo(img, CV_32F, 1.0 / 255.0);

        std::vector<float> img_vec;
        img_vec.reserve(img_size * img_size);

        for (int r = 0; r < img.rows; ++r) {
            const float* row = img.ptr<float>(r);
            img_vec.insert(img_vec.end(), row, row + img.cols);
        }

        images.push_back(std::move(img_vec));
    }
}

vector<vector<float>> image_data::train_data(float train_ratio) {
    int train_size = static_cast<int>(images.size() * train_ratio);
    return vector<vector<float>>(images.begin(), images.begin() + train_size);
}

vector<vector<float>> image_data::train_labels(float train_ratio) {
    int train_size = static_cast<int>(labels.size() * train_ratio);
    return vector<vector<float>>(labels.begin(), labels.begin() + train_size);
}

vector<vector<float>> image_data::test_data(float train_ratio, float val_ratio) {
    int train_size = static_cast<int>(images.size() * train_ratio);
    return vector<vector<float>>(images.begin() + train_size, images.end()- (int)images.size() * val_ratio);
}

vector<vector<float>> image_data::test_labels(float train_ratio, float val_ratio) {
    int train_size = static_cast<int>(labels.size() * train_ratio);
    return vector<vector<float>>(labels.begin() + train_size, labels.end()- (int)labels.size() * val_ratio);
}

vector<vector<float>> image_data::validation_data(float train_ratio, float val_ratio) {
    int train_size = (int)(images.size() * train_ratio);
    int val_size = (int)(images.size() * val_ratio);
    return vector<vector<float>>(images.end() - val_size, images.end());
}
vector<vector<float>> image_data::validation_labels(float train_ratio, float val_ratio) {
    int train_size = (int)(labels.size() * train_ratio);
    int val_size = (int)(labels.size() * val_ratio);
    return vector<vector<float>>(labels.end() - val_size, labels.end());
}
vector<vector<float>> image_data::get_images(){
    return images;
}