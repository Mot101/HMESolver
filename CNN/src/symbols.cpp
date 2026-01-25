#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace std;
void detect_and_save_symbols(const string& image_path, const string& output_dir, int output_size=45) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        throw runtime_error("detect_and_save_symbols: Could not open or find the image: " + image_path);
    }
    cout << "Image loaded successfully.\n";
    cv::Mat gray, binary; // creating a matrix for images

    // see more about cvtColor: https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#gaf86c09fe702ed037c03c2bc603ceab14
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // converting an image to a black-and-white filter

    // making the image white for numbers and black for the background
    // cv::THRESH_BINARY_INV | cv::THRESH_OTSU sets the rule for determining the digit/background - thresh_binary_inv: if the value is greater than the threshold (which is determined by the Otsu algorithm), then 0, otherwise 255 - maxval 
    // see more about threshold: https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57 
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU); //

    vector<vector<cv::Point>> contours; // vector of future countours
    vector<cv::Vec4i> hierarchy; // create vector of 4 int numbers (4i)

    // each detected contour stored as a vector of Point
    // hierarchy - connection between contours contains as many elements as there are detected contours
    // cv::RETR_EXTERNAL - mode of finding contours, use only external boundary (from docs: retrieves only the extreme outer contours. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.)
    // cv::CHAIN_APPROX_SIMPLE - contour preservation method, compresses to end points (from docs: compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an upright rectangular contour is encoded with 4 points.)
    //see more about findContours: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
    cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    
    sort(contours.begin(), contours.end(), [](const vector<cv::Point>& a, const vector<cv::Point>& b) {
        return cv::boundingRect(a).x < cv::boundingRect(b).x;
    });

    vector<cv::Rect> rectangles; // we will use to find the equal sign

    for(vector<vector<cv::Point>>::iterator i = contours.begin() ; i!=contours.end(); i++){
        if (i->size() < 10) continue; // skip small contours

        cv::Rect rect = cv::boundingRect(*i); // create rectangle boundary of contour
        rectangles.push_back(rect);
    }

    // Now we will group rectangles that are close to each other (to combine two lines of the equal sign)
    vector<bool> merged(rectangles.size(), false); // vector to track merged rectangles
    vector<cv::Rect> grouped; // vector to store grouped rectangles
    for (int i = 0 ; i < rectangles.size(); ++i) {
        if (merged[i]) continue;

        cv::Rect current = rectangles[i];
        for (int j = i + 1; j < rectangles.size(); ++j) {
            if (merged[j]) continue;

            // Check if rectangles are close in x direction
            int max_x_gap = 10;  // max distance in x between rectangles

            if (std::abs(rectangles[j].x - current.x) <= max_x_gap) {
                // Merge rectangles
                current = current | rectangles[j]; // union of two rectangles
                merged[j] = true;
            }
        }

        grouped.push_back(current);
    }

    int symbol_idx = 0;
    for (vector<cv::Rect>::iterator i = grouped.begin() ; i!=grouped.end(); i++) { 
        int pad = 2; // size of padding in pixels
        // rect.x - coordinate of the up-left corner on the x-axis
        // rect.y - coordinate of the up-left corner on the y-axis
        // rect.width - width of the rectangle
        // rect.height - height of the rectangle
        // img.cols - total image width
        // img.rows - total image height
        // check if we can allocate the necessary margin
        cv::Rect rect = *i;
        rect.x = std::max(0, rect.x - pad);
        rect.y = std::max(0, rect.y - pad);
        rect.width = std::min(img.cols - rect.x, rect.width + 2 * pad);
        rect.height = std::min(img.rows - rect.y, rect.height + 2 * pad);
        
        // save symbol as independent file
        cv::Mat symbol = img(rect).clone();
        cv::cvtColor(symbol, symbol, cv::COLOR_BGR2GRAY);
        float mean_val = (cv::mean(symbol(cv::Rect(0,0,2,2)))[0] + cv::mean(symbol(cv::Rect(symbol.cols - 2, 0, 2, 2)))[0] +
                          cv::mean(symbol(cv::Rect(0, symbol.rows - 2, 2, 2)))[0] + 
                          cv::mean(symbol(cv::Rect(symbol.cols - 2, symbol.rows - 2, 2, 2)))[0]) / 4.0f;
        float border_value = (mean_val > 128.0f) ? 255.0f : 0.0f;
        float scale = min(output_size / (float)rect.width, output_size / (float)rect.height);
        int new_width = (int)(rect.width * scale);
        int new_height = (int)(rect.height * scale);
        cv::resize(symbol, symbol, cv::Size(new_width, new_height), 0, 0, cv::INTER_AREA);
        cv::Mat padded_symbol(output_size, output_size, symbol.type(), cv::Scalar(border_value));
        int x = (output_size - new_width) / 2;
        int y = (output_size - new_height) / 2;
        symbol.copyTo(padded_symbol(cv::Rect(x, y, new_width, new_height)));
        std::string filename =  output_dir + "/" + "symbol_" + std::to_string(symbol_idx) + ".jpg";
        cv::imwrite(filename, padded_symbol);
        symbol_idx++;
    }

    std::cout << "Saved " << symbol_idx << " symbols\n";
}
