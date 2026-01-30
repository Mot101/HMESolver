#include "Preprocessing.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv ; 
using namespace std; 
Mat preprocess(const string &filename)
{
    Mat image = imread(filename);
    if(image.empty())
    {
        cerr<<"error : Cannot read the image" <<filename << endl ; 
        return cv::Mat() ; 
    }

    //if we can read it correctly 
    Mat gray , binary ; 
    // changing the image into greyscale 
    cvtColor(image,gray,COLOR_BGR2GRAY);
    // we apply a guassian with a tiny kernel of size (3,3)
    // to reduce the noise and improve the thresholding quality 
    GaussianBlur(gray,gray, Size(3,3),0);
    //now it's black and white 
    threshold(gray,binary,0,255,THRESH_BINARY_INV|THRESH_OTSU);
    //the effect here is to focus on the main symbol
    Mat kernel=getStructuringElement(MORPH_RECT, Size(2,2));
    //remove tiny noise 
    morphologyEx(binary , binary , MORPH_OPEN ,kernel);
    return binary;
}