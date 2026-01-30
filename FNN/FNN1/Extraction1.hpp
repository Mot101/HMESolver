#pragma once // replace the ifndef , define and endif 
#include <opencv2/opencv.hpp>
#include <string>

std::vector<cv::Rect> detectSymbols(const cv::Mat &binary);
std::vector<cv::Mat> extractSymbols(const cv::Mat &binary , const std::vector<cv::Rect> &rects);
