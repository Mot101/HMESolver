#include "Extraction1.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>

using namespace cv;
using namespace std;

std::vector<cv::Rect> detectSymbols(const cv::Mat &binary)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> rectangles;
    rectangles.reserve(contours.size());

    // keep small symbols (dot), but filter tiny noise
    const double MIN_AREA = 5.0;

    for (const auto &c : contours)
    {
        if (cv::contourArea(c) < MIN_AREA) continue;
        cv::Rect r = cv::boundingRect(c);
        rectangles.push_back(r);
    }

    // sort left-to-right
    std::sort(rectangles.begin(), rectangles.end(),
              [](const cv::Rect &a, const cv::Rect &b)
              {
                  if (a.x != b.x) return a.x < b.x;
                  return a.y < b.y;
              });

    return rectangles;
}

std::vector<cv::Mat> extractSymbols(const cv::Mat &im, const std::vector<cv::Rect> &rectangles)
{
    std::vector<cv::Mat> symbols;
    symbols.reserve(rectangles.size());

    const int pad = 5; // a bit bigger to avoid cutting digits (helps '3')

    for (auto rect : rectangles)
    {
        rect.x = std::max(0, rect.x - pad);
        rect.y = std::max(0, rect.y - pad);

        rect.width  = std::min(im.cols - rect.x, rect.width + 2 * pad);
        rect.height = std::min(im.rows - rect.y, rect.height + 2 * pad);

        symbols.push_back(im(rect).clone());
    }

    return symbols;
}
