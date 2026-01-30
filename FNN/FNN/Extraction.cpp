// this Extraction.cpp file is made for the polynomial case, here we will take into account detecting "=" sign

#include "Extraction.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>   

using namespace cv;
using namespace std;

static float overlap1D(int a0, int a1, int b0, int b1){
    int lo = max(a0, b0);
    int hi = min(a1, b1);
    return (hi>lo) ? float(hi-lo) : 0.0f;
}

static float overlapRatioX(const Rect& A, const Rect& B){
    float ov = overlap1D(A.x, A.x + A.width, B.x, B.x + B.width);
    float denom = float(min(A.width, B.width));
    return (denom>0.0f) ? (ov/denom) : 0.0f;
}

// two small horizontal bars that form '='
static bool isEqualsPair(const Rect& A, const Rect& B){
    // here we are making sure that A is above B 
    Rect top = A, bot = B;
    if (top.y>bot.y) std::swap(top, bot);

    // both should be thin horizontal-ish
    auto thin = [](const Rect& r){
        return (r.height<=12 && r.width>=8 && r.width>=2*r.height);
    };
    if (!thin(top) || !thin(bot)) return false;

    //similar widths
    float wRatio = float(min(top.width, bot.width))/float(max(top.width, bot.width));
    if (wRatio<0.65f) return false;

    if (overlapRatioX(top, bot) < 0.65f) return false;

    // vertical gap should be small
    int gapY = bot.y-(top.y+top.height);
    if (gapY<0) gapY = 0; 
    if (gapY>18) return false;

    // they should be roughly aligned horizontally
    float cx1 = top.x+top.width*0.5f;
    float cx2 = bot.x+bot.width*0.5f;
    if (std::abs(cx1-cx2)>0.35f*float(max(top.width, bot.width))) return false;
    return true;
}

static vector<Rect> mergeEqualsBars(vector<Rect> rects){
    if (rects.size()<2) return rects;

    // we sort top-to-bottom then left-to-right for stable pairing
    sort(rects.begin(), rects.end(), [](const Rect& a, const Rect& b){
        if (a.y == b.y) return a.x<b.x;
        return a.y<b.y;
    });

    vector<int> used(rects.size(), 0);
    vector<Rect> out;
    out.reserve(rects.size());

    for (size_t i=0; i<rects.size(); i++){
        if (used[i]) continue;

        int bestJ=-1;
        int bestScore=-1;

        for (size_t j=i+1; j<rects.size(); j++){
            if (used[j]) continue;

            // quick reject far away in x
            if (std::abs((rects[i].x+rects[i].width/2) - (rects[j].x+rects[j].width/2)) >200)
                continue;

            if (isEqualsPair(rects[i], rects[j])){
                // score: bigger overlap and closer vertical distance
                Rect A = rects[i], B = rects[j];
                Rect top = A, bot = B;
                if (top.y>bot.y) std::swap(top, bot);

                int gapY = bot.y-(top.y+top.height);
                if (gapY<0) gapY = 0;

                int score = int(overlapRatioX(top, bot)*100.0f)-gapY;
                if (score>bestScore){
                    bestScore = score;
                    bestJ = (int)j;
                }
            }
        }

        if (bestJ != -1){
            used[i] = used[bestJ] = 1;
            out.push_back(rects[i] | rects[bestJ]); // merge into '=' box
        } else {
            used[i] = 1;
            out.push_back(rects[i]);
        }
    }
    return out;
}

// here is conservative merge for rare cases of split symbol pieces
static bool shouldMerge(const Rect& A, const Rect& B){
    float yov = overlap1D(A.y, A.y+A.height, B.y, B.y+B.height);
    float yovMin = min(A.height, B.height);
    float yRatio = (yovMin>0) ? (yov/yovMin) : 0.0f;

    int gap;
    if (B.x>=A.x) gap = B.x-(A.x+A.width);
    else gap = A.x-(B.x+B.width);
    return (yRatio>0.75f && gap>=0 && gap<= 2);
}

std::vector<cv::Rect> detectSymbols(const cv::Mat& binary){
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Rect> rects;
    rects.reserve(contours.size());

    // 1)we use bounding boxes + filter tiny noise
    for (const auto& c:contours){
        double area = contourArea(c);
        if (area<20) continue;
        Rect r = boundingRect(c);
        if (r.width<2 || r.height<2) continue;
        rects.push_back(r);
    }

    if (rects.empty()) return {};

    // 2)then we do merging
    rects = mergeEqualsBars(rects);

    // 3)after we sort left to right
    sort(rects.begin(), rects.end(), [](const Rect& a, const Rect& b){
        if (a.x==b.x) return a.y<b.y;
        return a.x<b.x;
    });

    // 4) we do conservative merge pass for split parts (just in case)
    vector<Rect> out;
    out.reserve(rects.size());
    Rect cur = rects[0];
    for (size_t i = 1; i < rects.size(); i++){
        if (shouldMerge(cur, rects[i])) cur = (cur | rects[i]);
        else {
            out.push_back(cur);
            cur = rects[i];
        }
    }
    out.push_back(cur);

    // 5) and repeat sort left to right
    sort(out.begin(), out.end(), [](const Rect& a, const Rect& b){
        if (a.x == b.x) return a.y < b.y;
        return a.x < b.x;
    });
    return out;
}

std::vector<cv::Mat> extractSymbols(const cv::Mat& im, const std::vector<cv::Rect>& rectangles){
    vector<Mat> symbols;
    symbols.reserve(rectangles.size());
    const int pad = 2;
    for (auto r : rectangles){
        r.x = max(0, r.x-pad);
        r.y = max(0, r.y-pad);
        r.width = min(im.cols-r.x, r.width+2*pad);
        r.height = min(im.rows-r.y, r.height+2*pad);
        symbols.push_back(im(r).clone());
    }
    return symbols;
}
