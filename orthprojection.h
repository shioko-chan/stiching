#ifndef ORTHPROJECTION_H
#define ORTHPROJECTION_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace Ort;
using namespace cv;

class OrthProjection {
public:
    int init(const string modelFn);
    vector<float> img2vector(const Mat& image);
    Mat vector2img(const float *dat, size_t rows, size_t cols);
    static void inferno(Mat &depthMap);
    static int imageScale(cv::Mat &mat, string imgFn);
    Mat predict(const string modelFn, const string imgFn);
    Mat orthProjection(Mat &depthMap, string imgFn);
    static void test();

};

#endif