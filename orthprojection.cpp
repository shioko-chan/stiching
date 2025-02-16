
#include "orthprojection.h"
#include "triangulation.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <string.h>

int OrthProjection::init(const string modelFn)
{
    cout << "load model done!\n";
    return 0;
}

vector<float> OrthProjection::img2vector(const Mat &image)
{
    Mat rgbImage;
    vector<Mat> channels;
    cvtColor(image, rgbImage, COLOR_BGR2RGB);
    split(rgbImage, channels);
    vector<float> imgData;
    for (int c = 0; c < channels.size(); c++)
    {
        for (int h = 0; h < image.rows; h++)
        {
            for (int w = 0; w < image.cols; w++)
            {
                imgData.push_back(channels[c].at<uchar>(h, w)/255.0);
            }
        }
    }

    return imgData;
}

Mat OrthProjection::vector2img(const float *dat, size_t rows, size_t cols)
{
    int idx = 0, sz = rows * cols;
    float maxVal = -RAND_MAX;
    float minVal = RAND_MAX;
    vector<float> depthMap;
    for (idx = 0; idx < sz; idx++)
    {
        minVal = minVal > dat[idx] ? dat[idx] : minVal;
        maxVal = maxVal < dat[idx] ? dat[idx] : maxVal;
        depthMap.push_back(dat[idx]);
    }
    idx = 0;
    Mat img = Mat_<uchar>(rows, cols);
    for (int h = 0; h < img.rows; h++)
    {
        for (int w = 0; w < img.cols; w++)
        {
            img.at<u_char>(h, w) = (u_char)floor(((depthMap[idx] - minVal) * 255.0) / (maxVal - minVal));
            idx++;
        }
    }
    return img;
}

int OrthProjection::imageScale(cv::Mat &mat, string imgFn)
{
    float maxVal = 0, minVal = RAND_MAX;
    cv::Mat view = Mat_<u_char>(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
        {
            if (mat.at<float>(i, j) > maxVal)
            {
                maxVal = mat.at<float>(i, j);
            }
            if (mat.at<float>(i, j) < minVal)
            {
                minVal = mat.at<float>(i, j);
            }
        }
    }
    float val = 0;
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
        {
            val = (mat.at<float>(i, j) - minVal) / (maxVal - minVal);
            view.at<u_char>(i, j) = (u_char)floor(val * 255);
        }
    }
    cv::imshow("img", view);
    cv::imwrite(imgFn.c_str(), view);
    cv::waitKey(0);
    return 0;
}

void OrthProjection::inferno(Mat &depthMap)
{
    cv::Mat depthColor;
    cv::applyColorMap(depthMap, depthColor, cv::COLORMAP_INFERNO);
    cv::imshow("hi", depthColor);
    cv::imwrite("../inferno.jpg", depthColor);
}

Mat OrthProjection::predict(const string modelFn, const string imgFn)
{
    Mat img, blob;
    Mat predict;
    img = imread(imgFn);
    vector<float> imgData = img2vector(img);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DepthAnything");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    static Ort::Session depthSession(env, modelFn.c_str(), sessionOptions);

    vector<int64_t> inputShape{1, 3, img.rows, img.cols};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, imgData.data(), imgData.size(), inputShape.data(), inputShape.size());
    const char *input_names[] = {"image"};
    const char *output_names[] = {"depth"};
    Ort::RunOptions run_options;
    vector<Ort::Value> outputs = depthSession.Run(run_options, input_names, &inputTensor, 1, output_names, 1);
    float *map = (float *)outputs[0].GetTensorMutableData<void>();
    std::vector<int64_t> matShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    // for (auto it = matShape.begin(); it != matShape.end(); it++)
    // {
    //     cout << *it << endl;
    // }
    Mat view = vector2img(map, img.rows, img.cols);

    return view;
}

Mat OrthProjection::orthProjection(Mat &depthMap, string imgFn)
{
    Mat img = imread(imgFn);    
    int h = img.rows;
    int w = img.cols;    
    Mat orthProjectionMap(Size(w, h), img.type(), Scalar::all(0));
    float focal = 35.0;

    // double deepMax, deepMin;
    // Point minLoc, maxLoc;
    // minMaxLoc(depthMap, &deepMin, &deepMax, &minLoc, &maxLoc);

    float deepMax = 100.0, deepMin = 30.0;

    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.cols; ++j){
            int deep = (int)((255 - depthMap.at<uchar>(i, j)) / (255 / (deepMax - deepMin))) + deepMin;
            int h_direct = (i - h / 2);
            int w_direct = (j - w / 2);
            int y = (int)(h / 2 *((deep + deepMin) / 1.0 / deepMax) * sin(atan2(h_direct * 2 / 1.0 / h * 12, focal)) / sin(atan2(12, focal)) + h / 2);
            int x = (int)(w / 2 *((deep + deepMin) / 1.0 / deepMax) * sin(atan2(w_direct * 2 / 1.0 / w * 18, focal)) / sin(atan2(18, focal)) + w / 2);
            if(x >=0 && x < orthProjectionMap.cols && y >= 0 && y < orthProjectionMap.rows){
                orthProjectionMap.at<Vec3b>(y, x) = img.at<Vec3b>(i, j);
            }
            
        }
    }
    return orthProjectionMap;
}

void OrthProjection::test()
{
    string modelFn = "/media/media01/zyj/stiching/models/depth_anything_vitb14.onnx";
    // string imgFn = "/media/media01/zyj/stiching/imgs/scale/DJI_20230815135402_0024_V.JPG";
    string imgFn = "/media/media01/zyj/stiching/imgs/scale_8/DJI_0525_scale8.JPG";
    OrthProjection hi;
    Mat depthMap, orthProjectionMap;
    hi.init(modelFn);
    depthMap = hi.predict(modelFn, imgFn);
    Triangulation tirangulation;
    tirangulation.triangulation_3(depthMap);
    // inferno(result);
    // orthProjectionMap = hi.orthProjection(depthMap, imgFn);
    // imshow("img", orthProjectionMap);
    // waitKey(0);

}
