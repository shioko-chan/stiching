#ifndef IMAGERETIFY_H
#define IMAGERETIFY_H

#include "imagematch.hpp"
#include "superpoint.h"
#include "lightglue.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <cmath>

// using namespace detail;

class ImageRetify {
    
    public:
        /**visualize intermediate image results**/
    static int imageScale(cv::Mat &mat, string imgFn)
    {
        float maxVal = 0, minVal = RAND_MAX;
        cv::Mat view = Mat_<u_char>(mat.rows, mat.cols);
        for(int i = 0; i < mat.rows; i++)
        {
            for (int j = 0; j < mat.cols; j++)
            {
                if( mat.at<float>(i, j) > maxVal)
                {
                    maxVal = mat.at<float>(i, j);
                }
                if( mat.at<float>(i, j) < minVal)
                {
                    minVal = mat.at<float>(i, j);
                }
            }
        }
        float val = 0;
        for(int i = 0; i < mat.rows; i++)
        {
            for (int j = 0; j < mat.cols; j++)
            {
                val = (mat.at<float>(i, j)-minVal)/(maxVal - minVal);
                view.at<u_char>(i, j) = (u_char)floor(val*255);
            }
        }
        // cout << maxVal << "\t" << minVal << endl;
        imshow("img", view);
        imwrite(imgFn.c_str(), view);
        waitKey(0);
        return 0;
    }

    static int retifyImgPair(cv::Mat &img1, cv::Mat &img2, MatchesInfo &img2imgMatch, cv::Mat &retImg1, cv::Mat &retImg2)
    {
        string supermodel = "/media/media01/zyj/stiching/models/superpoint.onnx";
        string lightmodel = "/media/media01/zyj/stiching/models/superpoint_lightglue.onnx";
        ImageMatch match;
        match.init(supermodel, lightmodel);
        vector<P2PMatch> ptPairs = match.getMatchedPtPairs(img1, img2);
        vector<Point2f> img1Pts, img2Pts;
        cv::Mat rimg1(img1.size(), img1.type());
        cv::Mat rimg2(img1.size(), img1.type());
        cv::Mat F, greyImg1, greyImg2;
        for(vector<P2PMatch>::iterator it = ptPairs.begin(); it != ptPairs.end(); it++)
        {
            Point2f pt1;
            Point2f pt2;
            pt1.x = it->sx;
            pt1.y = it->sy;
            pt2.x = it->ex;
            pt2.y = it->ey;
            // cout << pt1.x << '\t' << pt1.y << '\t' << pt2.x << '\t' << pt2.y << endl;
            // cout << pt1.x - pt2.x<< '\t' << pt1.y - pt2.y << endl;
            img1Pts.push_back(pt1);
            img2Pts.push_back(pt2);
        }
        F = findHomography(img1Pts, img2Pts, cv::RANSAC, 5.0);

        warpPerspective(img1, rimg1, F, img1.size(),1);

        cv::cvtColor(img2, greyImg1, COLOR_BGR2GRAY);
        cv::cvtColor(rimg1, greyImg2, COLOR_BGR2GRAY);
        cv::imshow("greyImg2", greyImg2);
        cv::waitKey(0);

        cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(16,15);
        cv::Mat disparity;
        stereo->compute(greyImg1, greyImg2, disparity);

        
        cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8U);
        // cout << disparity << endl;
        cv::imwrite("../disparity.jpg", disparity);

        // //cv::Mat disparity = greyImg1 - greyImg2;
        // cv::Mat disparity(greyImg1.size(), CV_32FC1);
        // for(int i = 0; i < greyImg1.rows; i++)
        // {
        //     for(int j = 0; j < greyImg1.cols; j++)
        //     {
        //        float a = (int)greyImg1.at<u_char>(i, j) - (int)greyImg2.at<u_char>(i, j);
        //     //    cout << a << " ";
        //        disparity.at<float>(i, j) = a;
        //     }
        //     // cout << endl;
        // }

        // // cout << disparity << endl;
        // imageScale(disparity, "./disparity.jpg");
        return 0;
    }

    static void test()
    {
        string imgFn1 = "/media/media01/zyj/stiching/imgs/2image/DJI_20230815135249_0001_V.JPG";
        // string imgFn2 = "/media/media01/zyj/stiching/imgs/2image/DJI_20230815135253_0002_V.JPG";
        string imgFn2 = "/media/media01/zyj/stiching/imgs/2image/DJI_20230815135256_0003_V.JPG";

        Mat img1 = imread(imgFn1.c_str());
        Mat img2 = imread(imgFn2.c_str());
        Mat rimg1, rimg2;
        MatchesInfo img2imgMatch;
        easyexif::EXIFInfo info = getCameraInfo(imgFn1);
        cout << info.FocalLength << endl;
        
        retifyImgPair(img1, img2, img2imgMatch, rimg1, rimg2);
    }
};

#endif