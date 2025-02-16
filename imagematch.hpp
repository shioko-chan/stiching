#ifndef IMAGEMATCH_H
#define IMAGEMATCH_H

#include "superpoint.h"
#include "lightglue.h"
#include "imagefile.hpp"
#include "registration.cpp"
#include <opencv2/stitching.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <cmath>

using namespace cv;

struct P2PMatch {
    public:
        int sx, sy;
        int ex, ey;
};

struct Offset {
    public:
        double offset_up;
        double offset_down;
        double offset_left;
        double offset_right;
};

struct Vector{
    double x, y;
    Vector(double _x, double _y) : x(_x), y(_y){}
};


class ImageMatch {

private:
    string supermodel_fn;
    string lightmodel_fn;
    SuperPoint *extractor;
    LightGlue *matcher;

public:
    int init(string superpt_model, string lightglu_model);
    void getMatch2Image(vector<string> imglist, vector<bool> isRotated);
    void stitching(vector<string> imglist, vector<bool> isRotated, vector<int> nn);
    
    cv::detail::MatchesInfo computeMatches(Mat& img1, Mat& img2);
    pair<double, cv::detail::MatchesInfo> bestQuality(const vector<pair<double, cv::detail::MatchesInfo>>& matchQuality);
    vector<bool> sparseImage(vector<bool> isRotated);

    int blendAll(vector<Mat> imglist, vector<Mat> matrix, vector<bool> isRotated);
    void blendSparse(vector<Mat> imglist, vector<Mat> matrix, vector<bool> isBlend);

    Offset createCanvas(vector<Mat> matrix);
    Mat rotateImage(const Mat &source, double angle);
    vector<Vector> computeVector(const vector<std::pair<double, double>>& geolocation);
    vector<double> computeImageAngleByGeolocation(const vector<Vector>& _vector);
    vector<double> computeImageAngleByMagnitudeSpectrum(const vector<string>& imglist);
    vector<bool> isRotated(vector<double> imageAngle1, vector<double> imageAngle2);
    vector<int> nearestNeighbor(vector<pair<double, double>> geolocation, vector<bool> isRotated);

    vector<P2PMatch> getMatchedPtPairs(Mat& img1, Mat& img2);

    static void test();
    void test2ImagesBlend(Mat& img1, Mat& img2, Mat& matrix, int index);

};

int ImageMatch::init(string superpt_model, string lightglu_model)
{
    supermodel_fn = superpt_model;
    lightmodel_fn = lightglu_model;

    extractor     = new SuperPoint(supermodel_fn);
    matcher       = new LightGlue(lightmodel_fn, cv::Stitcher::SCANS, 0.1);
    return 0;
}

void ImageMatch::getMatch2Image(vector<string> imglist, vector<bool> isRotated)
{
    vector<Mat> Matrix;
    vector<Mat> imgs;
    imgs.push_back(imread(imglist[0].c_str()));
    Mat identity_matrix = Mat::eye(3, 3, CV_64F);
    Matrix.push_back(identity_matrix);
    for(int k = 0; k < imglist.size() - 1; k++){
        Mat img1 = imread(imglist[k+1].c_str());
        Mat img2 = imread(imglist[k].c_str());  

        double angle = 0.0;
        Point2f src_center(img1.cols/2.0f, img1.rows/2.0f);
        double length = img1.cols, width = img1.rows;
        vector<pair<double, cv::detail::MatchesInfo>> matchQuality;
        imgs.push_back(img1);

        cv::detail::MatchesInfo Img2ImgMatch0 = computeMatches(img1, img2);
        matchQuality.push_back(pair<double, cv::detail::MatchesInfo>(angle, Img2ImgMatch0));

        if(isRotated[k] == true){
            angle = 90.0;
            Mat img1_90 = rotateImage(img1, angle);
            cv::detail::MatchesInfo Img2ImgMatch90 = computeMatches(img1_90, img2);
            matchQuality.push_back(pair<double, cv::detail::MatchesInfo>(angle, Img2ImgMatch90));

            angle = -90.0;
            Mat img1_n90 = rotateImage(img1, angle);
            cv::detail::MatchesInfo Img2ImgMatchN90 = computeMatches(img1_n90, img2);
            matchQuality.push_back(pair<double, cv::detail::MatchesInfo>(angle, Img2ImgMatchN90));

            angle = 180.0;
            Mat img1_180 = rotateImage(img1, angle);
            cv::detail::MatchesInfo Img2ImgMatch180 = computeMatches(img1_180, img2);
            matchQuality.push_back(pair<double, cv::detail::MatchesInfo>(angle, Img2ImgMatch180));

            img1_90.release();
            img1_n90.release();
            img1_180.release();

            // img1 = imread("/media/media01/zyj/stiching/imgs/1.JPG");
            // imshow("../img1.jpg", img1);
            // cv::imwrite("../img1.jpg", img1);
        }
        pair<double, cv::detail::MatchesInfo> result = bestQuality(matchQuality);
        angle = result.first;
        cv::detail::MatchesInfo Img2ImgMatch = result.second;
        std::cout << "matches count:" << Img2ImgMatch.matches.size() << std::endl;

        if(angle != 0.0){
            Mat rotateParam = getRotationMatrix2D(src_center, angle, 1.0);
            Mat pre_matrix = Mat::eye(3, 3, CV_64F);
            rotateParam.copyTo(pre_matrix(Rect(0, 0, 3, 2)));
            if(angle != 180){
                pre_matrix.at<double>(0,2) += (width / 2.0f - length / 2.0f);
                pre_matrix.at<double>(1,2) += (length / 2.0f - width / 2.0f);           
            }
            Img2ImgMatch.H = Img2ImgMatch.H * pre_matrix ;
        }

        // vector<DMatch> mt = Img2ImgMatch.getMatches();
        // // cout << "num:" << mt.size() << endl;
        // for(int i = 0; i < mt.size(); i++){
        //     int q = mt[i].queryIdx;
        //     int p = mt[i].trainIdx;
        //     P2PMatch oneMt;
        //     oneMt.sx = feat1.keypoints[q].pt.x;
        //     oneMt.sy = feat1.keypoints[q].pt.y;
        //     oneMt.ex = feat1.keypoints[p].pt.x;
        //     oneMt.ey = feat1.keypoints[p].pt.y;
        //     cout << oneMt.sx << "\t" << oneMt.sy << "\t" << oneMt.ex << "\t" << oneMt.ey << endl;
        // }

        // cout << Matrix.back() * Img2ImgMatch.H << endl;  

        Matrix.push_back(Matrix.back() * Img2ImgMatch.H);
        std::cout << k+1 << "->" << k+2 <<" images matched" << std::endl;
        test2ImagesBlend(img2, img1, Img2ImgMatch.H, k);//debug


        img1.release();
        img2.release();
    }

    blendAll(imgs, Matrix, isRotated);
}

void ImageMatch::stitching(vector<string> imglist, vector<bool> isRotated, vector<int> nn)
{
    vector<Mat> Matrix;
    vector<Mat> imgs;
    imgs.push_back(imread(imglist[0].c_str()));
    Mat identity_matrix = Mat::eye(3, 3, CV_64F);
    Matrix.push_back(identity_matrix);
    for(int k = 0; k < imglist.size() - 1; k++){
        Mat img1;
        Mat img2;
        if(nn[k+1] == -1){
            img1 = imread(imglist[k+1].c_str());
            img2 = imread(imglist[k].c_str());

            double angle = 0.0;
            Point2f src_center(img1.cols/2.0f, img1.rows/2.0f);
            double length = img1.cols, width = img1.rows;
            vector<pair<double, cv::detail::MatchesInfo>> matchQuality;
            imgs.push_back(img1);

            cv::detail::MatchesInfo Img2ImgMatch0 = computeMatches(img1, img2);
            matchQuality.push_back(pair<double, cv::detail::MatchesInfo>(angle, Img2ImgMatch0));

            if(isRotated[k] == true){
                angle = 90.0;
                Mat img1_90 = rotateImage(img1, angle);
                cv::detail::MatchesInfo Img2ImgMatch90 = computeMatches(img1_90, img2);
                matchQuality.push_back(pair<double, cv::detail::MatchesInfo>(angle, Img2ImgMatch90));

                angle = -90.0;
                Mat img1_n90 = rotateImage(img1, angle);
                cv::detail::MatchesInfo Img2ImgMatchN90 = computeMatches(img1_n90, img2);
                matchQuality.push_back(pair<double, cv::detail::MatchesInfo>(angle, Img2ImgMatchN90));

                angle = 180.0;
                Mat img1_180 = rotateImage(img1, angle);
                cv::detail::MatchesInfo Img2ImgMatch180 = computeMatches(img1_180, img2);
                matchQuality.push_back(pair<double, cv::detail::MatchesInfo>(angle, Img2ImgMatch180));

                img1_90.release();
                img1_n90.release();
                img1_180.release();

            }

            pair<double, cv::detail::MatchesInfo> result = bestQuality(matchQuality);
            angle = result.first;
            cv::detail::MatchesInfo Img2ImgMatch = result.second;
            std::cout << "matches count:" << Img2ImgMatch.matches.size() << std::endl;

            if(angle != 0.0){
                Mat rotateParam = getRotationMatrix2D(src_center, angle, 1.0);
                Mat pre_matrix = Mat::eye(3, 3, CV_64F);
                rotateParam.copyTo(pre_matrix(Rect(0, 0, 3, 2)));
                if(angle != 180){
                    pre_matrix.at<double>(0,2) += (width / 2.0f - length / 2.0f);
                    pre_matrix.at<double>(1,2) += (length / 2.0f - width / 2.0f);           
                }
                Img2ImgMatch.H = Img2ImgMatch.H * pre_matrix ;
            }
            Matrix.push_back(Matrix.back() * Img2ImgMatch.H);
            std::cout <<  k+2 << " image matched" << std::endl;
            // test2ImagesBlend(img2, img1, Img2ImgMatch.H, k);//debug

        }else{
            img1 = imread(imglist[k+1].c_str());
            img2 = imread(imglist[nn[k+1]].c_str());
            imgs.push_back(img1);

            Point2f src_center(img1.cols/2.0f, img1.rows/2.0f);

            double angle = 180.0;
            Mat img1_180 = rotateImage(img1, angle);
            cv::detail::MatchesInfo Img2ImgMatch = computeMatches(img1_180, img2);
            std::cout << "matches count:" << Img2ImgMatch.matches.size() << std::endl;
            Mat rotateParam = getRotationMatrix2D(src_center, angle, 1.0);
            Mat pre_matrix = Mat::eye(3, 3, CV_64F);
            rotateParam.copyTo(pre_matrix(Rect(0, 0, 3, 2)));
            Img2ImgMatch.H = Img2ImgMatch.H * pre_matrix;
            Matrix.push_back(Matrix[nn[k+1]] * Img2ImgMatch.H);
            std::cout <<  k+2 << " image matched" << std::endl;      

            img1_180.release();
            // test2ImagesBlend(img2, img1, Img2ImgMatch.H, k);//debug
        }
        

        img1.release();
        img2.release();
    }
    vector<bool> isBlend = sparseImage(isRotated);
    blendSparse(imgs, Matrix, isBlend);
    // blendAll(imgs, Matrix, isRotated);
}

cv::detail::MatchesInfo ImageMatch::computeMatches(Mat& img1, Mat& img2){
    ImageFeatures feat1;
    ImageFeatures feat2;
    cv::detail::MatchesInfo Img2ImgMatch;
    feat1.img_size.height = img1.rows;
    feat1.img_size.width = img1.cols;
    feat2.img_size.height = img2.rows;
    feat2.img_size.width = img2.cols;

    this->extractor->detectAndCompute(img1, feat1.keypoints, feat1.descriptors, false);
    this->extractor->detectAndCompute(img2, feat2.keypoints, feat2.descriptors, false);
    // cout << feat1.descriptors.size() << "\t" << feat2.descriptors.size() << endl;
    //this->matcher->doMatch(feat1, feat2, Img2ImgMatch);
    this->matcher->match(feat1, feat2, Img2ImgMatch);

    feat1.keypoints.clear();
    feat2.keypoints.clear();
    feat1.descriptors.release();
    feat2.descriptors.release();

    return Img2ImgMatch;
}

vector<P2PMatch> ImageMatch::getMatchedPtPairs(Mat& img1, Mat& img2){
    ImageFeatures feat1;
    ImageFeatures feat2;
    cv::detail::MatchesInfo Img2ImgMatch;
    feat1.img_size.height = img1.rows;
    feat1.img_size.width = img1.cols;
    feat2.img_size.height = img2.rows;
    feat2.img_size.width = img2.cols;

    this->extractor->detectAndCompute(img1, feat1.keypoints, feat1.descriptors, false);
    this->extractor->detectAndCompute(img2, feat2.keypoints, feat2.descriptors, false);
    // cout << feat1.descriptors.size() << "\t" << feat2.descriptors.size() << endl;
    //this->matcher->doMatch(feat1, feat2, Img2ImgMatch);
    this->matcher->match(feat1, feat2, Img2ImgMatch);
    
    vector<P2PMatch> ptPairs;
    vector<DMatch> dmatch = Img2ImgMatch.getMatches();
    for(int i = 0; i < dmatch.size(); i++){
        P2PMatch p2p;
        p2p.sx = feat1.keypoints[dmatch[i].queryIdx].pt.x;
        p2p.sy = feat1.keypoints[dmatch[i].queryIdx].pt.y;
        p2p.ex = feat2.keypoints[dmatch[i].trainIdx].pt.x;
        p2p.ey = feat2.keypoints[dmatch[i].trainIdx].pt.y;
        ptPairs.push_back(p2p);
    }

    feat1.keypoints.clear();
    feat2.keypoints.clear();
    feat1.descriptors.release();
    feat2.descriptors.release();

    return ptPairs;
}

pair<double, cv::detail::MatchesInfo> ImageMatch::bestQuality(const vector<pair<double, cv::detail::MatchesInfo>>& matchQuality){
    double angle = matchQuality[0].first;
    cv::detail::MatchesInfo best =matchQuality[0].second;

    for(const auto& pair : matchQuality){
        if(pair.second.matches.size() > best.matches.size()){
            best = pair.second;
            angle = pair.first;
        }
    }
    
    return pair<double, cv::detail::MatchesInfo>(angle, best);
}

vector<bool> ImageMatch::sparseImage(vector<bool> isRotated){
    vector<bool> isBlend;
    isBlend.push_back(true);
    bool isNeedImage = false;
    bool inZeroGroup = false;
    for(int i = 0; i < isRotated.size(); i++){
        if(isRotated[i] == 0){
            if(!inZeroGroup){
                inZeroGroup = true;
                isNeedImage = !isNeedImage;
            }

            if(isNeedImage){
                isBlend.push_back(true);
            }else{
                isBlend.push_back(false);
            }
        }else{
            inZeroGroup = false;
            isBlend.push_back(true);
        }
    }

    // for(int i = 0; i < isRotated.size(); i++){
    //     cout << isRotated[i] << '\t' << isBlend[i+1] << endl;
    // }

    return isBlend;

}

int ImageMatch::blendAll(vector<Mat> imglist, vector<Mat> matrix, vector<bool> isRotated)
{
    Offset offset = createCanvas(matrix);
    Mat img_1 = imglist[0];
    double canvasWidth = offset.offset_right - offset.offset_left + 2 * std::sqrt(std::pow(img_1.cols,2)+std::pow(img_1.rows,2));
    double canvasHight = offset.offset_down - offset.offset_up + 2 * std::sqrt(std::pow(img_1.cols,2)+std::pow(img_1.rows,2));
    Size canvasSize(canvasWidth, canvasHight);
    Mat intermediateImage(canvasSize, img_1.type(), cv::Scalar::all(0));

    //TODO
    int offset_x = canvasWidth - offset.offset_right - std::sqrt(std::pow(img_1.cols,2)+std::pow(img_1.rows,2)) - 0.5 * img_1.cols;
    int offset_y = canvasHight - offset.offset_down - std::sqrt(std::pow(img_1.cols,2)+std::pow(img_1.rows,2)) - 0.5 * img_1.rows;

    for(int i = 0; i < matrix.size(); i++){
        Mat img = imglist[i]; 
        Mat panorama(canvasSize, img.type(), cv::Scalar::all(0));

        matrix[i].at<double>(0,2) +=  offset_x;
        matrix[i].at<double>(1,2) +=  offset_y;

        warpPerspective(img, panorama, matrix[i], canvasSize, INTER_NEAREST);

        for(int y = 0; y < intermediateImage.rows; y++){
            for(int x = 0; x < intermediateImage.cols; x++){
                Vec3b &pixel = panorama.at<Vec3b>(y, x);
                Vec3b &resultPixel = intermediateImage.at<Vec3b>(y, x);
                for(int c = 0; c < 3; c++){
                    if(pixel[c] != 0 && resultPixel[c] != 0){
                        // resultPixel[c] = (pixel[c] + resultPixel[c]) / 2;
                        resultPixel[c] = pixel[c];
                    }else{
                        resultPixel[c] = pixel[c] + resultPixel[c];
                    }
                }
            }
        }
        
    }

    cv::imwrite("../result.jpg", intermediateImage);

    return 0;
}

void ImageMatch::blendSparse(vector<Mat> imglist, vector<Mat> matrix, vector<bool> isBlend)
{
    Offset offset = createCanvas(matrix);
    Mat img_1 = imglist[0];
    double canvasWidth = offset.offset_right - offset.offset_left + 2 * std::sqrt(std::pow(img_1.cols,2)+std::pow(img_1.rows,2));
    double canvasHight = offset.offset_down - offset.offset_up + 2 * std::sqrt(std::pow(img_1.cols,2)+std::pow(img_1.rows,2));
    Size canvasSize(canvasWidth, canvasHight);
    Mat intermediateImage(canvasSize, img_1.type(), cv::Scalar::all(0));

    int offset_x = canvasWidth - offset.offset_right - std::sqrt(std::pow(img_1.cols,2)+std::pow(img_1.rows,2)) - 0.5 * img_1.cols;
    int offset_y = canvasHight - offset.offset_down - std::sqrt(std::pow(img_1.cols,2)+std::pow(img_1.rows,2)) - 0.5 * img_1.rows;

    for(int i = 0; i < matrix.size(); i++){
        Mat img = imglist[i]; 
        Mat panorama(canvasSize, img.type(), cv::Scalar::all(0));

        matrix[i].at<double>(0,2) +=  offset_x;
        matrix[i].at<double>(1,2) +=  offset_y;

        // cv::Point2f point(img.cols / 2, img.rows / 2);
        // cv::Mat homogeneousPoint = (cv::Mat_<double>(3, 1) << point.x, point.y, 1);
        
        warpPerspective(img, panorama, matrix[i], canvasSize, INTER_NEAREST);
        // cv::Mat resultPoint = matrix[i] * homogeneousPoint;
        
        // double centerX = resultPoint.at<double>(0, 0);
        // double centerY = resultPoint.at<double>(1, 0);

        if(i % 3 == 0 && isBlend[i] == true){//debug
            for(int y = 0; y < intermediateImage.rows; y++){
                for(int x = 0; x < intermediateImage.cols; x++){
                    Vec3b &pixel = panorama.at<Vec3b>(y, x);
                    Vec3b &resultPixel = intermediateImage.at<Vec3b>(y, x);
                    for(int c = 0; c < 3; c++){
                        if(pixel[c] != 0 && resultPixel[c] != 0){
                            // // weighted processing
                            // double distance = sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
                            // double maxDistance = sqrt((img.cols / 2) * (img.cols / 2) + (img.rows / 2) * (img.rows / 2));
                            // double weight = 1 - (distance / maxDistance);
                            // resultPixel[c] = pixel[c] * weight + resultPixel[c] * (1 - weight);

                            // resultPixel[c] = (pixel[c] + resultPixel[c]) / 2;
                            resultPixel[c] = pixel[c];
                        }else{
                            resultPixel[c] = pixel[c] + resultPixel[c];
                        }
                    }
                }
            }
        }
        
    }

    cv::imwrite("../result.jpg", intermediateImage);
}

Offset ImageMatch::createCanvas(vector<Mat> matrix)
{
    double offset_x, offset_y;
    vector<double> right, left, up, down;
    Offset offset;
    for(int i = 0; i < matrix.size(); i++){
        offset_x = matrix[i].at<double>(0,2);
        offset_y = matrix[i].at<double>(1,2);
        offset_x > 0 ? right.push_back(offset_x) : left.push_back(offset_x);
        offset_y > 0 ? down.push_back(offset_y) : up.push_back(offset_y);
    }
    auto maxElement1 = std::max_element(right.begin(), right.end());
    maxElement1 != right.end() ? offset.offset_right = *maxElement1 : offset.offset_right = 0;
    auto maxElement2 = std::max_element(down.begin(), down.end());
    maxElement2 != down.end() ? offset.offset_down = *maxElement2 : offset.offset_down = 0;
    auto maxElement3 = std::min_element(left.begin(), left.end());
    maxElement3 != left.end() ? offset.offset_left = *maxElement3 : offset.offset_left = 0;
    auto maxElement4 = std::min_element(up.begin(), up.end());
    maxElement4 != up.end() ? offset.offset_up = *maxElement4 : offset.offset_up = 0;

    return offset;
}

Mat ImageMatch::rotateImage(const Mat &source, double angle)
{
    Point2f src_center(source.cols/2.0f, source.rows/2.0f);
    Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    double bound = std::sqrt(std::pow(source.cols,2)+std::pow(source.rows,2));

    rot_mat.at<double>(0,2) += bound/2 - src_center.x;
    rot_mat.at<double>(1,2) += bound/2 - src_center.y;
    Mat dst;
    warpAffine(source, dst, rot_mat, Size(bound, bound));

    if(angle == 90.0 || angle == -90.0){
        int x = (dst.cols - source.rows) / 2; x = max(x,0);
        int y = (dst.rows - source.cols) / 2; y = max(y,0);
        Rect crop_region(x, y, source.rows, source.cols);
        Mat cropped_image = dst(crop_region);    
        return cropped_image;
    }else{
        int x = (dst.cols - source.cols) / 2; x = max(x,0);
        int y = (dst.rows - source.rows) / 2; y = max(y,0);
        Rect crop_region(x, y, source.cols, source.rows);
        Mat cropped_image = dst(crop_region);
        return cropped_image;
    }
}

vector<Vector> ImageMatch::computeVector(const vector<std::pair<double, double>>& geolocation){
    vector<Vector> _vector;
    _vector.push_back(Vector(0.0, 0.0));
    for(int i = 0; i < geolocation.size() - 1; i++){
        _vector.push_back(Vector(geolocation[i+1].first - geolocation[i].first, geolocation[i+1].second - geolocation[i].second));
    }
    return _vector;
}

vector<double> ImageMatch::computeImageAngleByGeolocation(const vector<Vector>& _vector){
    vector<double> imageAngle;
    for(int i = 0; i < _vector.size() - 1; i++){
        double dotProduct =  _vector[i].x * _vector[i+1].x + _vector[i].y * _vector[i+1].y;
        double magnitudeA = std::sqrt(_vector[i].x * _vector[i].x + _vector[i].y * _vector[i].y);
        double magnitudeB = std::sqrt(_vector[i+1].x * _vector[i+1].x + _vector[i+1].y * _vector[i+1].y);
        double cosTheta = dotProduct / (magnitudeA * magnitudeB);
        double crossProduct = _vector[i].x * _vector[i+1].y - _vector[i].y * _vector[i+1].x;
        imageAngle.push_back(-crossProduct / abs(crossProduct) * std::acos(cosTheta) * 180.0 / M_PI);
    }
    return imageAngle;
}

vector<double> ImageMatch::computeImageAngleByMagnitudeSpectrum(const vector<string>& imglist){
    vector<double> imageAngle;
    Registration re;
    for(int i = 0; i < imglist.size() - 1; i++){
        cv::Mat img1 = cv::imread(imglist[i], 1);
        cv::Mat img2 = cv::imread(imglist[i+1], 1);
        cv::Mat greyImg1, greyImg2;
        cv::Mat dst1, dst2;
        float theta = 0;
        cv::cvtColor(img1, greyImg1, cv::COLOR_BGRA2GRAY);
        cv::cvtColor(img2, greyImg2, cv::COLOR_BGRA2GRAY);
        greyImg1.convertTo(dst1, CV_32FC1);
        greyImg2.convertTo(dst2, CV_32FC1);
        theta  = re.estimateRotation(dst1, dst2);
        imageAngle.push_back(theta);        
    }

    return imageAngle; 
}

vector<bool> ImageMatch::isRotated(vector<double> imageAngle1, vector<double> imageAngle2){
    vector<bool> isRotated;
    imageAngle1[0] = 0.0;
    for(int i = 0; i < imageAngle1.size(); i++){
        if(fabs(imageAngle1[i]) > 10.0 || fabs(imageAngle2[i]) > 3.0){
            isRotated.push_back(true);
        }else{
            isRotated.push_back(false);            
        }

    }
    return isRotated;
}

vector<int> ImageMatch::nearestNeighbor(vector<pair<double, double>> geolocation, vector<bool> isRotated){
    vector<int> nn; 
    nn.push_back(-1);
    int i = 0, end = 0;
    while(isRotated[i] != true){
        nn.push_back(-1);   
        i++;
    }
    for(i; i < isRotated.size(); i++){
        if(isRotated[i] == true){
            nn.push_back(-1);
            end = i;
        }
        else{
            double x = geolocation[i+1].first;
            double y = geolocation[i+1].second;
            double minDistance = RAND_MAX;
            int minIdx = 0;
            for(int k = 0; k < end; k++){
                double distance = std::sqrt(std::pow(x - geolocation[k].first, 2) + std::pow(y - geolocation[k].second, 2));
                if(distance < minDistance){
                    minDistance = distance;
                    minIdx = k;
                }
            }
            nn.push_back(minIdx);
        }
    }
    
    vector<int> result = nn;
    if(nn.size() > 6){
        for(int j = 2; j < nn.size() - 2; j++){
            if(nn[j-2] == -1 || nn[j+2] == -1){
                result[j] = -1;
            }else{
                if(j % 2 != 0){
                    result[j] = -1;
                }
            }
        }        
    }

    return result;
}

void ImageMatch::test2ImagesBlend(Mat& img1, Mat& img2, Mat& matrix, int index){
    double canvasWidth = 1000.0;
    double canvasHight = 800.0;
    Size canvasSize(canvasWidth, canvasHight);
    
    //TODO
    int offset_x = canvasWidth / 2 - img1.cols / 2;
    int offset_y = canvasHight / 2 - img1.rows / 2;

    Mat intermediateImage(canvasSize, img1.type(), cv::Scalar::all(0));
    Mat panorama(canvasSize, img1.type(), cv::Scalar::all(0));

    Mat eMatrix = Mat::eye(3, 3, CV_64F);

    matrix.at<double>(0,2) +=  offset_x;
    matrix.at<double>(1,2) +=  offset_y;
    eMatrix.at<double>(0,2) +=  offset_x;
    eMatrix.at<double>(1,2) +=  offset_y;

    warpPerspective(img1, intermediateImage, eMatrix, canvasSize);
    warpPerspective(img2, panorama, matrix, canvasSize);

    for(int y = 0; y < intermediateImage.rows; y++){
        for(int x = 0; x < intermediateImage.cols; x++){
            Vec3b &pixel = panorama.at<Vec3b>(y, x);
            Vec3b &resultPixel = intermediateImage.at<Vec3b>(y, x);
            for(int c = 0; c < 3; c++){
                if(pixel[c] != 0 && resultPixel[c] != 0){
                    resultPixel[c] = (pixel[c] + resultPixel[c]) / 2;
                    // resultPixel[c] = pixel[c];
                }else{
                    resultPixel[c] = pixel[c] + resultPixel[c];
                }
            }
        }
    }
        
    cv::imwrite("../70m/result" + std::to_string(index) + string(".jpg"), intermediateImage);
}

void ImageMatch::test()
{
    string folderPath = "/media/media01/zyj/stiching/imgs/2image";
    string supermodel = "/media/media01/zyj/stiching/models/superpoint.onnx";
    string lightmodel = "/media/media01/zyj/stiching/models/superpoint_lightglue.onnx";
    vector<string> imglist = getImgList(folderPath);
    
    ImageMatch match;
    vector<pair<double, double>> geolocation;
    for(int i = 0; i < imglist.size(); i++){
        geolocation.push_back(get_image_info(imglist[i]));
    }

    vector<Vector> _vector = match.computeVector(geolocation);
    //imageAnle[i]:stores the angle required to rotate the i-th picture to the previous picture, and counterclockwise is positive.
    vector<double> imageAngle1 = match.computeImageAngleByGeolocation(_vector);
    vector<double> imageAngle2 = match.computeImageAngleByMagnitudeSpectrum(imglist);
    vector<bool> isRotated = match.isRotated(imageAngle1, imageAngle2);

    vector<int> nn = match.nearestNeighbor(geolocation, isRotated);

    match.init(supermodel, lightmodel);
    match.getMatch2Image(imglist, isRotated);
    // match.stitching(imglist, isRotated, nn);

    // // match.blendAll(imglist, matrix);
}

#endif