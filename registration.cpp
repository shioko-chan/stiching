#ifndef REGISTRATION_HPP
#define REGISTRATION_HPP

#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fftw3.h>
#include <iostream>
#include <string>
#include <cmath>

/***
 * 
 * @function: this class is responsible for rotation estimation 
 * between two images in the same size
 * 
 * @author: Wan-Lei Zhao
 * 
 * @date: 2024-3-20
 * 
 * **/

using namespace std;
using namespace cv;

struct VComplex
{
public:
    double re, im;
};

class Registration
{
    public:
    /**perform element-wise dot-product on the matrix, currently not being used**/
    static cv::Mat eWiseHamming(cv::Mat &img1, cv::Mat &hWin)
    {
        cv::Mat hImg = cv::Mat_<float>(img1.rows, img1.cols);
        for (unsigned i = 0; i < img1.rows; i++)
        {
            for (unsigned j = 0; j < img1.cols; j++)
            {
                hImg.at<float>(i, j) = img1.at<u_char>(i, j) * hWin.at<float>(i, j);
            }
        }
        return hImg;
    }

    /**perform element-wise absolute on the complex matrix**/
    static cv::Mat eWiseAbs(VComplex **srcMat, int nrow, int ncol)
    {
        cv::Mat result = cv::Mat_<float>(nrow, ncol);

        for (int i = 0; i < nrow; i++)
        {
            for (int j = 0; j < ncol; j++)
            {
                result.at<float>(i, j) = (float)sqrt(srcMat[i][j].re * srcMat[i][j].re + srcMat[i][j].im * srcMat[i][j].im);
            }
        }

        return result;
    }

    /**perform element-wise log on the matrix**/
    static int eWiseLog(cv::Mat &mat)
    {
        for (unsigned i = 0; i < mat.rows; i++)
        {
            for (unsigned j = 0; j < mat.cols; j++)
            {
                mat.at<float>(i, j) = log(mat.at<float>(i, j));
            }
        }
        return 0;
    }

    /**centralize 2D fourier transform**/
    static int fftshift2d(VComplex **srcMat, int nrow, int ncol, VComplex **dstMat)
    {
        int ir = 0, ic = 0;
        for (int i = 0; i < nrow; i++)
        {
            for (int j = 0; j < ncol; j++)
            {
                ir = (i + nrow / 2) % nrow;
                ic = (j + ncol / 2) % ncol;
                dstMat[ir][ic].re = srcMat[i][j].re;
                dstMat[ir][ic].im = srcMat[i][j].im;
            }
        }

        return 0;
    }

    static int printComplexMat(VComplex **srcMat, int nrow, int ncol)
    {
        cout << "----------------------------------\n";
        for (int i = 0; i < nrow; i++)
        {
            for (int j = 0; j < ncol; j++)
            {
                cout << "(" << srcMat[i][j].re << ", " << srcMat[i][j].im << ") ";
            }
            cout << endl;
        }
        cout << "----------------------------------\n";
        return 0;
    }

    /**extract out nrow of a matrix**/
    static cv::Mat getHalfMat(cv::Mat &img, int nrow)
    {
        int hcols = img.cols;
        cv::Mat mymat = Mat_<float>(nrow, hcols);

        for(int i = 0; i < nrow; i++)
        {
            for(int j = 0; j < hcols; j++)
            {
                mymat.at<float>(i,j) = img.at<float>(i,j);
            }
        }

        return mymat;
    }

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
        imshow("img", view);
        imwrite(imgFn.c_str(), view);
        waitKey(0);
        return 0;
    }
    
    /**allocate 2D array memory for complex numbers**/
    static VComplex **allocComplexMat(int nrow, int ncol)
    {
        assert(nrow > 0 && ncol > 0);
        VComplex **cmat = new VComplex *[nrow];
        for (int i = 0; i < nrow; i++)
        {
            cmat[i] = new VComplex[ncol];
            for (int j = 0; j < ncol; j++)
            {
                cmat[i][j].im = cmat[i][j].im = 0;
            }
        }
        return cmat;
    }
    /**release 2D array memory for complex numbers**/
    static void deallocComplexMat(VComplex **cmat, int nrow)
    {
        for (int i = 0; i < nrow; i++)
        {
            delete[] cmat[i];
            cmat[i] = nullptr;
        }
        delete[] cmat;
        cmat = nullptr;
    }
    /**perform 2D discrete fourier transform**/
    static VComplex **runfft2d(cv::Mat mat)
    {
        size_t nrow = mat.rows;
        size_t ncol = mat.cols;
        size_t idx = 0;
        double *tmpMat = fftw_alloc_real(nrow * ncol);
        VComplex **result = new VComplex *[nrow];
        for (int i = 0; i < nrow; i++)
        {
            result[i] = new VComplex[ncol];
            for (int j = 0; j < ncol; j++)
            {
                result[i][j].im = result[i][j].im = 0;
            }
        }
        /**/
        fftw_complex *outMat = fftw_alloc_complex(nrow * (ncol / 2 + 1));
        for (unsigned i = 0; i < mat.rows; i++)
        {
            for (unsigned j = 0; j < mat.cols; j++)
            {
                tmpMat[idx] = mat.at<float>(i, j);
                idx++;
            }
        } /**/

        
        fftw_plan p = fftw_plan_dft_r2c_2d(nrow, ncol, tmpMat, outMat, FFTW_ESTIMATE);
        fftw_execute(p);
        int ir = 0, ic = 0, loc = 0, xcol = ncol / 2 + 1;
        float sign = 1;
        for (int i = 0; i < nrow; i++)
        {
            for (int j = 0; j < ncol; j++)
            {
                if (j > ncol / 2)
                {
                    ir = (nrow - i) % nrow;
                    ic = (ncol - j) % ncol;
                    sign = -1;
                }
                else
                {
                    ir = i;
                    ic = j;
                    sign = 1;
                }
                result[i][j].re = *outMat[ir * xcol + ic];
                result[i][j].im = *(outMat[ir * xcol + ic] + 1) * sign;
            }
        }
        // printComplexMat(result, nrow, ncol);
        fftw_destroy_plan(p);
        fftw_free(outMat);
        fftw_free(tmpMat);

        return result;
    }

    /**estimate the rotation betweeen two images in the same size**/
    static float estimateRotation(cv::Mat &img1, cv::Mat &img2)
    {
        VComplex **fftImg1 = runfft2d(img1);
        VComplex **fftImg2 = runfft2d(img2);
        VComplex **shifted = allocComplexMat(img1.rows, img1.cols);
        cv::Mat polarMat1 = cv::Mat_<float>(img1.rows, img1.cols);
        cv::Mat polarMat2 = cv::Mat_<float>(img1.rows, img1.cols);

        fftshift2d(fftImg1, img1.rows, img1.cols, shifted);
        cv::Mat logMat1 = eWiseAbs(shifted, img1.rows, img1.cols);
        eWiseLog(logMat1);

        fftshift2d(fftImg2, img2.rows, img2.cols, shifted);
        cv::Mat logMat2 = eWiseAbs(shifted, img2.rows, img2.cols);
        eWiseLog(logMat2);

        int center_x = img1.cols / 2;
        int center_y = img1.rows / 2;
        int radius = min(center_x, center_y);
        int polar_width = radius;
        int polar_height = int(ceil((radius * CV_PI) / 2.0));

        warpPolar(logMat1, polarMat1, Size(polar_width, polar_height), Point2f(center_x, center_y),
                  radius,
                  WARP_POLAR_LOG + INTER_CUBIC + WARP_FILL_OUTLIERS);

        warpPolar(logMat2, polarMat2, Size(polar_width, polar_height), Point2f(center_x, center_y),
                  radius,
                  WARP_POLAR_LOG + INTER_CUBIC + WARP_FILL_OUTLIERS); 
        
        cv::Mat hpolarImg1 = Registration::getHalfMat(polarMat1, 180);
        cv::Mat hpolarImg2 = Registration::getHalfMat(polarMat2, 180);

        Point2d pt = cv::phaseCorrelate(hpolarImg1, hpolarImg2);

        float theta_shift = 360.0 / polar_height * -pt.y;

        deallocComplexMat(fftImg1, img1.rows);
        deallocComplexMat(fftImg2, img2.rows);
        deallocComplexMat(shifted, img1.rows);

        return theta_shift;
    }
};

#endif

// int main()
// {
//     string imgFn1 = "/media/media01/zyj/stiching/imgs/scale_8/DJI_0533_scale8.JPG";
//     string imgFn2 = "/media/media01/zyj/stiching/imgs/scale_8/DJI_0534_scale8.JPG";
//     // string imgFn3 = "/home/wlzhao/datasets/boden/img1.jpg";
//     // string imgFn4 = "/home/wlzhao/datasets/boden/img2.jpg";

//     cv::Mat img1 = cv::imread(imgFn1, 1);
//     cv::Mat img2 = cv::imread(imgFn2, 1);
//     cv::Mat greyImg1, greyImg2;
//     cv::Mat dst1, dst2;
//     float theta = 0;
//     cv::cvtColor(img1, greyImg1, cv::COLOR_BGRA2GRAY);
//     cv::cvtColor(img2, greyImg2, cv::COLOR_BGRA2GRAY);
//     greyImg1.convertTo(dst1, CV_32FC1);
//     greyImg2.convertTo(dst2, CV_32FC1);
  
//     theta  = Registration::estimateRotation(dst1, dst2);
//     cout << "Rotation is: " << theta << endl;

//     return 0;
// }