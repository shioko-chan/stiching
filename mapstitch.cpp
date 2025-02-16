#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include "superpoint.h"
#include "lightglue.h"
#include "imagematch.hpp"
#include "orthprojection.h"
#include <iostream>
#include <fstream>
#include <codecvt>

using namespace std;
using namespace cv;

bool divide_images = false;
Stitcher::Mode mode = Stitcher::SCANS;
vector<Mat> imgs;
string stichedImage = "./stiching.jpg";
string matchingImage = "./matching.jpg";
string superPointPath;//SuperPoint ONNX format model path
string lightGluePath;//LightGlue ONNX format model path
string dstMatchFn;
float matchthresh = 0.0f;
void printUsage(char** argv);
int parseCmdArgs(int argc, char** argv);

int main(int argc, char* argv[])
{
    // OrthProjection::test();
    // ImageMatch::test();
    // ImageRetify::test();
    // return 0;
    
    int retval = parseCmdArgs(argc, argv);
    if (retval) return EXIT_FAILURE;

    //std::string_convert<std::codecvt_utf8<char_t>> converter;
    std::string sp = superPointPath;
    std::string lh = lightGluePath;
    //![stitching]
    Mat pano;
    Ptr<Stitcher> stitcher = Stitcher::create(mode);
    
    Ptr<SuperPoint> superpointp = makePtr<SuperPoint>(sp);
    Ptr<LightGlue> lightglue = makePtr<LightGlue>(lh, mode, matchthresh);
    stitcher->setPanoConfidenceThresh(0.1f);
    stitcher->setFeaturesFinder(superpointp);//SpuerPoint feature extraction
    stitcher->setFeaturesMatcher(lightglue);//LightGlue feature matching
    Stitcher::Status status = stitcher->stitch(imgs, pano);
    if (status == Stitcher::OK)
    {
        cout << "OK!" << endl;
        imshow(stichedImage, pano);
        cv::imwrite(stichedImage,pano);
    }
    // vector<KeyPoint>  keyps1;
    // superpointp->detect(imgs[0], keyps1);
    // cout << "Keypoints: " << keyps1.size() << endl;
    // for(int i = 0; i < keyps1.size(); i++)
    // {
    //     cout << keyps1[i].pt.x << "\t" << keyps1[i].pt.y << endl;
    // }

    //Draw Matches
    std::vector<cv::detail::ImageFeatures> features = lightglue->features();
    std::vector<cv::detail::MatchesInfo> matches = lightglue->matchinfo();
    ofstream matchStrm(dstMatchFn, ios::out);
    if(!matchStrm.is_open())
    {
        cout << "File '" << dstMatchFn << "' cannot open for write!\n";
        exit(0);
    }
    for (int i=0;i< matches.size();i++)
    {
        Mat srcImg = imgs[matches[i].src_img_idx];
        Mat dstImg = imgs[matches[i].dst_img_idx]; 
     
        cv::detail::ImageFeatures srcFeature;
        cv::detail::ImageFeatures dstFeature;
        for (int j = 0; j < features.size(); j++)
        {
            if (features[j].img_idx == matches[i].src_img_idx)
                srcFeature = features[j];
            if (features[j].img_idx == matches[i].dst_img_idx)
                dstFeature = features[j];
        }
        float img1size[2] = {srcFeature.img_size.width/2.0f, srcFeature.img_size.height/2.0f};
        float img2size[2] = {dstFeature.img_size.width/2.0f, dstFeature.img_size.height/2.0f};

        for(int k = 0; k < matches[i].matches.size(); k++)
        {
            //cout <<  << "\t" <<matches[i].matches[k] << endl;
            int idxq = matches[i].matches[k].queryIdx;
            int idxt = matches[i].matches[k].trainIdx; 
            matchStrm << srcFeature.keypoints[idxq].pt.x << "\t" << srcFeature.keypoints[idxq].pt.y << "\t";
            matchStrm << srcFeature.keypoints[idxt].pt.x << "\t" << srcFeature.keypoints[idxt].pt.y << "\n";
        }
        cout << matches[i].matches.size() << endl;

        //-- Draw matches
        Mat img_matches;
        Mat SrcresizedImage;
        resize(srcImg, SrcresizedImage, srcFeature.img_size);
        Mat DstresizedImage;
        resize(dstImg, DstresizedImage, dstFeature.img_size);
        drawMatches(SrcresizedImage, srcFeature.keypoints, DstresizedImage, dstFeature.keypoints, matches[i].matches, img_matches);
        cv::imwrite(matchingImage, img_matches);
        // -- Show detected matches
        imshow(matchingImage, img_matches);
        cv::waitKey();
    }
    matchStrm.close();

    return EXIT_SUCCESS;
}

void printUsage(char** argv)
{
    cout <<
         "Images stitcher.\n\n" << "Usage :\n" << argv[0] <<" [Flags] img1 img2 [...imgN]\n\n"
         "Flags:\n"
         "  --d3\n"
         "      internally creates three chunks of each image to increase stitching success\n"
         "  --mode (panorama|scans)\n"
         "      Determines configuration of stitcher. The default is 'panorama',\n"
         "      mode suitable for creating photo panoramas. Option 'scans' is suitable\n"
         "      for stitching materials under affine transformation, such as scans.\n"
         "  --output <result_img>\n"
         "      The default is 'result.jpg'.\n\n"
         "  --match <matchfile>\n"
         "      File saves the matches'.\n\n"
         "  --sp <SuperPoint ONNX format model path>\n"
         "  --lg <LightGlue ONNX format model path>\n"
         "Example usage :\n" << argv[0] << " --d3 --mode scans img1.jpg img2.jpg\n";
}


int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage(argv);
        return EXIT_FAILURE;
    }

    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage(argv);
            return EXIT_FAILURE;
        }
        else if (string(argv[i]) == "--d3")
        {
            divide_images = true;
        }
        else if (string(argv[i]) == "--sp")//SuperPoint ONNX format model path
        {
            superPointPath = argv[i+1];
            i++;
        }
        else if (string(argv[i]) == "--lg")//LightGlue ONNX format model path
        {
            lightGluePath = argv[i+1];
            i++;
        }
        else if (string(argv[i]) == "--mthresh")//匹配阈值
        {
            matchthresh = std::stof(argv[i + 1]); // 将字符串转换为float;
            i++;
        }
        else if (string(argv[i]) == "--d3")
        {
            divide_images = true;
        }
        else if (string(argv[i]) == "--output")
        {
            stichedImage = argv[i + 1];
            i++;
        }else if  (string(argv[i]) == "--match")
        {
            dstMatchFn = argv[i+1];
            i++;
        }
        else if (string(argv[i]) == "--mode")
        {
            if (string(argv[i + 1]) == "panorama")
                mode = Stitcher::PANORAMA;
            else if (string(argv[i + 1]) == "scans")
                mode = Stitcher::SCANS;
            else
            {
                cout << "Bad --mode flag value\n";
                return EXIT_FAILURE;
            }
            i++;
        }
        else
        {
            Mat img = imread(samples::findFile(argv[i]));
            if (img.empty())
            {
                cout << "Can't read image '" << argv[i] << "'\n";
                return EXIT_FAILURE;
            }

            if (divide_images)
            {
                Rect rect(0, 0, img.cols / 2, img.rows);
                imgs.push_back(img(rect).clone());
                rect.x = img.cols / 3;
                imgs.push_back(img(rect).clone());
                rect.x = img.cols / 2;
                imgs.push_back(img(rect).clone());
            }
            else
                imgs.push_back(img);
        }
    }
    return EXIT_SUCCESS;
}
