#ifndef SUPERPOINT_H
#define SUPERPOINT_H

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/stitching.hpp>
#include <onnxruntime_cxx_api.h>

using namespace cv::detail;
using namespace cv;
using namespace std;

class SuperPoint: public Feature2D
{
protected:
	OrtSession *extractorSession;
	std::string m_modelPath;
	OrtApi ortapi;
	vector<float> ApplyTransform(const Mat& image, float& mean, float& std);
public:
	SuperPoint(std::string modelPath);
	void detectAndCompute(InputArray image, 
		std::vector<KeyPoint>& keypoints,
		OutputArray descriptors,
		bool useProvidedKeypoints = false);
	void detect(InputArray image,
		std::vector<KeyPoint>& keypoints,
		InputArray mask = noArray());
	void CheckStatus(OrtStatus* status);
	void compute(InputArray image,
		std::vector<KeyPoint>& keypoints,
		OutputArray descriptors);

};

#endif