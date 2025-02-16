#pragma once
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


class  LightGlue :public FeaturesMatcher
{
protected:
	Stitcher::Mode m_mode;//Affine or Perspective
	std::string m_modelPath;

	std::vector<detail::ImageFeatures> features_;
	std::vector<detail::MatchesInfo> pairwise_matches_;
	float m_matchThresh = 0.0;
	Ort::Session *lightglueSession;

	CV_WRAP_AS(apply) void operator ()(const ImageFeatures& features1, const ImageFeatures& features2,
		CV_OUT MatchesInfo& matches_info) {
		match(features1, features2, matches_info);
	}
	void addFeature(detail::ImageFeatures features);
	void addMatcheinfo(const MatchesInfo& matches_info);
public:

	LightGlue(std::string modelPath, Stitcher::Mode mode, float matchThresh);
	void match(const ImageFeatures& features1, const ImageFeatures& features2,
		MatchesInfo& matches_info);
	void doMatch(const ImageFeatures& features1, const ImageFeatures& features2,
	MatchesInfo& matches_info);
	std::vector<detail::ImageFeatures> features() { return features_; };
	std::vector<detail::MatchesInfo> matchinfo() { return pairwise_matches_; };

};
