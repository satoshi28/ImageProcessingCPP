#ifndef TRACKING_OBJECT
#define TRACKING_OBJECT

#include <opencv2/opencv.hpp>

#include "../PatternDetector\ExtractFeatures.h"
#include "../PatternDetector\Matching.h"
#include "../DataStructures/Pattern.hpp"

class TrackingObject
{
public:
	TrackingObject();
	~TrackingObject();

	void buildPatternFromImage(const cv::Mat image);

	void getTransformMatrix(cv::Mat queryImage, double* matrix);

private:

	static bool refineMatchesWithHomography(
		const std::vector<cv::KeyPoint>& queryKeypoints, 
		const std::vector<cv::KeyPoint>& trainKeypoints, 
		float reprojectionThreshold,
		std::vector<cv::DMatch>& matches, 
		cv::Mat& homography);

	cv::RotatedRect getBoundingRect(std::vector<cv::KeyPoint> keypoints, std::vector<cv::DMatch> matches);

private:
	ExtractFeatures extractor;
    Pattern trackingPattern;					//トラッキングする物体の情報
};

#endif