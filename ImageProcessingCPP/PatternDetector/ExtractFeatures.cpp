#include "ExtractFeatures.h"

ExtractFeatures::ExtractFeatures()
{
}

ExtractFeatures::~ExtractFeatures()
{
	m_detector.release();				//特徴点検出器
	m_extractor.release();
	std::cout << "close extarctor" << std::endl;
}

void ExtractFeatures::create(const char* _detectorName,const char* _extractorName)
{
	if(detectorName == "SURF")
		this->m_detector = new cv::SURF(400);
	else
		this->m_detector = cv::FeatureDetector::create(_detectorName);

        //cv::SurfFeatureDetector detector = cv::SurfFeatureDetector(400), 
    this->m_extractor = cv::DescriptorExtractor::create(_extractorName);
}

bool ExtractFeatures::getFeatures(cv::Mat& image,
									  Features& features)
{
	bool extractFlag = false;

	//グレイスケール化
	cv::Mat grayImg;										
	getGray(image, grayImg);

	//特徴量の抽出
	extractFlag = extractFeatures(grayImg, features.keypoints, features.descriptors);

	grayImg.release();

	return extractFlag;
}

void ExtractFeatures::getGray(const cv::Mat& image, cv::Mat& gray)
{
    if (image.channels()  == 3)
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, CV_BGRA2GRAY);
    else if (image.channels() == 1)
        gray = image;
}

bool ExtractFeatures::extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    assert(!image.empty());
    assert(image.channels() == 1);


	if(m_extractor.empty() == true)
		return false;

    m_detector->detect(image, keypoints);
    if (keypoints.empty())
        return false;

    m_extractor->compute(image, keypoints, descriptors);
    if (keypoints.empty())
        return false;

    return true;
}