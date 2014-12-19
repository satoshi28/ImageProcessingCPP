#ifndef EXTRACT_FEATURES
#define EXTRACT_FEATURES

////////////////////////////////////////////////////////////////////

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include "../DataStructures/Pattern.hpp"
#include "../DataStructures/CONSTANT.h"

/**
 * Store the kepoints and descriptors of image
 */
class ExtractFeatures
{
public:
	/**
     * Initialize a pattern detector with specified feature detector, descriptor extraction and matching algorithm
     */
    ExtractFeatures();

		//detector,extractor�̍쐬
	void create(const char* _detectorName,const char* _extractorName);

	~ExtractFeatures();


	/**
    * �摜���󂯎��,�����ʂ𒊏o���� 
    * ���o���������_,�����ʂ�Pattern�\���̂Ƃ��ĕۑ�����
    */
	bool getFeatures(cv::Mat& image, Features& features);

private:
	/**
    * ���͂��ꂽ�摜����O���C�X�P�[���摜���擾����
	* Supported input images types - 1 channel (no conversion is done), 3 channels (assuming BGR) and 4 channels (assuming BGRA).
    */
	void getGray(const cv::Mat& image, cv::Mat& grayImg);

	/**
    * ���͂��ꂽ�摜��������_,�����ʂ𒊏o����
    */
	bool extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

private:
	cv::Ptr<cv::FeatureDetector> m_detector;				//�����_���o��
    cv::Ptr<cv::DescriptorExtractor> m_extractor;

};




#endif