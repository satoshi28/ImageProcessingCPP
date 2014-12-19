#ifndef MATCHING_
#define MATCHING_

////////////////////////////////////////////////////////////////////
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <functional>

#include "../DataStructures/Pattern.hpp"
#include "../DataStructures/CONSTANT.h"

/**
 * Store the image data and computed descriptors of target pattern
 */
class Matching
{
public:
	/**
     *
     */
    Matching(cv::Ptr<cv::DescriptorMatcher>   matcher   = cv::DescriptorMatcher::create(matcherName));
	~Matching();

    /**
    * �P�������̏W���ƃN�G���������}�b�`���O��,�}�b�`���OID���X�g��Ԃ�
    */
	int getMatches(const Features queryFeatures);
	
	/**
    * �N�G�������ƌP���������}�b�`���O��,�}�b�`���O�����y�A��Ԃ�
    */
	void getMatches(const Features queryFeatures,const Features trainFeatures, std::vector<cv::DMatch>& matches);

	void train(const std::vector<Features> trainFeatures );

	/*
	* return queryMatchingFeatures
	*/
	Features getQueryMatches();

private:	
	//�e���v���[�g�摜����Pattern���쐬
	
	/* �}�b�`���O�i���炩����train�����s�j */
	void match(cv::Mat queryDescriptors,cv::Ptr<cv::DescriptorMatcher>& matcher,std::vector<cv::DMatch>& matches);

private:
    
	//�摜�Z�b�g�̐�
	int dataSetSize;
	//�����W���ɑ΂���matcher
    cv::Ptr<cv::DescriptorMatcher> m_matcher;
	//�P���������i�[����Features�\����
	std::vector<Features> m_trainFeatures;
	//�N�G���摜���̃}�b�`���O�����f�[�^
	Features queryMatchingFeatures;
};


#endif