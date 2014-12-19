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
    Matching(cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(new cv::flann::LshIndexParams(6,12,1), new cv::flann::SearchParams(50)));
	~Matching();

    /**
    * 訓練特徴の集合とクエリ特徴をマッチングし,マッチングIDリストを返す
    */
	int getMatches(const Features queryFeatures);
	
	/**
    * クエリ特徴と訓練特徴をマッチングし,マッチングしたペアを返す
    */
	void getMatches(const Features queryFeatures,const Features trainFeatures, std::vector<cv::DMatch>& matches);

	void train(const std::vector<Features> trainFeatures );

	/*
	* return queryMatchingFeatures
	*/
	Features getQueryMatches();

private:	
	//テンプレート画像からPatternを作成
	
	/* マッチング（あらかじめtrainを実行） */
	void match(cv::Mat queryDescriptors, cv::FlannBasedMatcher& m_matcher,std::vector<cv::DMatch>& matches);

private:
    
	//画像セットの数
	int dataSetSize;
	//特徴集合に対するmatcher
    cv::FlannBasedMatcher m_matcher;
	//訓練特徴を格納するFeatures構造体
	std::vector<Features> m_trainFeatures;
	//クエリ画像中のマッチングしたデータ
	Features queryMatchingFeatures;
};


#endif