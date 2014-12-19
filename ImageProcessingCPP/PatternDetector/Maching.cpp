#include "Matching.h"

Matching::Matching(cv::Ptr<cv::DescriptorMatcher> matcher)
	: m_matcher(matcher)
{
	std::cout << "open matching" << std::endl;
}

Matching::~Matching()
{
	std::cout << "close matching" << std::endl;
}


int Matching::getMatches(const Features queryFeatures)
{
	//マッチングしたペア
	std::vector<cv::DMatch> matches;
	// Get matches
	match( queryFeatures.descriptors,m_matcher, matches);

	std::vector< std::pair<int, int> > imageRankingList(dataSetSize);	//各画像のランキング(rank, index)

	for(int i = 0; i < dataSetSize; i++)
	{
		imageRankingList[i].first = 0;
		imageRankingList[i].second = i;
	}
	
	//評価
	int num;
	for(int i = 0; i < matches.size(); i++)
	{
		num = matches[i].imgIdx;
		imageRankingList[num].first += 1;
	}


	//画像のランキングに基づいて降順に並び替え
	std::sort(imageRankingList.begin(), imageRankingList.end(),std::greater<std::pair<int, int>>() );

	int id;
	if(imageRankingList[0].first > 3)
	{
		//データベース内でのIDを求める(tranFeaturesでのID->DBのID)
		id = m_trainFeatures[imageRankingList[0].second].ID ;
		
		//クエリのマッチングした特徴を保存
		for(int i = 0; i < matches.size(); i++)
		{
			if( matches[i].imgIdx == imageRankingList[0].second)
			{
				int queryID = matches[i].queryIdx;
				queryMatchingFeatures.descriptors.push_back(queryFeatures.descriptors.row(queryID));
				queryMatchingFeatures.keypoints.push_back(queryFeatures.keypoints[queryID]);
			}
		}

	}else
	{
		id = -1;
	}
	return id;
}



void Matching::train(const std::vector<Features> trainPatterns )
{
// API of cv::DescriptorMatcher is somewhat tricky
	// First we clear old train data:
	this->m_matcher->clear();

	m_trainFeatures = trainPatterns;
	this->dataSetSize = m_trainFeatures.size();
	std::vector<cv::Mat> descriptors( this->dataSetSize );

	for(int i = 0; i < trainPatterns.size(); i++)
	{
		// Then we add vector of descriptors (each descriptors matrix describe one image). 
		// This allows us to perform search across multiple images:

		descriptors[i]= trainPatterns[i].descriptors.clone();
	}

	m_matcher->add(descriptors);
	// After adding train data perform actual train:
	m_matcher->train();
	
}



void Matching::match(cv::Mat queryDescriptors, cv::Ptr<cv::DescriptorMatcher>& m_matcher,std::vector<cv::DMatch>& matches)
{
	const float minRatio = 0.8f;
	matches.clear();

	//最近傍点の探索

	//knnマッチング
	std::vector< std::vector<cv::DMatch>>  knnMatches;

	// queryとmatcherに保存されている特徴量をknn構造体を用いて最近傍点を検索する.
	m_matcher->knnMatch(queryDescriptors, knnMatches, 2);

	//ratio test
	for(int j = 0; j < knnMatches.size(); j++)
	{
		if(knnMatches[j].empty() == false )
		{
			const cv::DMatch& bestMatch = knnMatches[j][0];
			const cv::DMatch& betterMatch = knnMatches[j][1];

			float distanceRatio = bestMatch.distance / betterMatch.distance;

			//距離の比が1.5以下の特徴だけ保存
			if(distanceRatio < minRatio)
			{
				matches.push_back(bestMatch);
			}
		}
	}
	knnMatches.clear();
}

void Matching::getMatches(const Features queryFeatures,const Features trainFeatures, std::vector<cv::DMatch>& matches)
{
	cv::Ptr<cv::DescriptorMatcher>   matcher   = new cv::BFMatcher(cv::NORM_HAMMING, false);
	matcher->clear();

	std::vector<cv::Mat> descriptors(1);
    descriptors[0] = trainFeatures.descriptors.clone();
	matcher->add(descriptors);
	
	// After adding train data perform actual train:
	matcher->train();
	

	// Get matches
	match(queryFeatures.descriptors, matcher, matches);

}

Features Matching::getQueryMatches()
{
	return queryMatchingFeatures;
}