#include "PatternDetector.h"


PatternDetector::PatternDetector()
{
	extractor.create(detectorName, extractorName);
}

PatternDetector
::~PatternDetector
()
{
	m_matching.~Matching();
	extractor.~ExtractFeatures();
}

void PatternDetector::setFeatures(std::vector<Features>& trainFeatures)
{
	//訓練データを格納(このデータに対しマッチングされる)
	//m_matching.train(trainPatterns);
}

int PatternDetector::findPattern(cv::Mat queryImage, std::vector<Features> trainPatterns )
{
	Features queryFeatures;

	//処理用
	// 特徴量をPatternに保存
	bool isExtracted =  extractor.getFeatures(queryImage,queryFeatures);

	//特徴量が抽出できなかったら終了
	if(isExtracted == false)
		return -1;
	//
	m_matching.train(trainPatterns);

	// すべての画像同士をマッチングする
	return m_matching.getMatches(queryFeatures);
}

Features PatternDetector::getMatchingQueryFeatures()
{
	return m_matching.getQueryMatches();
}