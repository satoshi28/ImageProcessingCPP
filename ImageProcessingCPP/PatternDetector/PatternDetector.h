#ifndef PATTERN_DETECTOR
#define PATTERN_DETECTOR

#include "ExtractFeatures.h"
#include "Matching.h"

class PatternDetector
{
public:
	PatternDetector();
	~PatternDetector();

	void setFeatures(std::vector<Features>& trainFeatures);

	/* 物体検出 */
	int findPattern(cv::Mat queryImage, std::vector<Features> trainPatterns );

	/* マッチングしたクエリデータを取得 */
	Features getMatchingQueryFeatures();

private:
	Matching m_matching;
	ExtractFeatures extractor;
};


#endif