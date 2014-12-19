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

	/* ���̌��o */
	int findPattern(cv::Mat queryImage, std::vector<Features> trainPatterns );

	/* �}�b�`���O�����N�G���f�[�^���擾 */
	Features getMatchingQueryFeatures();

private:
	Matching m_matching;
	ExtractFeatures extractor;
};


#endif