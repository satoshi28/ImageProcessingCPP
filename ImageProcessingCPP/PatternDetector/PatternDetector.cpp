#include "PatternDetector.h"


PatternDetector::PatternDetector()
{
	extractor.create(detectorName, extractorName);
}

PatternDetector
::~PatternDetector
()
{
	
}

void PatternDetector::setFeatures(std::vector<Features>& trainFeatures)
{
	//�P���f�[�^���i�[(���̃f�[�^�ɑ΂��}�b�`���O�����)
	//m_matching.train(trainPatterns);
}

int PatternDetector::findPattern(cv::Mat queryImage, std::vector<Features> trainPatterns )
{
	Features queryFeatures;

	//�����p
	// �����ʂ�Pattern�ɕۑ�
	extractor.getFeatures(queryImage,queryFeatures);

	//
	m_matching.train(trainPatterns);

	// ���ׂẲ摜���m���}�b�`���O����
	return m_matching.getMatches(queryFeatures);
}

Features PatternDetector::getMatchingQueryFeatures()
{
	return m_matching.getQueryMatches();
}