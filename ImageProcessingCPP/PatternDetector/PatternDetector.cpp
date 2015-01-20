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
	//�P���f�[�^���i�[(���̃f�[�^�ɑ΂��}�b�`���O�����)
	//m_matching.train(trainPatterns);
}

int PatternDetector::findPattern(cv::Mat queryImage, std::vector<Features> trainPatterns )
{
	Features queryFeatures;

	//�����p
	// �����ʂ�Pattern�ɕۑ�
	bool isExtracted =  extractor.getFeatures(queryImage,queryFeatures);

	//�����ʂ����o�ł��Ȃ�������I��
	if(isExtracted == false)
		return -1;
	//
	m_matching.train(trainPatterns);

	// ���ׂẲ摜���m���}�b�`���O����
	return m_matching.getMatches(queryFeatures);
}

Features PatternDetector::getMatchingQueryFeatures()
{
	return m_matching.getQueryMatches();
}