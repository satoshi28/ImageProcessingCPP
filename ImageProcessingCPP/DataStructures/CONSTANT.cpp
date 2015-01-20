#include "CONSTANT.h"

const char* detectorName = "SURF";		//特徴点検出アルゴリズム
const char* trackingDetectorName = "SURF";
const char* extractorName = "BRISK";	//特徴量抽出アルゴリズム
const char* matcherName = "BruteForce-Hamming";		//マッチングアルゴリズム
const int descriptorRows = 200;
const int descriptorDim = 64;
const bool enableHomographyRefinement = false;