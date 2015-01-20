// TestCPPLibrary.h
#ifdef TESTFUNCDLL_EXPORT
#define TESTFUNCDLL_API __declspec(dllexport) 
#else
#define TESTFUNCDLL_API __declspec(dllimport) 
#endif

#include <opencv2\opencv.hpp>


#include "PatternDetector\PatternDetector.h"
#include "TrackingObject\TrackingObject.hpp"



extern "C" {
	TESTFUNCDLL_API void* getCamera(int CameraIndex);
    TESTFUNCDLL_API void releaseCamera(void* camera);
    TESTFUNCDLL_API void getCameraTexture(void* camera, unsigned char* data, int width, int height);
	TESTFUNCDLL_API bool setDescriptors(void* array, int id);
	TESTFUNCDLL_API bool setKeypoints(void* array_px, void* array_py, int id);

	TESTFUNCDLL_API int findObject();
	TESTFUNCDLL_API void getTransfromMatrix(double* matrix);

	TESTFUNCDLL_API void setTrackingPattern();

	TESTFUNCDLL_API void getProcessingTime(double time);

	cv::Rect getBoundingRect(std::vector<cv::KeyPoint> keypoints);

	PatternDetector patternDetector;
	TrackingObject tracking;
	std::vector<Features> trainFeatures;
	cv::Mat cameraImage;
	int preID;
	double time;

}