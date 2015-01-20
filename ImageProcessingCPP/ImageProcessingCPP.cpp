// TestCPPLibrary.cpp : Defines the exported functions for the DLL application.
//

#include "ImageProcessingCPP.h"

extern "C" {
	   
	void* getCamera(int CameraIndex)
	{
		cv::VideoCapture camera(CameraIndex);
		if(camera.isOpened() == true)
		{
			camera.release();
			return static_cast<void*>(new cv::VideoCapture(CameraIndex));
		}
		camera.release();

		//指定のデバイスが存在しないときはデバイス0を返す
	    return static_cast<void*>(new cv::VideoCapture(0));

	}
	
	void releaseCamera(void* camera)
	{
	    auto vc = static_cast<cv::VideoCapture*>(camera);
	    delete vc;
		
		patternDetector.~PatternDetector();
		tracking.~TrackingObject();
		trainFeatures.clear();
		
	}
	
	void getCameraTexture(void* camera, unsigned char* data, int width, int height)
	{
	    auto vc = static_cast<cv::VideoCapture*>(camera);
	    
	    // カメラ画の取得
		cv::Mat image;
	    *vc >> image;
	    
	    // リサイズ
	    cv::Mat resized_img(height, width, image.type());
	    cv::resize(image, resized_img, resized_img.size(), cv::INTER_CUBIC);
	    
		cameraImage = resized_img;
	    
		cv::Mat flipedImg;
		cv::flip(resized_img, flipedImg, 0); // 水平軸で反転（垂直反転）

	    // RGB --> ARGB 変換
	    cv::Mat argb_img;
	    cv::cvtColor(flipedImg, argb_img, CV_RGB2BGRA);
	    std::vector<cv::Mat> bgra;
	    cv::split(argb_img, bgra);
	    std::swap(bgra[0], bgra[3]);
	    std::swap(bgra[1], bgra[2]);
	    std::memcpy(data, argb_img.data, argb_img.total() * argb_img.elemSize());
	}

	bool setDescriptors(void* array, int id)
	{
		try
		{
			//一次元配列のアドレスを固定、取得
			unsigned char* data = static_cast<unsigned char*>(array);
			cv::Mat descriptors = cv::Mat::zeros(descriptorRows, descriptorDim, CV_8U);
			int n = 0;

			for(int i= 0; i < descriptorRows; i++)
			{
				for(int j = 0; j < descriptorDim; j++)
				{
					descriptors.at<unsigned char>(i, j) = (unsigned char)data[n++] ;
				}
			}

			//Patternに保存
			Features features;
			features.descriptors = descriptors.clone();
			features.ID = id;
			trainFeatures.push_back(features);
			
			return true;
		}catch(char *str)
		{
			// data;
			return false;
		}

	}

	bool setKeypoints(void* array_px,void* array_py, int id)
	{
		try
		{
			//一次元配列のアドレスを固定、取得
			unsigned char* data_px = static_cast<unsigned char*>(array_px);
			unsigned char* data_py = static_cast<unsigned char*>(array_py);
			std::vector<cv::KeyPoint> keypoints;
			int n = 0;

			for (int i = 0; i < descriptorRows; i++)
			{
				cv::KeyPoint point;
				point.pt.x = (float)data_px[n];
				point.pt.y = (float)data_py[n];
				keypoints.push_back(point);

				n++;
			}

			//Patternに保存
			for (int i = 0; i < trainFeatures.size(); i++)
			{
				if (trainFeatures[i].ID == id)
				{
					trainFeatures[i].keypoints = keypoints;
					break;
				}
			}

			return true;
		}
		catch (char *str)
		{
			// data;
			return false;
		}

	}

	int findObject()
	{
		int ID = patternDetector.findPattern(cameraImage, trainFeatures);

		//トラッキングパターンの構築
		if(ID > 0 && preID != ID)
		{
			preID = ID;
			/*
			//画像の切り抜き
			Features features= patternDetector.getMatchingQueryFeatures();
			cv::Rect rect = getBoundingRect(features.keypoints);
			cv::Mat roi_img(cameraImage, rect);

			tracking.buildPatternFromImage(roi_img);
			*/
			for(int i = 0; i < trainFeatures.size(); i++)
			{
				if(trainFeatures[i].ID == ID)
				{
					tracking.buildPatternFromFeatures(trainFeatures[ID]);
					break;
				}
			}
			
		}
		return ID;
	}

	void getTransfromMatrix(double* matrix)
	{

		tracking.getTransformMatrix(cameraImage, matrix);
	}

	void setTrackingPattern()
	{
		/*
		cv::Mat trackImg = cv::imread("C:\\Users\\satoshi\\Documents\\Image\\masteringOpencv.jpg", 1);

		tracking.buildPatternFromImage(trackImg);
		*/
		tracking.buildPatternFromImage(cameraImage);
	}

	void getProcessingTime(double t_time)
	{
		t_time = 1.0;
	}

	cv::Rect getBoundingRect(std::vector<cv::KeyPoint> keypoints)
	{
	// Prepare data for cv::findHomography
    std::vector<cv::Point2f> points(keypoints.size());

	for (size_t i = 0; i < keypoints.size(); i++)
    {
        points[i] = keypoints[i].pt;
    }

	cv::Mat pointsMat = cv::Mat(points);

	return cv::boundingRect(pointsMat);
	}
}