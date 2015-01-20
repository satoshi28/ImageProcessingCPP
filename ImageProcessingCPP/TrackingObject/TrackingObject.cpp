#include "TrackingObject.hpp"


TrackingObject::TrackingObject()
{
	extractor.create(trackingDetectorName, extractorName);
}

TrackingObject::~TrackingObject()
{
	extractor.~ExtractFeatures();
	cv::destroyAllWindows();
}




void TrackingObject::buildPatternFromImage(const cv::Mat image)
{
	//新規に作成
	Pattern pattern;
	trackingPattern = pattern;

	int numImages = 4;
    float step = sqrtf(2.0f);

    // Store original image in pattern structure
    trackingPattern.size = cv::Size(image.cols, image.rows);
    trackingPattern.frame = image.clone();
    
    // Build 2d and 3d contours (3d contour lie in XY plane since it's planar)
	trackingPattern.points2d.resize(4);
    trackingPattern.points3d.resize(4);

    // Image dimensions
    const float w = image.cols;
    const float h = image.rows;

    // Normalized dimensions:
    const float maxSize = std::max(w,h);
    const float unitW = w / maxSize;
    const float unitH = h / maxSize;

    trackingPattern.points2d[0] = cv::Point2f(0,0);
    trackingPattern.points2d[1] = cv::Point2f(w,0);
    trackingPattern.points2d[2] = cv::Point2f(w,h);
    trackingPattern.points2d[3] = cv::Point2f(0,h);

    trackingPattern.points3d[0] = cv::Point3f(-unitW, -unitH, 0);
    trackingPattern.points3d[1] = cv::Point3f( unitW, -unitH, 0);
    trackingPattern.points3d[2] = cv::Point3f( unitW,  unitH, 0);
    trackingPattern.points3d[3] = cv::Point3f(-unitW,  unitH, 0);

	//処理用の一時ファイル
	extractor.getFeatures(trackingPattern.frame, trackingPattern.features);

	cv::imshow("buildPattern", trackingPattern.frame);


}

void TrackingObject::buildPatternFromFeatures(const Features trackFeatures)
{
	//新規に作成
	Pattern pattern;
	trackingPattern = pattern;

    float step = sqrtf(2.0f);

	//点を囲む領域を取得
    std::vector<cv::Point2f> points(trackFeatures.keypoints.size());
	for (size_t i = 0; i < trackFeatures.keypoints.size(); i++)
    {
        points[i] = trackFeatures.keypoints[i].pt;
    }
	cv::Mat pointsMat = cv::Mat(points);
	cv::Rect rect = cv::boundingRect(pointsMat);

    // Store original image in pattern structure
    trackingPattern.size = cv::Size(rect.width, rect.height);
    
    // Build 2d and 3d contours (3d contour lie in XY plane since it's planar)
	trackingPattern.points2d.resize(4);
    trackingPattern.points3d.resize(4);

    // Image dimensions
    const float w = rect.width;
    const float h = rect.height;

    // Normalized dimensions:
    const float maxSize = std::max(w,h);
    const float unitW = w / maxSize;
    const float unitH = h / maxSize;

    trackingPattern.points2d[0] = cv::Point2f(0,0);
    trackingPattern.points2d[1] = cv::Point2f(w,0);
    trackingPattern.points2d[2] = cv::Point2f(w,h);
    trackingPattern.points2d[3] = cv::Point2f(0,h);

    trackingPattern.points3d[0] = cv::Point3f(-unitW, -unitH, 0);
    trackingPattern.points3d[1] = cv::Point3f( unitW, -unitH, 0);
    trackingPattern.points3d[2] = cv::Point3f( unitW,  unitH, 0);
    trackingPattern.points3d[3] = cv::Point3f(-unitW,  unitH, 0);

	//要素のコピー
	trackingPattern.features.descriptors = trackFeatures.descriptors.clone();
	std::copy(trackFeatures.keypoints.begin(),trackFeatures.keypoints.end(), back_inserter(trackingPattern.features.keypoints) );
}


void TrackingObject::getTransformMatrix(cv::Mat queryImage, double* transformMatrix)
{
	int64 start = cv::getTickCount();

	CameraCalibration calibration(626.318420f, 629.552124f, 322.617615f, 298.039093f);
	Matching matching;


	PatternTrackingInfo info;
	std::vector<cv::DMatch> matches;
	cv::Mat                   m_roughHomography;


	//処理用の一時ファイル
	Pattern queryPattern;

	extractor.getFeatures(queryImage, queryPattern.features);
	matching.getMatches(queryPattern.features, trackingPattern.features, matches);

	bool homographyFound = refineMatchesWithHomography(
		queryPattern.features.keypoints,
		trackingPattern.features.keypoints,
		3,
		matches,
		m_roughHomography);

	if (homographyFound)
    {
		
        // If homography refinement enabled improve found transformation
        if (enableHomographyRefinement)
        {
			// Get refined matches:
			cv::Mat warpedImg;
			Features warpedFeatures;
            std::vector<cv::DMatch> refinedMatches;
			cv::Mat                   m_refinedHomography;

			// Warp image using found homography
            cv::warpPerspective(queryImage, warpedImg, m_roughHomography, trackingPattern.size, cv::WARP_INVERSE_MAP | cv::INTER_CUBIC);

            // Detect features on warped image
            extractor.getFeatures(warpedImg, warpedFeatures);

            // Match with pattern
			matching.getMatches(warpedFeatures, trackingPattern.features, refinedMatches);

            // Estimate new refinement homography
            homographyFound = refineMatchesWithHomography(
                warpedFeatures.keypoints, 
                trackingPattern.features.keypoints, 
                3, 
                refinedMatches, 
                m_refinedHomography);

            // Get a result homography as result of matrix product of refined and rough homographies:
            info.homography = m_roughHomography * m_refinedHomography;

            // Transform contour with rough homography

            // Transform contour with precise homography
            cv::perspectiveTransform(trackingPattern.points2d, info.points2d, info.homography);
        }
        else
        {
			info.homography = m_roughHomography;
		}

		int64 end = cv::getTickCount();
		float time = (end - start) * 1000 / cv::getTickFrequency();

		cv::putText(queryImage, std::to_string(time), cv::Point(50,50),cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,0,200), 2, CV_AA);
	

		// Transform contour with rough homography
		cv::perspectiveTransform(trackingPattern.points2d, info.points2d, m_roughHomography);
		info.computePose(trackingPattern, calibration);
		info.draw2dContour(queryImage, cv::Scalar(255,255,0));
		cv::Point2f point = cv::Point2f(0, 0);
		for (size_t i = 0; i < info.points2d.size(); i++)
		{
			point.x += info.points2d[i].x;
			point.y += info.points2d[i].y;
		}
		point *= 1 / 4.0f;


		for(int i = 0; i < matches.size(); i++)
		{
			cv::circle(queryImage, trackingPattern.features.keypoints[matches[i].trainIdx].pt , 1, cv::Scalar(0,0,255),2, CV_FILLED);
		}

		cv::RotatedRect brect = getBoundingRect(trackingPattern.features.keypoints, matches);
		cv::Point2f vertices[4];
		brect.points(vertices);
		for (int i = 0; i < 4; i++)
			line(queryImage, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0));
		//cv::rectangle(queryImage, brect.tl(), brect.br(), cv::Scalar(0,128,255), 2);

		cv::imshow("result", queryImage);
		
		//モデルビュー行列
		Matrix44 glMatrix = info.pose3d.getMat44().getTransposed();
		glMatrix.data[3] = point.x;
		glMatrix.data[7] = point.y;
		cv::circle(queryImage, point, 10, cv::Scalar(0, 255, 255), -1, CV_FILLED);
		//std::memcpy(transformMatrix, glMatrix.data, sizeof(glMatrix.data));
		for(int i = 0; i < 16; i++)
		{
			transformMatrix[i] = glMatrix.data[i];
		}

	}else
	{
		for(int i = 0; i < 16; i++)
		{
			transformMatrix[i] = 1;
		}
	}

}


bool TrackingObject::refineMatchesWithHomography
    (
    const std::vector<cv::KeyPoint>& queryKeypoints,
    const std::vector<cv::KeyPoint>& trainKeypoints, 
    float reprojectionThreshold,
    std::vector<cv::DMatch>& matches,
    cv::Mat& homography
    )
{
    const int minNumberMatchesAllowed = 20;

    if (matches.size() < minNumberMatchesAllowed)
        return false;

    // Prepare data for cv::findHomography
    std::vector<cv::Point2f> srcPoints(matches.size());
    std::vector<cv::Point2f> dstPoints(matches.size());

    for (size_t i = 0; i < matches.size(); i++)
    {
        srcPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
        dstPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
    }

    // Find homography matrix and get inliers mask
    std::vector<unsigned char> inliersMask(srcPoints.size());
    homography = cv::findHomography(srcPoints, 
                                    dstPoints, 
                                    CV_FM_RANSAC, 
                                    reprojectionThreshold, 
                                    inliersMask);

    std::vector<cv::DMatch> inliers;
    for (size_t i=0; i<inliersMask.size(); i++)
    {
        if (inliersMask[i])
            inliers.push_back(matches[i]);
    }

    matches.swap(inliers);
    return matches.size() > minNumberMatchesAllowed;
}

cv::RotatedRect TrackingObject::getBoundingRect(std::vector<cv::KeyPoint> keypoints, std::vector<cv::DMatch> matches)
{
	// Prepare data for cv::findHomography
    std::vector<cv::Point2f> points(matches.size());

	for (size_t i = 0; i < matches.size(); i++)
    {
        points[i] = keypoints[matches[i].trainIdx].pt;
    }

	cv::Mat pointsMat = cv::Mat(points);

	return cv::minAreaRect(pointsMat);
}