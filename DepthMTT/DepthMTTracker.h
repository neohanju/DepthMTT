/**************************************************************************
* Title        : CSCTracker
* Author       : Haanju Yoo
* Initial Date : 2014.03.01 (ver. 0.9)
* Version Num. : 1.0 (since 2016.09.16)
* Description  :
*	Single camera multiple target tracker
**************************************************************************/

#pragma once

#include <vector>
#include <list>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/video/tracking.hpp"
#include "types.hpp"

namespace hj
{

/////////////////////////////////////////////////////////////////////////
// INPUT DETECTION
/////////////////////////////////////////////////////////////////////////
class CDetectedObject
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	CDetectedObject();
	~CDetectedObject();

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	unsigned int id;
	CDetection   detection;
	double       depth;
	bool         bMatchedWithTracklet;
	bool         bCoveredByOtherDetection;

	/* back tracking related */
	std::vector<std::vector<cv::Point2f>> vecvecTrackedFeatures; // current -> past order
	std::vector<Rect> boxes; // current -> past order

	/* appearance related */
	//cv::Mat patchGray;
	//cv::Mat patchRGB;
};


class CTrajectory
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	CTrajectory();
	~CTrajectory();

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	unsigned int id;
	unsigned int timeStart;
	unsigned int timeEnd;
	unsigned int timeLastUpdate;
	unsigned int duration;
	std::deque<int> trackletIDs;
	std::deque<hj::Rect> boxes;
	std::deque<double> depths;
};
typedef std::deque<CTrajectory> TrajectoryVector;


/////////////////////////////////////////////////////////////////////////
// SINGLE TARGET TRACKER
/////////////////////////////////////////////////////////////////////////
class CTracklet
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	CTracklet();
	~CTracklet();

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	unsigned int id;
	unsigned int timeStart;
	unsigned int timeEnd;
	unsigned int timeLastUpdate;
	unsigned int duration;
	unsigned int numStatic;
	double confidence;
	std::deque<hj::Rect> boxes;	
	std::deque<double> depths;
	std::vector<cv::Point2f> featurePoints;
	std::vector<cv::Point2f> trackedPoints;	
	hj::Rect estimatedBox;
	CTrajectory *ptTrajectory;
};
typedef std::deque<CTracklet*> TrackletPtQueue;


/////////////////////////////////////////////////////////////////////////
// MULTI-TARGET TRACKER
/////////////////////////////////////////////////////////////////////////
class DepthMTTracker
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	DepthMTTracker();
	~DepthMTTracker();

	void Initialize(stParamTrack &_stParams);
	void Finalize(void);
	CTrackResult& Track(
		DetectionSet vecInputDetections,
		cv::Mat curFrame, 
		int frameIdx);

private:
	/* MAIN OPERATIONS */	
	void GenerateDetectedObjects(
		const cv::Mat frameImage,
		DetectionSet &vecDetections,
		std::vector<CDetectedObject> &vecDetectedObjects);
	void BackwardTracking(std::vector<CDetectedObject> &vecDetectedObjects);
	void ForwardTracking(TrackletPtQueue &queueTracklets);
	void MatchingAndUpdating(
		const std::vector<CDetectedObject> &vecDetectedObjects, 
		TrackletPtQueue &queueTracklets);
	void ManagingTrajectories(const TrackletPtQueue &_queueActiveTracklets);
	void ResultPackaging();

	/* TRACKING RELATED */
	bool FeatureExtraction(
		const hj::Rect inputBox, 
		const cv::Mat inputImage, 
		std::vector<cv::Point2f> &vecFeaturePoints);
	bool FeatureTracking(
		const hj::Rect inputBox, 
		const cv::Mat inputImage, 
		const cv::Mat targetImage, 
		std::vector<cv::Point2f> &vecInputFeatures, 
		std::vector<cv::Point2f> &vecOutputFeatures, 
		std::vector<int> &vecFeatureInlierIndex, 
		hj::Rect &trackingResult);
	std::vector<cv::Point2f> FindInlierFeatures(
		std::vector<cv::Point2f> *vecInputFeatures, 
		std::vector<cv::Point2f> *vecOutputFeatures, 
		std::vector<unsigned char> *vecPointStatus);
	Rect LocalSearchKLT(
		Rect preBox, 
		std::vector<cv::Point2f> &preFeatures, 
		std::vector<cv::Point2f> &curFeatures, 
		std::vector<int> &inlierFeatureIndex);	
	static double BoxMatchingCost(Rect &box1, Rect &box2);
	static double GetTrackingConfidence(Rect &box, std::vector<cv::Point2f> &vecTrackedFeatures);

	/* ETC */
	double GetEstimatedDepth(const cv::Mat frameImage, const Rect objectBox);
	void ResultWithTrajectories(CTracklet *curTrajectory, CObjectInfo &outObjectInfo);

	/* FOR DEBUGGING */
	void VisualizeResult();

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	bool         bInit_;
	stParamTrack stParam_;	
	unsigned int nCurrentFrameIdx_;

	/* calibration related */	
	unsigned int nInputWidth_;
	unsigned int nInputHeight_;	

	/* input related */
	std::vector<CDetectedObject> vecDetectedObjects_;
	CMatFIFOBuffer cImageBuffer_;
	cv::Size sizeBufferImage_;
	cv::Mat  matGrayImage_;
	cv::Mat  matResizedGrayImage_;	
	
	/* traker related */
	unsigned int         nNewTrackletID_;
	std::list<CTracklet> listCTracklet_;
	TrackletPtQueue      queueActiveTracklets_;
	TrackletPtQueue      queueNewTracklets_;

	unsigned int             nNewTrajectoryID_;
	std::list<CTrajectory>   listCTrajectories_;
	std::deque<CTrajectory*> queueActiveTrajectories_;	

	/* matching related */
	std::vector<float> arrTrackletToDetectionMatchingCost_;
	std::vector<float> arrInterTrackletMatchingCost_;

	/* feature tracking related */
	cv::Ptr<cv::AgastFeatureDetector> featureDetector_;
	cv::Mat matFeatureExtractionMask_;

	/* result related */
	CTrackResult trackingResult_;

	/* visualization related */
	bool        bVisualizeResult_;
	cv::Mat     matTrackingResult_;
	std::string strVisWindowName_;
	std::vector<cv::Scalar> vecColors_;

	// TEMPORAL
	bool bFirstDraw_;
};

}

//()()
//('')HAANJU.YOO
