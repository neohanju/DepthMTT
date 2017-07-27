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
#include <queue>
#include <list>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>  // for video recording

namespace hj
{

/////////////////////////////////////////////////////////////////////////
// DETECTION (FOR THE INPUT OF THE TRACKING ALGORITHM & EVALUATION MODULE)
/////////////////////////////////////////////////////////////////////////
class CDetection
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	CDetection() : id(0), box(0.0, 0.0, 0.0, 0.0), depth(0.0), score(0.0) {}
	CDetection(const CDetection &_det) { *this = _det; }
	~CDetection() {}
	double GetDepth();

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	int        id;
	cv::Rect2d box;
	double     depth;
	double     score;
	cv::Mat    patch;	
};
typedef std::vector<CDetection> DetectionSet;


/////////////////////////////////////////////////////////////////////////
// FOR PER FRAME TRACKING RESULT
/////////////////////////////////////////////////////////////////////////
class CObjectInfo
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	// constructors
	CObjectInfo() : id(0), box(0.0, 0.0, 0.0, 0.0), depth(0.0) {}
	~CObjectInfo() {}

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	unsigned int id;
	cv::Rect2d   box;
	double       depth;
};


/////////////////////////////////////////////////////////////////////////
// VARIATION OF DETECTION OBJECT FOR BACKWARD FEATURE POINT TRACKING
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

	/* backward tracking related */
	std::vector<std::vector<cv::Point2f>> vecvecTrackedFeatures; // current -> past order
	std::vector<cv::Rect2d> boxes; // current -> past order
};


/////////////////////////////////////////////////////////////////////////
// TRAJECTORY (FINAL TRACKING RESULT OF EACH TARGET)
/////////////////////////////////////////////////////////////////////////
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
	std::deque<cv::Rect2d> boxes;
	std::deque<double> depths;
};
typedef std::deque<CTrajectory> TrajectoryVector;


/////////////////////////////////////////////////////////////////////////
// TRACKLET
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
	std::deque<cv::Rect2d> boxes;	
	std::deque<double> depths;
	std::vector<cv::Point2f> featurePoints;
	std::vector<cv::Point2f> trackedPoints;	
	cv::Rect2d estimatedBox;
	CTrajectory *ptTrajectory;
};
typedef std::deque<CTracklet*> TrackletPtQueue;


/////////////////////////////////////////////////////////////////////////
// TRACKING RESULT (OF ENTIRE TARGETS)
/////////////////////////////////////////////////////////////////////////
class CTrackResult
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	// constructors
	CTrackResult() : frameIdx(0), timeStamp(0), procTime(0) {}
	~CTrackResult() {}

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	unsigned int frameIdx;
	unsigned int timeStamp;
	time_t procTime;
	std::vector<CObjectInfo> objectInfos;

	std::vector<cv::Rect> vecDetectionRects;
	std::vector<cv::Rect> vecTrackerRects;
	cv::Mat matMatchingCost;
};


/////////////////////////////////////////////////////////////////////////
// ALGORITHM PARAMETERS
/////////////////////////////////////////////////////////////////////////
struct stParamTrack
{
	//------------------------------------------------
	// METHODS
	//------------------------------------------------
	stParamTrack()
		: nImageWidth(0)
		, nImageHeight(0)
		, dImageRescale(1.0)
		, dImageRescaleRecover(1.0)
		, dDepthEstimateCenterRegionRatio(0.4)
		, dDepthForegroundWindowSize(30)
		, nMaxTrackletLength(5)
		, nMinNumFeatures(4)
		, nMaxNumFeatures(100)
		, nBackTrackingLength(4)
		, dFeatureTrackWindowSizeRatio(1.0)
		, dMaxBoxDistance(1.0)
		, dMinBoxOverlapRatio(0.3)
		, dMaxBoxCenterDiffRatio(0.5)
		, dMinOpticalFlowMajorityRatio(0.5)
		, dMaxTranslationDistance(30.0)
		, dMaxDepthDistance(30.0)
		, nMaxPendingTime(100)
		, bVisualize(false)
		, bVideoRecord(false)
		, strVideoRecordPath("")
	{};
	~stParamTrack() {};

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
	int nImageWidth;
	int nImageHeight;

	/* speed-up */
	double dImageRescale;          // Image rescaling factor for speed-up.
	double dImageRescaleRecover;   // For restore image scale. Computed automatically with 'dImageRescale' variable. So, do not manually set this value.

	/* depth estimation */
	double dDepthEstimateCenterRegionRatio;
	double dDepthForegroundWindowSize;

	/* bi-directional tracking */
	int    nMaxTrackletLength;     // To cut off tracklets that are too long and unreliable.
	int    nMinNumFeatures;        // The minimum number of tracked feature points that are required to maintain a tracklet.
	int    nMaxNumFeatures;        // To prevent tracking too many feature points.
	int    nBackTrackingLength;    // The interval of bi-directional tracking of feature points.
	double dFeatureTrackWindowSizeRatio;  // The optical flow searching window size w.r.t. the size of a detection box.
	double dMaxBoxDistance;        // For validation condition
	double dMinBoxOverlapRatio;    // For validation condition
	double dMaxBoxCenterDiffRatio; // For validation condition
	double dMinOpticalFlowMajorityRatio;  // To filter out an ambiguities in the ownership of each feature point.

	/* matching related */
	double dMaxTranslationDistance;
	double dMaxDepthDistance;
	int    nMaxPendingTime;

	/* visualization for debugging */
	bool bVisualize;

	/* video recording for result visualization */
	bool        bVideoRecord;
	std::string strVideoRecordPath;
};


////////////////////////////////////////////////////////////////////////
// IMAGE BUFFER
/////////////////////////////////////////////////////////////////////////
class CMatFIFOBuffer
{
	//------------------------------------------------
	// METHODS
	//------------------------------------------------
public:
	CMatFIFOBuffer() : bInit_(false) {}
	CMatFIFOBuffer(int _bufferSize) : bInit_(true), bufferSize_(_bufferSize) {}
	~CMatFIFOBuffer() { clear(); }
	bool    set(int _bufferSize);
	bool    clear();
	bool    insert(cv::Mat _newMat);
	bool    insert_resize(cv::Mat _newMat, cv::Size _resizeParam);
	bool    remove(int _pos);
	size_t  size() { return bufferSize_; }
	size_t  num_elements() { return buffer_.size(); }
	cv::Mat front()	{ assert(bInit_); return buffer_.front(); }
	cv::Mat back()	{ assert(bInit_); return buffer_.back(); }
	cv::Mat get(int _pos) { assert(bInit_ && _pos < bufferSize_); return buffer_[_pos]; }
	int     get_back_idx() { return (int)buffer_.size(); }

	/* iterators */
	typedef std::deque<cv::Mat>::iterator iterator;
	typedef std::deque<cv::Mat>::const_iterator const_iterator;
	typedef std::deque<cv::Mat>::reverse_iterator reverse_iterator;
	typedef std::deque<cv::Mat>::const_reverse_iterator const_reverse_iterator;
	iterator begin() { return buffer_.begin(); }
	iterator end() { return buffer_.end(); }
	const_iterator begin() const { return buffer_.begin(); }
	const_iterator end()   const { return buffer_.end(); }
	reverse_iterator rbegin() { return buffer_.rbegin(); }
	reverse_iterator rend() { return buffer_.rend(); }
	const_reverse_iterator rbegin() const { return buffer_.rbegin(); }
	const_reverse_iterator rend()   const { return buffer_.rend(); }


	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
private:
	bool bInit_;
	int  bufferSize_;
	std::deque<cv::Mat> buffer_;
};


/////////////////////////////////////////////////////////////////////////
// MULTI-TARGET TRACKER
/////////////////////////////////////////////////////////////////////////
class CDepthMTTracker
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	CDepthMTTracker();
	~CDepthMTTracker();

	void Initialize(stParamTrack &_stParams);
	void Finalize(void);
	CTrackResult Track(
		DetectionSet _vecInputDetections,
		cv::Mat _curFrame, 
		int _frameIdx);
	static DetectionSet ReadDetectionResultWithTxt(std::string _strFilePath);

private:
	/* MAIN OPERATIONS */	
	void GenerateDetectedObjects(
		const cv::Mat _frameImage,
		DetectionSet &_vecDetections,
		std::vector<CDetectedObject> &_vecDetectedObjects);
	void BackwardTracking(std::vector<CDetectedObject> &_vecDetectedObjects);
	void ForwardTracking(TrackletPtQueue &_queueTracklets);
	void DetectionToTrackletMatching(
		const std::vector<CDetectedObject> &_vecDetectedObjects, 
		TrackletPtQueue &_queueTracklets);
	void TrackletToTrajectoryMatching(const TrackletPtQueue &_queueActiveTracklets);
	void ResultPackaging();

	/* TRACKING RELATED */
	bool FeatureExtraction(
		const cv::Rect _inputBox, 
		const cv::Mat _inputImage, 
		std::vector<cv::Point2f> &_vecFeaturePoints);
	bool FeatureTracking(
		const cv::Rect _inputBox, 
		const cv::Mat _inputImage, 
		const cv::Mat _targetImage, 
		std::vector<cv::Point2f> &_vecInputFeatures, 
		std::vector<cv::Point2f> &_vecOutputFeatures, 
		std::vector<int> &_vecFeatureInlierIndex, 
		cv::Rect &_trackingResult);
	std::vector<cv::Point2f> FindInlierFeatures(
		std::vector<cv::Point2f> *_vecInputFeatures, 
		std::vector<cv::Point2f> *_vecOutputFeatures, 
		std::vector<unsigned char> *_vecPointStatus);
	cv::Rect LocalSearchKLT(
		cv::Rect _preBox, 
		std::vector<cv::Point2f> &_preFeatures, 
		std::vector<cv::Point2f> &_curFeatures, 
		std::vector<int> &_inlierFeatureIndex);
	static double BoxCenterDistanceWRTScale(cv::Rect &_box1, cv::Rect &_box2);
	static double GetTrackingConfidence(cv::Rect &_box, std::vector<cv::Point2f> &_vecTrackedFeatures);

	/* ETC */
	double GetEstimatedDepth(const cv::Mat _frameImage, const cv::Rect _objectBox);
	CObjectInfo GetObjectInfo(CTrajectory *_curTrajectory);

	/* VISUALIZATION */
	void VisualizeResult();
	cv::Scalar hsv2rgb(double h, double s, double v);
	std::vector<cv::Scalar> GenerateColors(unsigned int numColor);
	cv::Scalar getColorByID(unsigned int nID, std::vector<cv::Scalar> *vecColors);
	void DrawBoxWithID(cv::Mat &imageFrame, cv::Rect box, unsigned int nID, int lineStyle, int fontSize, cv::Scalar curColor);

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
	hj::CMatFIFOBuffer cImageBuffer_;
	cv::Size sizeBufferImage_;
	cv::Mat  matGrayImage_;
	cv::Mat  matResizedGrayImage_;	
	
	/* tracklet related */
	unsigned int         nNewTrackletID_;
	std::list<CTracklet> listCTracklet_;
	TrackletPtQueue      queueActiveTracklets_;
	TrackletPtQueue      queueNewTracklets_;

	/* trajectory related */
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

	// record
	bool bRecord_;
	bool bVideoWriterInit_;
	std::string strRecordPath_;
	CvVideoWriter *videoWriter_;
};

}

//()()
//('')HAANJU.YOO
