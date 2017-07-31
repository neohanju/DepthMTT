#include <limits>
#include <assert.h>
#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "HungarianMethod.h"
#include "DepthMTTracker.h"
#include "haanju_utils.hpp"


namespace hj
{

/////////////////////////////////////////////////////////////////////////
// CDetection MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
double CDetection::GetDepth()
{
	assert(!patch.empty());
	int numPixels = 0;
	double totalDepths = 0.0;
	double windowSize = patch.cols / 4;

	for (int col = std::max(0, int(patch.cols / 2 - windowSize));
		col < std::min(int(patch.cols / 2 + windowSize), patch.cols);
		col++)
	{
		for (int row = std::max(0, int(patch.rows / 2 - windowSize));
			row < std::min(int(patch.rows / 2 + windowSize), patch.rows);
			row++)
		{
			numPixels++;
			totalDepths += (double)patch.at<unsigned char>(row, col);
		}
	}
	depth = totalDepths / (double)numPixels;

	return depth;
}


/////////////////////////////////////////////////////////////////////////
// CDetectedObject MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
CDetectedObject::CDetectedObject()
	: id(0)	
	, depth(0.0)
	, bMatchedWithTracklet(false)
	, bCoveredByOtherDetection(false)
{
}


CDetectedObject::~CDetectedObject()
{
	for (size_t vecIdx = 0; vecIdx < vecvecTrackedFeatures.size(); vecIdx++)
	{
		vecvecTrackedFeatures[vecIdx].clear();
	}
	vecvecTrackedFeatures.clear();
	boxes.clear();
}


/////////////////////////////////////////////////////////////////////////
// CTracklet MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
CTracklet::CTracklet()
	: id(0)
	, timeStart(0)
	, timeEnd(0)
	, timeLastUpdate(0)
	, duration(0)
	, numStatic(0)
	, confidence(0.0)	
	, estimatedBox(0.0, 0.0, 0.0, 0.0)
	, ptTrajectory(NULL)
{
}


CTracklet::~CTracklet()
{
	boxes.clear();
	depths.clear();
	featurePoints.clear();
	trackedPoints.clear();
}


/////////////////////////////////////////////////////////////////////////
// CTrajectory MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
CTrajectory::CTrajectory()
	: id(0)
	, timeStart(0)
	, timeEnd(0)
	, timeLastUpdate(0)
	, duration(0)	
{
}


CTrajectory::~CTrajectory()
{
	trackletIDs.clear();
	boxes.clear();
	depths.clear();
}


/////////////////////////////////////////////////////////////////////////
// CMatFIFOBuffer MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
bool CMatFIFOBuffer::set(int _bufferSize)
{
	if (bInit_) { this->clear(); }
	try
	{
		bufferSize_ = _bufferSize;
	}
	catch (int e)
	{
		printf("An execption occured in CircularBuffer::set. Exeption number is %d.\n", e);
		return false;
	}
	return bInit_ = true;
}


bool CMatFIFOBuffer::clear()
{
	if (!bInit_) { return true; }
	try
	{
		for (int bufferIdx = 0; bufferIdx < buffer_.size(); bufferIdx++)
		{
			this->remove(bufferIdx);
		}
	}
	catch (int e)
	{
		printf("An execption occured in CircularBuffer::clear. Exeption number is %d.\n", e);
		return false;
	}
	bufferSize_ = 0;
	bInit_ = false;

	return true;
}


bool CMatFIFOBuffer::insert(cv::Mat _newMat)
{
	try
	{
		cv::Mat newBufferMat = _newMat.clone();
		buffer_.push_back(newBufferMat);

		// circulation
		if (bufferSize_ < buffer_.size())
		{
			buffer_.pop_front();
		}
	}
	catch (int e)
	{
		printf("An execption occured in CircularBuffer::insert. Exeption number is %d.\n", e);
		return false;
	}

	return true;
}


bool CMatFIFOBuffer::insert_resize(cv::Mat _newMat, cv::Size _resizeParam)
{
	try
	{
		cv::Mat newBufferMat;
		cv::resize(_newMat, newBufferMat, _resizeParam);
		buffer_.push_back(newBufferMat);

		// circulation
		if (bufferSize_ < buffer_.size())
		{
			buffer_.pop_front();
		}
	}
	catch (int e)
	{
		printf("An execption occured in CircularBuffer::insert. Exeption number is %d.\n", e);
		return false;
	}

	return true;
}


bool CMatFIFOBuffer::remove(int _pos)
{
	assert(bInit_ && _pos < bufferSize_);
	if (_pos >= buffer_.size())
	{
		return true;
	}
	if (!buffer_[_pos].empty())
	{
		buffer_[_pos].release();
	}
	return true;
}


/////////////////////////////////////////////////////////////////////////
// DepthMTTracker MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
CDepthMTTracker::CDepthMTTracker()
	: bInit_(false)
	, bVisualizeResult_(false)
	, strVisWindowName_("")
{
}


CDepthMTTracker::~CDepthMTTracker()
{
	Finalize();
}


/************************************************************************
 Method Name: Initialize
 Description:
	- Initialize the multiple target tracker with the given parameters.
 Input Arguments:
	- _stParams: structure of parameters
 Return Values:
	- None
************************************************************************/
void CDepthMTTracker::Initialize(stParamTrack &_stParams)
{
	// check duplicated initialization
	if (bInit_) { Finalize(); }

	stParam_ = _stParams;
	stParam_.dImageRescaleRecover = 1.0 / stParam_.dImageRescale;

	nCurrentFrameIdx_ = 0;	
	
	trackingResult_.frameIdx = nCurrentFrameIdx_;
	trackingResult_.objectInfos.clear();

	nInputWidth_  = _stParams.nImageWidth;
	nInputHeight_ = _stParams.nImageHeight;

	// detection related
	vecDetectedObjects_.clear();

	// tracker related
	nNewTrackletID_ = 0;
	listCTracklet_.clear();
	queueActiveTracklets_.clear();

	// trajectory related
	nNewTrajectoryID_ = 0;

	// input related
	sizeBufferImage_ = cv::Size(
		(int)((double)nInputWidth_  * stParam_.dImageRescale),
		(int)((double)nInputHeight_ * stParam_.dImageRescale));
	matGrayImage_ = cv::Mat(nInputHeight_, nInputWidth_, CV_8UC1);
	cImageBuffer_.set(stParam_.nBackTrackingLength);	

	// feature tracking related		
	featureDetector_          = cv::AgastFeatureDetector::create();
	matFeatureExtractionMask_ = cv::Mat(sizeBufferImage_, CV_8UC1, cv::Scalar(0));

	// visualization related
	bVisualizeResult_ = stParam_.bVisualize;
	strVisWindowName_ = "Tracking result";
	if (bVisualizeResult_)
	{
		vecColors_ = GenerateColors(400);
	}

	// record
	bRecord_ = stParam_.bVideoRecord;
	bVideoWriterInit_ = false;
	if (bRecord_) {
		strRecordPath_ = stParam_.strVideoRecordPath;

		// get time
		time_t curTimer = time(NULL);
		struct tm timeStruct;
		localtime_s(&timeStruct, &curTimer);

		// make file name
		char resultOutputFileName[256];
		sprintf_s(resultOutputFileName, "%s_result_%02d%02d%02d_%02d%02d%02d.avi",
			strRecordPath_.c_str(),
			timeStruct.tm_year + 1900,
			timeStruct.tm_mon + 1,
			timeStruct.tm_mday,
			timeStruct.tm_hour,
			timeStruct.tm_min,
			timeStruct.tm_sec);

		// init video writer
		CvSize imgSize;
		imgSize.width = stParam_.nImageWidth;
		imgSize.height = stParam_.nImageHeight;
		videoWriter_ = cvCreateVideoWriter(resultOutputFileName, CV_FOURCC('M', 'J', 'P', 'G'), 15, imgSize, 1);
		bVideoWriterInit_ = true;
	}

	// initialization flag
	bInit_ = true;
}


/************************************************************************
 Method Name: Finalize
 Description:
	- Terminate the class with memory clean up.
 Input Arguments:
	- None
 Return Values:
	- None
************************************************************************/
void CDepthMTTracker::Finalize(void)
{	
	if (!bInit_) { return; }

	/* detection related */
	this->vecDetectedObjects_.clear();

	/* tracker related */	
	listCTracklet_.clear();
	queueActiveTracklets_.clear();

	/* input related */
	cImageBuffer_.clear();
	if (!matGrayImage_.empty()) { matGrayImage_.release(); }
	if (!matTrackingResult_.empty()) { matTrackingResult_.release(); }

	/* matching related */
	arrTrackletToDetectionMatchingCost_.clear();

	/* result related */
	trackingResult_.objectInfos.clear();

	/* visualize related */
	if (bVisualizeResult_) { cv::destroyWindow(strVisWindowName_); }

	// record
	if (bRecord_)
	{
		cvReleaseVideoWriter(&this->videoWriter_);
	}

	/* initialization flag */
	bInit_ = false;
}


/************************************************************************
 Method Name: Track
 Description:
	- Run the tracking algorithm on the current input frame and detections.
 Input Arguments:
	- _vecInputDetections: Input detections of the current frame
	- _curFrame: Current input frame image
	- _frameIdx: Current frame index
 Return Values:
	- CTrackResult: Tracking result of the current frame
************************************************************************/
CTrackResult CDepthMTTracker::Track(
	DetectionSet _vecInputDetections, 
	cv::Mat _curFrame, 
	int _frameIdx)
{
	time_t timeStartTrack = clock();

	assert(bInit_ && _curFrame.rows == matGrayImage_.rows
		&& _curFrame.cols == matGrayImage_.cols);	
	nCurrentFrameIdx_ = _frameIdx;

	/* input frame buffering (for bi-directional tracking) */
	matGrayImage_ = _curFrame;
	cImageBuffer_.insert_resize(matGrayImage_, sizeBufferImage_);
	if (!matTrackingResult_.empty()) { matTrackingResult_.release(); }
	cv::cvtColor(_curFrame, matTrackingResult_, cv::COLOR_GRAY2BGR);

	/* pre-processing of input detections */
	GenerateDetectedObjects(_curFrame, _vecInputDetections, vecDetectedObjects_);

	/* tracklet management (detection-to-tracklet matching) */
	BackwardTracking(vecDetectedObjects_);
	ForwardTracking(queueActiveTracklets_);	
	DetectionToTrackletMatching(vecDetectedObjects_, queueActiveTracklets_);

	/* trajectory management (tracklet-to-trajectory matching) */
	TrackletToTrajectoryMatching(queueActiveTracklets_);

	/* result packaging */
	ResultPackaging();	
	trackingResult_.procTime = clock() - timeStartTrack;

	/* visualize */
	if (bVisualizeResult_) { VisualizeResult(); }

	return this->trackingResult_;
}


/************************************************************************
 Method Name: ReadDetectionResultWithTxt
 Description:
	- Read detections from pre-generated txt files.
 Input Arguments:
	- _strFilePath : Path to an input file
 Return Values:
	- DetectionSet: Read detections.
************************************************************************/
DetectionSet CDepthMTTracker::ReadDetectionResultWithTxt(std::string _strFilePath)
{
	DetectionSet vec_result;
	int num_detection = 0;
	float x, y, w, h, depth, id;

	FILE *fid;
	try {
		fopen_s(&fid, _strFilePath.c_str(), "r");
		if (NULL == fid) { return vec_result; }

		// read # of detections
		fscanf_s(fid, "%d\n", &num_detection);
		vec_result.reserve(num_detection);

		// read box infos
		for (int detect_idx = 0; detect_idx < num_detection; detect_idx++)
		{
			fscanf_s(fid, "%f %f %f %f %f %f\n", &id, &depth, &x, &y, &w, &h);
			CDetection cur_detection;
			cur_detection.id = (int)id;
			cur_detection.box = cv::Rect2d((double)x, (double)y, (double)w, (double)h);
			vec_result.push_back(cur_detection);
		}
		fclose(fid);
	}
	catch (int dwError) {
		printf("[ERROR] file open error with detection result reading: %d\n", dwError);
	}
	return vec_result;
}


/************************************************************************
 Method Name: GenerateDetectedObjects
 Description:
	- Generate set of 'CDetectedObject' with input detection.
 Input Arguments:
	- _frameImage        : Input frame image
	- _vecDetections     : Input detections
	- _vecDetectedObjects: (output) Pre-processed detections
 Return Values:
	- None
************************************************************************/
void CDepthMTTracker::GenerateDetectedObjects(
	const cv::Mat _frameImage,
	DetectionSet &_vecDetections,
	std::vector<CDetectedObject> &_vecDetectedObjects)
{
	/* reset 'vecDetectedObjecs' for usage of the current frame */
	size_t detectionID = 0;
	vecDetectedObjects_.clear();
	vecDetectedObjects_.reserve(_vecDetections.size());
	for (size_t detectionIdx = 0; detectionIdx < _vecDetections.size(); detectionIdx++)
	{
		/* generate detection information */
		CDetectedObject curDetection;
		curDetection.id            = (unsigned int)detectionID++;
		curDetection.detection     = _vecDetections[detectionIdx];
		curDetection.detection.box = hj::Rescale(curDetection.detection.box, stParam_.dImageRescale);
		curDetection.depth         = GetEstimatedDepth(_frameImage, _vecDetections[detectionIdx].box);
		curDetection.bMatchedWithTracklet     = false;
		curDetection.bCoveredByOtherDetection = false;
		curDetection.vecvecTrackedFeatures.reserve(stParam_.nBackTrackingLength);
		curDetection.boxes.reserve(stParam_.nBackTrackingLength);
		curDetection.boxes.push_back(curDetection.detection.box);

		_vecDetectedObjects.push_back(curDetection);
	}
}


/************************************************************************
 Method Name: BackwardTracking
 Description:
	- Track the input detections in a backward direction with input frame
	  buffers.
 Input Arguments: 
	- _vecDetectedObjects: (in/out) pre-processed detections with backward tracking results
 Return Values:
	- None
************************************************************************/
void CDepthMTTracker::BackwardTracking(std::vector<CDetectedObject> &_vecDetectedObjects)
{
	//---------------------------------------------------
	// BACKWARD FEATURE TRACKING
	//---------------------------------------------------
	for (size_t dIdx = 0; dIdx < _vecDetectedObjects.size(); dIdx++)
	{			
		cv::Rect rectRescaledDetectionBox = hj::Rescale(_vecDetectedObjects[dIdx].detection.box, stParam_.dImageRescale);
		cv::Rect rectEstimatedBox;

		std::vector<cv::Point2f> vecInputFeatures, vecTrackedFeatures;
		std::vector<int>         vecInlireIndices;

		// feature extraction
		FeatureExtraction(_vecDetectedObjects[dIdx].boxes.front(), *cImageBuffer_.rbegin(), vecInputFeatures);
		_vecDetectedObjects[dIdx].vecvecTrackedFeatures.push_back(vecInputFeatures);

		/* feature tracking */
		for (hj::CMatFIFOBuffer::reverse_iterator bufferIter = cImageBuffer_.rbegin();
			bufferIter != cImageBuffer_.rend() - 1;
			bufferIter++)
		{
			if (!FeatureTracking(rectRescaledDetectionBox,
				                 *bufferIter,
				                 *(bufferIter + 1),
				                 vecInputFeatures,
				                 vecTrackedFeatures,
				                 vecInlireIndices,
				                 rectEstimatedBox))
			{
				break;
			}

			_vecDetectedObjects[dIdx].boxes.push_back(hj::Rescale(rectEstimatedBox, stParam_.dImageRescaleRecover));

			/* save inliers */
			vecInputFeatures.clear();
			vecInputFeatures.reserve(vecInlireIndices.size());
			for (size_t indexIdx = 0; indexIdx < vecInlireIndices.size(); indexIdx++)
			{
				vecInputFeatures.push_back(vecTrackedFeatures[vecInlireIndices[indexIdx]]);
			}
			_vecDetectedObjects[dIdx].vecvecTrackedFeatures.push_back(vecInputFeatures);
		}
	}

	//---------------------------------------------------
	// CHECK OVERLAP
	//---------------------------------------------------
	for (int detect1Idx = 0; detect1Idx < _vecDetectedObjects.size(); detect1Idx++)
	{
		if (_vecDetectedObjects[detect1Idx].bCoveredByOtherDetection) { continue; }
		for (int detect2Idx = detect1Idx + 1; detect2Idx < _vecDetectedObjects.size(); detect2Idx++)
		{
			if (hj::CheckOverlap(_vecDetectedObjects[detect1Idx].detection.box, _vecDetectedObjects[detect2Idx].detection.box))
			{
				_vecDetectedObjects[detect1Idx].bCoveredByOtherDetection = true;
				break;
			}
		}
	}
}


/************************************************************************
 Method Name: ForwardTracking
 Description:
	- Estimate trackers positions at the current frame. Estimated positions
	  are inserted at the end of box array of each tracker.
 Input Arguments:
	- _queueTrackers: (in/out) trackers of which we want to estimate the location.
 Return Values:
	- None
************************************************************************/
void CDepthMTTracker::ForwardTracking(hj::TrackletPtQueue &_queueTracklets)
{
	if (2 > cImageBuffer_.num_elements())
	{ 
		// there is no frame to track
		return; 
	}

	//---------------------------------------------------
	// FORWARD FEATURE TRACKING
	//---------------------------------------------------
	for (size_t trackIdx = 0; trackIdx < _queueTracklets.size(); trackIdx++)
	{
		std::vector<cv::Point2f> vecTrackedFeatures;
		std::vector<int>         vecInlierIndices;
		cv::Rect estimatedBox;
		if (!FeatureTracking(_queueTracklets[trackIdx]->boxes.back(),
			                 *(cImageBuffer_.rbegin() + 1),
			                 *cImageBuffer_.rbegin(),
			                 _queueTracklets[trackIdx]->featurePoints,
			                 vecTrackedFeatures,
			                 vecInlierIndices,
			                 estimatedBox))
		{
			continue;
		}

		_queueTracklets[trackIdx]->trackedPoints.clear();
		if (stParam_.nMinNumFeatures > vecInlierIndices.size()) { continue; }
		for (size_t featureIdx = 0; featureIdx < vecInlierIndices.size(); featureIdx++)
		{
			_queueTracklets[trackIdx]->trackedPoints.push_back(vecTrackedFeatures[vecInlierIndices[featureIdx]]);
		}		
		_queueTracklets[trackIdx]->estimatedBox = estimatedBox;
	}
}


/************************************************************************
 Method Name: DetectionToTrackletMatching
 Description:
	- Do matching between detections and tracklets. Then, update tracklets.
 Input Arguments:
	- _vecDetectedObjects: (pre-processed) detections
	- _queueTracklets: trackers
 Return Values:
	- None
************************************************************************/
void CDepthMTTracker::DetectionToTrackletMatching(
	const std::vector<CDetectedObject> &_vecDetectedObjects,
	TrackletPtQueue &_queueTracklets)
{
	/////////////////////////////////////////////////////////////////////////////
	// CALCULATE MATCHING COSTS
	/////////////////////////////////////////////////////////////////////////////

	// matching cost matrix: default score is an infinite
	arrTrackletToDetectionMatchingCost_.clear();
	arrTrackletToDetectionMatchingCost_.resize(_vecDetectedObjects.size() * _queueTracklets.size(), std::numeric_limits<float>::infinity());
	
	// to determine occlusion
	std::vector<std::deque<int>> featuresInDetectionBox(_vecDetectedObjects.size());

	//---------------------------------------------------
	// COST WITH BI-DIRECTIONAL TRACKING
	//---------------------------------------------------
	for (size_t trackIdx = 0; trackIdx < _queueTracklets.size(); trackIdx++)
	{
		CTracklet *curTracker = _queueTracklets[trackIdx];

		for (size_t detectIdx = 0, costPos = trackIdx; 
			detectIdx < _vecDetectedObjects.size();
			detectIdx++, costPos += _queueTracklets.size())
		{
			// validate with backward tracking result
			if (!hj::CheckOverlap(curTracker->estimatedBox, _vecDetectedObjects[detectIdx].detection.box))
				continue;

			// count feature points inside the detection box
			for (int featureIdx = 0; featureIdx < curTracker->trackedPoints.size(); featureIdx++)
			{
				if (!_vecDetectedObjects[detectIdx].detection.box.contains(curTracker->trackedPoints[featureIdx]))
					continue;
				featuresInDetectionBox[detectIdx].push_back((int)trackIdx);
			}		

			// determine the possible longest comparison interval
			size_t lengthForCompare = std::min((size_t)stParam_.nBackTrackingLength, 
				std::min(curTracker->boxes.size(), _vecDetectedObjects[detectIdx].boxes.size()));
			
			// croll tracker boxes for comparison (reverse ordering)
			size_t numBoxCopy = lengthForCompare - 1;
			std::vector<cv::Rect> vecTrackerBoxes;
			vecTrackerBoxes.reserve(lengthForCompare);
			vecTrackerBoxes.push_back(curTracker->estimatedBox);
			if (0 < numBoxCopy)
			{
				vecTrackerBoxes.insert(vecTrackerBoxes.begin() + 1,
					curTracker->boxes.rbegin(), 
					curTracker->boxes.rbegin() + lengthForCompare - 1);
			}

			double boxCost = 0.0;
			cv::Rect detectionBox, trackerBox;
			for (size_t boxIdx = 0; boxIdx < lengthForCompare; boxIdx++)
			{
				detectionBox = _vecDetectedObjects[detectIdx].boxes[boxIdx];
				trackerBox   = vecTrackerBoxes[boxIdx];

				if (!hj::CheckOverlap(detectionBox, trackerBox) // rejection criterion
					|| stParam_.dMaxBoxDistance < BoxCenterDistanceWRTScale(detectionBox, trackerBox)
					|| stParam_.dMinBoxOverlapRatio > hj::OverlappedArea(detectionBox, trackerBox) / std::min(detectionBox.area(), trackerBox.area())
					|| stParam_.dMaxBoxCenterDiffRatio * std::max(detectionBox.width, trackerBox.width) < hj::NormL2(hj::Center(detectionBox) - hj::Center(trackerBox)))
				{
					boxCost = std::numeric_limits<double>::infinity();
					break;
				}
				boxCost += BoxCenterDistanceWRTScale(trackerBox, detectionBox);
			}
			if (std::numeric_limits<double>::infinity() == boxCost) { continue; }
			boxCost /= (double)lengthForCompare;

			arrTrackletToDetectionMatchingCost_[costPos] = (float)boxCost;
		}
	}

	//---------------------------------------------------
	// OCCLUSION HANDLING
	//---------------------------------------------------	
	// If a detection box contains feature points from more than one tracker, we examine whether there exists a
	// dominant tracker or not. If there is no dominant tracker, that means the ownership of the detection is 
	// not clear, we set the scores between that detection and trackers to infinite. This yields termination of
	// all related trackers.

	int numFeatureFromMajorTracker = 0,
		numFeatureFromCurrentTracker = 0,
		majorTrackerIdx = 0,
		currentTrackerIdx = 0;

	for (size_t detectIdx = 0, costPos = 0; 
		detectIdx < _vecDetectedObjects.size(); 
		detectIdx++, costPos += _queueTracklets.size())
	{
		if (0 == featuresInDetectionBox[detectIdx].size()) { continue; }

		// find dominant tracker of the detection
		numFeatureFromMajorTracker = numFeatureFromCurrentTracker = 0;
		majorTrackerIdx = featuresInDetectionBox[detectIdx].front();
		currentTrackerIdx = featuresInDetectionBox[detectIdx].front();
		for (int featureIdx = 0; featureIdx < featuresInDetectionBox[detectIdx].size(); featureIdx++)
		{
			if (currentTrackerIdx == featuresInDetectionBox[detectIdx][featureIdx])
			{
				// we assume that the same tracker indices in 'featuresInDetectionBox' are grouped together
				numFeatureFromCurrentTracker++;
				continue;
			}

			if (numFeatureFromCurrentTracker > numFeatureFromMajorTracker)
			{
				majorTrackerIdx = currentTrackerIdx;
				numFeatureFromMajorTracker = numFeatureFromCurrentTracker;
			}
			currentTrackerIdx = featuresInDetectionBox[detectIdx][featureIdx];
			numFeatureFromCurrentTracker = 0;
		}

		// case 1: only one tracker has its features points in the detecion box
		if (featuresInDetectionBox[detectIdx].front() == currentTrackerIdx)
		{
			continue;
		}

		// case 2: there is a domninant tracker among related trackers
		if (numFeatureFromMajorTracker > featuresInDetectionBox[detectIdx].size() * stParam_.dMinOpticalFlowMajorityRatio)
		{
			continue;
		}

		// case 3: more than one trackers are related and there is no dominant tracker
		for (size_t infCostPos = costPos; infCostPos < costPos + _queueTracklets.size(); infCostPos++)
		{
			arrTrackletToDetectionMatchingCost_[infCostPos] = std::numeric_limits<float>::infinity();
		}
	}

	//---------------------------------------------------
	// INFINITE HANDLING
	//---------------------------------------------------
	// To ensure a proper operation of our Hungarian implementation, we convert infinite to the finite value
	// that is little bit (=100.0f) greater than the maximum finite cost in the original cost function.
	float maxCost = -1000.0f;
	for (int costIdx = 0; costIdx < arrTrackletToDetectionMatchingCost_.size(); costIdx++)
	{
		if (!_finitef(arrTrackletToDetectionMatchingCost_[costIdx])) { continue; }
		if (maxCost < arrTrackletToDetectionMatchingCost_[costIdx]) { maxCost = arrTrackletToDetectionMatchingCost_[costIdx]; }
	}
	maxCost = maxCost + 100.0f;
	for (int costIdx = 0; costIdx < arrTrackletToDetectionMatchingCost_.size(); costIdx++)
	{
		if (_finitef(arrTrackletToDetectionMatchingCost_[costIdx])) { continue; }
		arrTrackletToDetectionMatchingCost_[costIdx] = maxCost;
	}


	/////////////////////////////////////////////////////////////////////////////
	// MATCHING
	/////////////////////////////////////////////////////////////////////////////
	trackingResult_.objectInfos.clear();
	size_t numDetection = this->vecDetectedObjects_.size();
	CHungarianMethod cHungarianMatcher;
	cHungarianMatcher.Initialize(arrTrackletToDetectionMatchingCost_, (unsigned int)numDetection, (unsigned int)this->queueActiveTracklets_.size());
	stMatchInfo *curMatchInfo = cHungarianMatcher.Match();
	for (size_t matchIdx = 0; matchIdx < curMatchInfo->rows.size(); matchIdx++)
	{
		if (maxCost == curMatchInfo->matchCosts[matchIdx]) { continue; }
		CDetectedObject *curDetection = &this->vecDetectedObjects_[curMatchInfo->rows[matchIdx]];
		CTracklet *curTracker = this->queueActiveTracklets_[curMatchInfo->cols[matchIdx]];

		//---------------------------------------------------
		// MATCHING VALIDATION
		//---------------------------------------------------
		if (curTracker->duration >= (unsigned int)stParam_.nMaxTrackletLength) { continue; }
		double curConfidence = 1.0;

		//---------------------------------------------------
		// TRACKER UPDATE
		//---------------------------------------------------		
		curTracker->timeEnd        = this->nCurrentFrameIdx_;
		curTracker->timeLastUpdate = this->nCurrentFrameIdx_;
		curTracker->duration       = curTracker->timeEnd - curTracker->timeStart + 1;
		curTracker->numStatic      = 0;		
		curTracker->confidence     = curConfidence;		
		curTracker->boxes.push_back(curDetection->detection.box);
		curTracker->depths.push_back(curDetection->depth);

		curDetection->bMatchedWithTracklet = true;		

		// update features with detection (after result packaging)
		curTracker->featurePoints = curDetection->vecvecTrackedFeatures.front();
		curTracker->trackedPoints.clear();
	}
	cHungarianMatcher.Finalize();


	/////////////////////////////////////////////////////////////////////////////
	// TRACKER GENERATION
	/////////////////////////////////////////////////////////////////////////////	
	for (std::vector<CDetectedObject>::iterator detectionIter = this->vecDetectedObjects_.begin();
		detectionIter != this->vecDetectedObjects_.end();
		detectionIter++)
	{
		if ((*detectionIter).bMatchedWithTracklet) { continue; }

		CTracklet newTracker;
		newTracker.id = this->nNewTrackletID_++;
		newTracker.timeStart = this->nCurrentFrameIdx_;
		newTracker.timeEnd = this->nCurrentFrameIdx_;
		newTracker.timeLastUpdate = this->nCurrentFrameIdx_;
		newTracker.duration = 1;
		newTracker.numStatic = 0;
		newTracker.boxes.push_back((*detectionIter).detection.box);
		newTracker.depths.push_back((*detectionIter).depth);
		newTracker.featurePoints = (*detectionIter).vecvecTrackedFeatures.front();
		newTracker.trackedPoints.clear();
		newTracker.confidence = 1.0;		

		// generate tracklet instance
		this->listCTracklet_.push_back(newTracker);
	}


	/////////////////////////////////////////////////////////////////////////////
	// TRACKER TERMINATION
	/////////////////////////////////////////////////////////////////////////////
	TrackletPtQueue newActiveTracklets;
	for (std::list<CTracklet>::iterator trackerIter = listCTracklet_.begin();
		trackerIter != listCTracklet_.end();
		/*trackerIter++*/)
	{
		if ((*trackerIter).timeLastUpdate + stParam_.nMaxPendingTime < nCurrentFrameIdx_)
		{
			// termination			
			trackerIter = this->listCTracklet_.erase(trackerIter);
			continue;
		}
		if ((*trackerIter).timeLastUpdate == nCurrentFrameIdx_)
			newActiveTracklets.push_back(&(*trackerIter));
		trackerIter++;		
	}
	queueActiveTracklets_ = newActiveTracklets;
}


/************************************************************************
 Method Name: TrackletToTrajectoryMatching
 Description:
	- Do matching between tracklets and trajectories. Then, update trajectories.
 Input Arguments:
	- _queueActiveTracklets: Input tracklets.
 Return Values:
	- None
************************************************************************/
void CDepthMTTracker::TrackletToTrajectoryMatching(const TrackletPtQueue &_queueActiveTracklets)
{
	// trajectory update
	TrackletPtQueue newTracklets;
	for (size_t i = 0; i < _queueActiveTracklets.size(); i++)
	{
		if (NULL == _queueActiveTracklets[i]->ptTrajectory) 
		{
			newTracklets.push_back(_queueActiveTracklets[i]);
			continue;
		}
		CTrajectory *curTrajectory = _queueActiveTracklets[i]->ptTrajectory;
		curTrajectory->timeEnd = this->nCurrentFrameIdx_;
		curTrajectory->timeLastUpdate = curTrajectory->timeEnd;
		curTrajectory->duration = curTrajectory->timeEnd - curTrajectory->timeStart + 1;
		curTrajectory->boxes.push_back(_queueActiveTracklets[i]->boxes.back());
		curTrajectory->depths.push_back(_queueActiveTracklets[i]->depths.back());
	}

	// delete expired trajectories and find pending trajectories
	queueActiveTrajectories_.clear();
	std::vector<CTrajectory*> vecPendedTrajectories;
	vecPendedTrajectories.reserve(listCTrajectories_.size());
	for (std::list<CTrajectory>::iterator trajIter = listCTrajectories_.begin();
		trajIter != listCTrajectories_.end();
		/*trajIter++*/)
	{
		if (trajIter->timeLastUpdate + stParam_.nMaxPendingTime < this->nCurrentFrameIdx_)
		{
			// termination			
			trajIter = this->listCTrajectories_.erase(trajIter);
			continue;
		}
		if (trajIter->timeLastUpdate == this->nCurrentFrameIdx_)
			queueActiveTrajectories_.push_back(&(*trajIter));
		else
			vecPendedTrajectories.push_back(&(*trajIter));

		trajIter++;
	}

	// trajectory-to-tracklet matching
	arrInterTrackletMatchingCost_.clear();
	arrInterTrackletMatchingCost_.resize(
		vecPendedTrajectories.size() * newTracklets.size(),
		std::numeric_limits<float>::infinity());
	for (size_t trajIdx = 0; trajIdx < vecPendedTrajectories.size(); trajIdx++)
	{
		for (size_t newIdx = 0, costPos = trajIdx;
			newIdx < newTracklets.size();
			newIdx++, costPos += vecPendedTrajectories.size())
		{
			double curCost = 0.0;

			// TODO: translation + depth distance
			double distTranslate = hj::NormL2(hj::Center(vecPendedTrajectories[trajIdx]->boxes.back()) - hj::Center(newTracklets[newIdx]->boxes.back()));
			if (distTranslate > stParam_.dMaxTranslationDistance)
				continue;
			curCost += distTranslate;

			double distDepth = std::abs(vecPendedTrajectories[trajIdx]->depths.back() - newTracklets[newIdx]->depths.back());
			if (distDepth > stParam_.dMaxDepthDistance)
				continue;
			curCost += distDepth;

			arrInterTrackletMatchingCost_[costPos] = (float)curCost;
		}
	}

	// handling infinite in the cost array
	float maxCost = -1000.0f;
	for (int costIdx = 0; costIdx < arrInterTrackletMatchingCost_.size(); costIdx++)
	{
		if (!_finitef(arrInterTrackletMatchingCost_[costIdx]))
			continue;
		if (maxCost < arrInterTrackletMatchingCost_[costIdx])
			maxCost = arrInterTrackletMatchingCost_[costIdx];
	}
	maxCost = maxCost + 100.0f;
	for (int costIdx = 0; costIdx < arrInterTrackletMatchingCost_.size(); costIdx++)
	{
		if (_finitef(arrInterTrackletMatchingCost_[costIdx]))
			continue;
		arrInterTrackletMatchingCost_[costIdx] = maxCost;
	}

	//---------------------------------------------------
	// MATCHING
	//---------------------------------------------------
	CHungarianMethod cHungarianMatcher;
	cHungarianMatcher.Initialize(arrInterTrackletMatchingCost_, (unsigned int)newTracklets.size(), (unsigned int)vecPendedTrajectories.size());
	stMatchInfo *curMatchInfo = cHungarianMatcher.Match();
	for (size_t matchIdx = 0; matchIdx < curMatchInfo->rows.size(); matchIdx++)
	{
		if (maxCost == curMatchInfo->matchCosts[matchIdx]) { continue; }
		CTracklet *curTracklet = newTracklets[curMatchInfo->rows[matchIdx]];
		CTrajectory *curTrajectory = vecPendedTrajectories[curMatchInfo->cols[matchIdx]];

		curTracklet->ptTrajectory = curTrajectory;

		// updata matched trajectory
		curTrajectory->timeEnd = this->nCurrentFrameIdx_;
		curTrajectory->timeLastUpdate = curTrajectory->timeEnd;
		curTrajectory->duration = curTrajectory->timeEnd - curTrajectory->timeStart + 1;
		curTrajectory->boxes.push_back(curTracklet->boxes.back());
		curTrajectory->depths.push_back(curTracklet->depths.back());
		curTrajectory->trackletIDs.push_back(curTracklet->id);

		queueActiveTrajectories_.push_back(curTrajectory);
	}
	cHungarianMatcher.Finalize();

	/////////////////////////////////////////////////////////////////////////////
	// TRAJECTORY GENERATION
	/////////////////////////////////////////////////////////////////////////////	
	for (TrackletPtQueue::iterator trackletIter = newTracklets.begin();
		trackletIter != newTracklets.end();
		trackletIter++)
	{
		if ((*trackletIter)->ptTrajectory != NULL) { continue; }

		CTrajectory newTrajectory;
		newTrajectory.id = this->nNewTrajectoryID_++;
		newTrajectory.timeStart = this->nCurrentFrameIdx_;
		newTrajectory.timeEnd = this->nCurrentFrameIdx_;
		newTrajectory.timeLastUpdate = this->nCurrentFrameIdx_;
		newTrajectory.duration = 1;
		newTrajectory.boxes.push_back((*trackletIter)->boxes.back());
		newTrajectory.depths.push_back((*trackletIter)->depths.back());
		newTrajectory.trackletIDs.push_back((*trackletIter)->id);

		// generate trajectory instance
		this->listCTrajectories_.push_back(newTrajectory);		
		(*trackletIter)->ptTrajectory = &this->listCTrajectories_.back();
		queueActiveTrajectories_.push_back(&this->listCTrajectories_.back());
	}
}


/************************************************************************
 Method Name: ResultPackaging
 Description:
	- Packaging the tracking result into 'trackingResult_'
 Input Arguments:
	- None
 Return Values:
	- None
************************************************************************/
void CDepthMTTracker::ResultPackaging()
{
	time_t timePackaging = clock();	
	trackingResult_.frameIdx  = nCurrentFrameIdx_;
	trackingResult_.timeStamp = (unsigned int)timePackaging;
	trackingResult_.objectInfos.clear();	
	for (size_t tIdx = 0; tIdx < queueActiveTrajectories_.size(); tIdx++)
	{
		trackingResult_.objectInfos.push_back(GetObjectInfo(queueActiveTrajectories_[tIdx]));
	}

	int cost_pos = 0;
	if (!this->trackingResult_.matMatchingCost.empty()) { trackingResult_.matMatchingCost.release(); }
	trackingResult_.matMatchingCost = 
		cv::Mat((int)vecDetectedObjects_.size(), (int)trackingResult_.vecTrackerRects.size(), CV_32F);
	for (int detectionIdx = 0; detectionIdx < vecDetectedObjects_.size(); detectionIdx++)
	{
		for (int trackIdx = 0; trackIdx < trackingResult_.vecTrackerRects.size(); trackIdx++)
		{
			this->trackingResult_.matMatchingCost.at<float>(detectionIdx, trackIdx) = arrTrackletToDetectionMatchingCost_[cost_pos];
			cost_pos++;
		}
	}
}


/************************************************************************
 Method Name: FeatureExtraction
 Description:
	- Tracks the input feature points of the input frame in the target frame.
 Input Arguments:
	- _inputBox         : bounding box of the target on the input frame
	- _inputImage       : image containing the input feature points
	- vecFeaturePoints : target image of feature point tracking
 Return Values:
	- bool: (true) success / (false) fail
************************************************************************/
bool CDepthMTTracker::FeatureExtraction(
	const cv::Rect _inputBox,
	const cv::Mat  _inputImage,
	std::vector<cv::Point2f> &_vecFeaturePoints)
{
	_vecFeaturePoints.clear();

	//---------------------------------------------------
	// EXTRACT FEATURE POINTS
	//---------------------------------------------------
	std::vector<cv::KeyPoint> newKeypoints;
	cv::Rect rectROI = hj::CropWithSize(_inputBox, cv::Size(_inputImage.cols, _inputImage.rows));
	matFeatureExtractionMask_(rectROI) = cv::Scalar(255); // masking with the bounding box
	featureDetector_->detect(_inputImage, newKeypoints, matFeatureExtractionMask_);
	matFeatureExtractionMask_(rectROI) = cv::Scalar(0);   // restore the mask image

	if (stParam_.nMinNumFeatures > newKeypoints.size())
	{
		// it is impossible to track the target because there are insufficient number of feature points
		return false;
	}

	//---------------------------------------------------
	// EXTRACT SELECTION
	//---------------------------------------------------	
	std::random_shuffle(newKeypoints.begin(), newKeypoints.end());
	for (int pointIdx = 0; pointIdx < std::min((int)newKeypoints.size(), stParam_.nMaxNumFeatures); pointIdx++)
	{
		_vecFeaturePoints.push_back(newKeypoints[pointIdx].pt);
	}

	return true;
}


/************************************************************************
 Method Name: FeatureTracking
 Description:
	- Tracks the input feature points of the input frame in the target frame.
 Input Arguments:
	- _inputBox         : bounding box of the target at the input frame
	- _inputImage       : image containing the input feature points.
	- _targetImage      : target image of feature point tracking.
	- _vecInputFeatures : input feature points.
	- _vecOutputFeatures: points of tracking result. Actually it is an output.
	- _vecFeatureInlierIndex: index of features that are inliers of estimated motion.
	- _trackingResult   : (output) estimated box at the target frame.
 Return Values:
	- bool: (true) success / (false) fail
************************************************************************/
bool CDepthMTTracker::FeatureTracking(
	const cv::Rect _inputBox,
	const cv::Mat  _inputImage,
	const cv::Mat  _targetImage,
	std::vector<cv::Point2f> &_vecInputFeatures,
	std::vector<cv::Point2f> &_vecOutputFeatures,
	std::vector<int>         &_vecFeatureInlierIndex,
	cv::Rect &_trackingResult)
{
	if (0 == _vecInputFeatures.size()) { return false; }

	_trackingResult = cv::Rect2d(0.0, 0.0, 0.0, 0.0);

	//---------------------------------------------------
	// CONVERT TO GRAY SCALE IMAGES
	//---------------------------------------------------
	cv::Mat currImage, nextImage;
	
	if (1 == _inputImage.channels()) { currImage = _inputImage; }
	else { cv::cvtColor(_inputImage, currImage, CV_BGR2GRAY); }
	
	if (1 == _targetImage.channels()) { nextImage = _targetImage; }
	else { cv::cvtColor(_targetImage, nextImage, CV_BGR2GRAY); }

	//---------------------------------------------------
	// EXTRACT FEATURE POINTS
	//---------------------------------------------------
	if (_vecInputFeatures.empty())
	{
		if (!FeatureExtraction(_inputBox, currImage, _vecInputFeatures))
		{
			return false;
		}
	}

	//---------------------------------------------------
	// FEATURE TRACKING
	//---------------------------------------------------
	std::vector<uchar> vecFeatureStatus;
	_vecOutputFeatures.clear();

	cv::Mat vecErrors;
	cv::calcOpticalFlowPyrLK(
		currImage,
		nextImage,
		_vecInputFeatures,
		_vecOutputFeatures,
		vecFeatureStatus,
		cv::noArray(),
		//vecErrors,
		cv::Size((int)(_inputBox.width * stParam_.dFeatureTrackWindowSizeRatio),
		         (int)(_inputBox.width * stParam_.dFeatureTrackWindowSizeRatio)));

	//---------------------------------------------------
	// BOX ESTIMATION
	//---------------------------------------------------	
	cv::Rect newRect = LocalSearchKLT(_inputBox, _vecInputFeatures, _vecOutputFeatures, _vecFeatureInlierIndex);
	if (stParam_.nMinNumFeatures > _vecFeatureInlierIndex.size())
	{ 
		// tracking failure because of insufficient number of tracked feature points
		return false; 
	}
	else
	{
		_trackingResult = newRect;
	}

	return true;
}


/************************************************************************
 Method Name: FindInlierFeatures
 Description:
	- Find inlier feature points
 Input Arguments:
	- _vecInputFeatures : input feature points.
	- _vecOutputFeatures: points of tracking result. Actually it is an output.
	- _vecPointStatus   : tracking status of each point.
 Return Values:
	- std::vector<cv::Point2f>: Inlier points.
************************************************************************/
std::vector<cv::Point2f> CDepthMTTracker::FindInlierFeatures(
	std::vector<cv::Point2f> *_vecInputFeatures,
	std::vector<cv::Point2f> *_vecOutputFeatures,
	std::vector<unsigned char> *_vecPointStatus)
{
	size_t numTrackedFeatures = 0;
	// find center of disparity
	cv::Point2f disparityCenter(0.0f, 0.0f);
	std::vector<cv::Point2f> vecDisparity;
	std::vector<size_t> vecInlierIndex;
	for (size_t pointIdx = 0; pointIdx < _vecPointStatus->size(); pointIdx++)
	{
		if (!(*_vecPointStatus)[pointIdx]) { continue; }
		vecDisparity.push_back((*_vecOutputFeatures)[pointIdx] - (*_vecInputFeatures)[pointIdx]);
		disparityCenter += vecDisparity.back();
		vecInlierIndex.push_back(pointIdx);
		numTrackedFeatures++;
	}
	disparityCenter = (1 / (float)numTrackedFeatures) * disparityCenter;

	// find distribution of disparity norm
	float norm;
	float normAverage = 0.0f;
	float normSqauredAverage = 0.0f;
	float normStd = 0.0;
	std::vector<float> vecNorm;
	for (size_t pointIdx = 0; pointIdx < vecDisparity.size(); pointIdx++)
	{
		norm = (float)cv::norm(vecDisparity[pointIdx] - disparityCenter);
		vecNorm.push_back(norm);
		normAverage += norm;
		normSqauredAverage += norm * norm;
	}
	normAverage /= (float)numTrackedFeatures;
	normSqauredAverage /= (float)numTrackedFeatures;
	normStd = sqrtf(((float)numTrackedFeatures / ((float)numTrackedFeatures - 1)) * (normSqauredAverage - (normAverage * normAverage)));

	std::vector<cv::Point2f> vecInlierFeatures;
	for (size_t pointIdx = 0; pointIdx < vecNorm.size(); pointIdx++)
	{
		if (abs(vecNorm[pointIdx] - normAverage) > 1 * normStd) { continue; }
		vecInlierFeatures.push_back((*_vecOutputFeatures)[vecInlierIndex[pointIdx]]);
	}

	return vecInlierFeatures;
}


/************************************************************************
 Method Name: LocalSearchKLT
 Description:
	- estimate current box location with feature tracking result
 Input Arguments:
	- _preFeatures       : feature positions at the previous frame
	- _curFeatures       : feature positions at the current frame
	- _inlierFeatureIndex: (output) indicates inlier features
 Return Values:
	- cv::Rect: estimated box
************************************************************************/
#define PSN_LOCAL_SEARCH_PORTION_INLIER (false)
#define PSN_LOCAL_SEARCH_MINIMUM_MOVEMENT (0.1)
#define PSN_LOCAL_SEARCH_NEIGHBOR_WINDOW_SIZE_RATIO (0.2)
cv::Rect CDepthMTTracker::LocalSearchKLT(
	cv::Rect _preBox,
	std::vector<cv::Point2f> &_preFeatures,
	std::vector<cv::Point2f> &_curFeatures,
	std::vector<int> &_inlierFeatureIndex)
{
	size_t numFeatures = _preFeatures.size();
	size_t numMovingFeatures = 0;
	_inlierFeatureIndex.clear();
	_inlierFeatureIndex.reserve(numFeatures);

	// find disparity of moving features
	std::vector<cv::Point2d> vecMovingVector;
	std::vector<int> vecMovingFeatuerIdx;
	std::vector<double> vecDx;
	std::vector<double> vecDy;
	vecMovingVector.reserve(numFeatures);
	vecMovingFeatuerIdx.reserve(numFeatures);
	vecDx.reserve(numFeatures);
	vecDy.reserve(numFeatures);
	cv::Point2d movingVector;
	double disparity = 0.0;
	for (int featureIdx = 0; featureIdx < (int)numFeatures; featureIdx++)
	{
		movingVector = _curFeatures[featureIdx] - _preFeatures[featureIdx];
		disparity = hj::NormL2(movingVector);
		if (disparity < PSN_LOCAL_SEARCH_MINIMUM_MOVEMENT * stParam_.dImageRescale) { continue; }

		vecMovingVector.push_back(movingVector);
		vecMovingFeatuerIdx.push_back(featureIdx);
		vecDx.push_back(movingVector.x);
		vecDy.push_back(movingVector.y);

		numMovingFeatures++;
	}

	// check static movement
	if (numMovingFeatures < numFeatures * 0.5)
	{ 
		for (int featureIdx = 0; featureIdx < (int)numFeatures; featureIdx++)
		{
			if (_preBox.contains(_curFeatures[featureIdx]))
			{
				_inlierFeatureIndex.push_back(featureIdx);
			}
		}
		return _preBox;
	}

	std::sort(vecDx.begin(), vecDx.end());
	std::sort(vecDy.begin(), vecDy.end());

	// estimate major disparity
	double windowSize = _preBox.width * PSN_LOCAL_SEARCH_NEIGHBOR_WINDOW_SIZE_RATIO * stParam_.dImageRescale;
	size_t maxNeighborX = 0, maxNeighborY = 0;
	cv::Point2d estimatedDisparity;
	for (size_t disparityIdx = 0; disparityIdx < numMovingFeatures; disparityIdx++)
	{
		size_t numNeighborX = 0;
		size_t numNeighborY = 0;
		// find neighbors in each axis
		for (size_t compIdx = 0; compIdx < numMovingFeatures; compIdx++)
		{
			if (std::abs(vecDx[disparityIdx] - vecDx[compIdx]) < windowSize) { numNeighborX++; } // X		
			if (std::abs(vecDy[disparityIdx] - vecDy[compIdx]) < windowSize) { numNeighborY++; } // Y
		}
		// disparity in X axis
		if (maxNeighborX < numNeighborX)
		{
			estimatedDisparity.x = vecDx[disparityIdx];
			maxNeighborX = numNeighborX;
		}
		// disparity in Y axis
		if (maxNeighborY < numNeighborY)
		{
			estimatedDisparity.y = vecDy[disparityIdx];
			maxNeighborY = numNeighborY;
		}
	}

	// find inliers
	for (int vectorIdx = 0; vectorIdx < (int)numMovingFeatures; vectorIdx++)
	{
		if (hj::NormL2(vecMovingVector[vectorIdx] - estimatedDisparity) < windowSize)
		{
			_inlierFeatureIndex.push_back(vecMovingFeatuerIdx[vectorIdx]);
		}
	}

	// estimate box
	cv::Rect2d estimatedBox = _preBox;
	estimatedBox.x += estimatedDisparity.x;
	estimatedBox.y += estimatedDisparity.y;

	return estimatedBox;
}


/************************************************************************
 Method Name: BoxCenterDistanceWRTScale
 Description:
	- Calculate the distance between two boxes
 Input Arguments:
	- _box1: the first box
	- _box2: the second box
 Return Values:
	- double: distance between two boxes
************************************************************************/
double CDepthMTTracker::BoxCenterDistanceWRTScale(cv::Rect &_box1, cv::Rect &_box2)
{
	double nominator = hj::NormL2(hj::Center(_box1) - hj::Center(_box2));
	double denominator = (_box1.width + _box2.width) / 2.0;
	double boxDistance = (nominator * nominator) / (denominator * denominator);

	return boxDistance;
}


/************************************************************************
 Method Name: GetTrackingConfidence
 Description:
	- Calculate the confidence of tracking by the number of features lay in the box
 Input Arguments:
	- _box: target position
	- _vecTrackedFeatures: tracked features
 Return Values:
	- tracking confidence
************************************************************************/
double CDepthMTTracker::GetTrackingConfidence(
	cv::Rect &_box, 
	std::vector<cv::Point2f> &_vecTrackedFeatures)
{
	double numFeaturesInBox = 0.0;
	for (std::vector<cv::Point2f>::iterator featureIter = _vecTrackedFeatures.begin();
		featureIter != _vecTrackedFeatures.end();
		featureIter++)
	{
		if (_box.contains(*featureIter))
		{
			numFeaturesInBox++;
		}
	}

	return numFeaturesInBox / (double)_vecTrackedFeatures.size();
}


/************************************************************************
Method Name: GetEstimatedDepth
Description:
	- Estimate the depth of head by averaging depth values in ceter region.
Input Arguments:
	- _frameImage: Entire current input frame image.
	- _objectBox : Head box.
Return Values:
	- double: Estimated detph of the head.
************************************************************************/
double CDepthMTTracker::GetEstimatedDepth(const cv::Mat _frameImage, const cv::Rect _objectBox)
{
	// center region
	int xMin = MAX(0, (int)(_objectBox.x + 0.5 * (1 - stParam_.dDepthEstimateCenterRegionRatio) * _objectBox.width)),
		xMax = MIN(_frameImage.cols, (int)(_objectBox.x + 0.5 * (1 + stParam_.dDepthEstimateCenterRegionRatio) * _objectBox.width) - 1),
		yMin = MAX(0, (int)(_objectBox.y + 0.5 * (1 - stParam_.dDepthEstimateCenterRegionRatio) * _objectBox.height)),
		yMax = MIN(_frameImage.rows, (int)(_objectBox.y + 0.5 * (1 + stParam_.dDepthEstimateCenterRegionRatio) * _objectBox.height) - 1);

	std::vector<uchar> vecCenterDepths;
	vecCenterDepths.reserve((int)(_objectBox.width * _objectBox.height));
	for (int r = yMin; r <= yMax; r++)
	{
		for (int c = xMin; c <= xMax; c++)
		{
			vecCenterDepths.push_back(_frameImage.at<uchar>(r, c));
		}
	}

	// histogram
	std::sort(vecCenterDepths.begin(), vecCenterDepths.end());
	uchar minDepth = vecCenterDepths.front();
	uchar maxDepth = vecCenterDepths.back();
	std::vector<std::pair<uchar, int>> histDepths(maxDepth - minDepth + 1);
	for (size_t i = 0; i < histDepths.size(); i++)
	{
		histDepths[i].first = minDepth + (uchar)i;
		histDepths[i].second = 0;
	}
	for (size_t i = 0; i < vecCenterDepths.size(); i++)
	{
		histDepths[vecCenterDepths[i]-minDepth].second++;
	}

	if (51 == this->nCurrentFrameIdx_)
	{
		int a = 0;
	}

	// count inliers 
	int maxNumInliers = 0;
	uchar estimatedCenterDepth = 0;
	for (int i = 0; i < histDepths.size(); i++)
	{
		int neighborStart = MAX(0, i - (int)(0.5 * stParam_.dDepthForegroundWindowSize));
		int neighborEnd = MIN((int)histDepths.size(), i + (int)(0.5 * stParam_.dDepthForegroundWindowSize)) - 1;
		int curNumInliers = 0;
		for (int j = neighborStart; j <= neighborEnd; j++)
		{
			curNumInliers += histDepths[j].second;
		}
		if (curNumInliers > maxNumInliers)
		{
			maxNumInliers = curNumInliers;
			estimatedCenterDepth = histDepths[i].first;
		}
	}

	return (double)estimatedCenterDepth;
}


/************************************************************************
 Method Name: GetObjectInfo
 Description:
	- Generate current frame's object info from the trajectory.
 Input Arguments:
	- _curTrajectory: Target trajectory.
 Return Values:
	- CObjectInfo: Current frame's state of the target trajectory.
************************************************************************/
CObjectInfo CDepthMTTracker::GetObjectInfo(CTrajectory *_curTrajectory)
{
	CObjectInfo outObjectInfo;
	assert(_curTrajectory->timeEnd == nCurrentFrameIdx_);

	cv::Rect curBox = hj::Rescale(_curTrajectory->boxes.back(), stParam_.dImageRescaleRecover);
	outObjectInfo.id = _curTrajectory->id;
	outObjectInfo.box = curBox;
	
	return outObjectInfo;
}


/************************************************************************
 Method Name: VisualizeResult
 Description:
	- Visualize the tracking result on the input image frame.
 Input Arguments:
	- None
 Return Values:
	- None
************************************************************************/
void CDepthMTTracker::VisualizeResult()
{
	/* frame information */
	char strFrameInfo[100];
	sprintf_s(strFrameInfo, "%04d", this->nCurrentFrameIdx_);
	cv::rectangle(matTrackingResult_, cv::Rect(5, 2, 60, 22), cv::Scalar(0, 0, 0), CV_FILLED);
	cv::putText(matTrackingResult_, strFrameInfo, cv::Point(6, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255));

	/* detections */
	for (int detIdx = 0; detIdx < vecDetectedObjects_.size(); detIdx++)
	{
		cv::rectangle(
			matTrackingResult_, 
			hj::Rescale(vecDetectedObjects_[detIdx].detection.box, stParam_.dImageRescaleRecover),
			cv::Scalar(255, 255, 255), 
			1);
	}

	/* tracklets */
	for (int tIdx = 0; tIdx < queueActiveTracklets_.size(); tIdx++)
	{
		CTracklet *curTracklet = queueActiveTracklets_[tIdx];

		// feature points
		for (int pointIdx = 0; pointIdx < curTracklet->featurePoints.size(); pointIdx++)
		{
			if (pointIdx < curTracklet->trackedPoints.size())
			{
				cv::circle(
					matTrackingResult_,
					curTracklet->featurePoints[pointIdx] * stParam_.dImageRescaleRecover,
					1, cv::Scalar(0, 255, 0), 1);
				cv::line(
					matTrackingResult_,
					curTracklet->featurePoints[pointIdx] * stParam_.dImageRescaleRecover,
					curTracklet->trackedPoints[pointIdx] * stParam_.dImageRescaleRecover,
					cv::Scalar(255, 255, 255), 1);
				cv::circle(
					matTrackingResult_,
					curTracklet->trackedPoints[pointIdx] * stParam_.dImageRescaleRecover,
					1, cv::Scalar(0, 255, 0), 1);
			}
			else
			{
				cv::circle(
					matTrackingResult_,
					curTracklet->featurePoints[pointIdx] * stParam_.dImageRescaleRecover,
					1, cv::Scalar(0, 0, 255), 1);
			}
		}

		// tracklet box
		//hj::DrawBoxWithID(matTrackingResult_, curObject->box, curObject->id, 0, 0, &vecColors_);
	}

	/* trajectories */
	for (int trajIdx = 0; trajIdx < queueActiveTrajectories_.size(); trajIdx++)
	{
		CTrajectory *curTrajectory = queueActiveTrajectories_[trajIdx];

		if (curTrajectory->timeEnd != this->nCurrentFrameIdx_) { continue; }
		DrawBoxWithID(
			matTrackingResult_, 
			hj::Rescale(curTrajectory->boxes.back(), stParam_.dImageRescaleRecover), 
			curTrajectory->id, 
			0, 
			0,
			getColorByID(curTrajectory->id, &vecColors_));
	}

	//---------------------------------------------------
	// RECORD
	//---------------------------------------------------
	if (bVideoWriterInit_)
	{
		IplImage *currentFrame = new IplImage(matTrackingResult_);
		cvWriteFrame(videoWriter_, currentFrame);
		delete currentFrame;
	}

	cv::namedWindow(strVisWindowName_);
	cv::moveWindow(strVisWindowName_, 10, 10);
	cv::imshow(strVisWindowName_, matTrackingResult_);
	cv::waitKey(1);
	matTrackingResult_.release();
}


cv::Scalar CDepthMTTracker::hsv2rgb(double h, double s, double v)
{
	int h_i = (int)(h * 6);
	double f = h * 6 - (double)h_i;
	double p = v * (1 - s);
	double q = v * (1 - f * s);
	double t = v * (1 - (1 - f) * s);
	double r, g, b;
	switch (h_i)
	{
	case 0: r = v; g = t; b = p; break;
	case 1: r = q; g = v; b = p; break;
	case 2: r = p; g = v; b = t; break;
	case 3: r = p; g = q; b = v; break;
	case 4: r = t; g = p; b = v; break;
	case 5: r = v; g = p; b = q; break;
	default:
		break;
	}

	return cv::Scalar((int)(r * 255), (int)(g * 255), (int)(b * 255));
}


std::vector<cv::Scalar> CDepthMTTracker::GenerateColors(unsigned int numColor)
{
	double golden_ratio_conjugate = 0.618033988749895;
	//double hVal = (double)std::rand()/(INT_MAX);
	double hVal = 0.0;
	std::vector<cv::Scalar> resultColors;
	resultColors.reserve(numColor);
	for (unsigned int colorIdx = 0; colorIdx < numColor; colorIdx++)
	{
		hVal += golden_ratio_conjugate;
		hVal = std::fmod(hVal, 1.0);
		resultColors.push_back(hsv2rgb(hVal, 0.5, 0.95));
	}
	return resultColors;
}


cv::Scalar CDepthMTTracker::getColorByID(unsigned int nID, std::vector<cv::Scalar> *vecColors)
{
	if (NULL == vecColors) { return cv::Scalar(255, 255, 255); }
	unsigned int colorIdx = nID % vecColors->size();
	return (*vecColors)[colorIdx];
}


void CDepthMTTracker::DrawBoxWithID(
	cv::Mat &imageFrame, 
	cv::Rect box, 
	unsigned int nID, 
	int lineStyle, 
	int fontSize, 
	cv::Scalar curColor)
{
	// get label length
	unsigned int labelLength = nID > 0 ? 0 : 1;
	unsigned int tempLabel = nID;
	while (tempLabel > 0)
	{
		tempLabel /= 10;
		labelLength++;
	}
	if (0 == fontSize)
	{
		cv::rectangle(imageFrame, box, curColor, 1);
		cv::rectangle(imageFrame, cv::Rect((int)box.x, (int)box.y - 10, 7 * labelLength, 14), curColor, CV_FILLED);
		cv::putText(imageFrame, std::to_string(nID), cv::Point((int)box.x, (int)box.y - 1), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0));
	}
	else
	{
		cv::rectangle(imageFrame, box, curColor, 1 + lineStyle);
		cv::putText(imageFrame, std::to_string(nID), cv::Point((int)box.x, (int)box.y + 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, curColor);
	}
}

}

//()()
//('')HAANJU.YOO

