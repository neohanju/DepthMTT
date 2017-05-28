#include "opencv2\highgui\highgui.hpp"
#include "haanju_string.hpp"
#include "haanju_fileIO.hpp"
#include "haanju_misc.hpp"
#include "FrameGrabber.h"
#include <iostream>


namespace hj
{

CFrameGrabber::CFrameGrabber()
	: bInit_(false)
	, bRealtimeOperation_(false)
	, bTerminate_(false)
	, nCamID_(0)
	, nCurrentFrameIndex_(0)
{
}


CFrameGrabber::~CFrameGrabber()
{
	Finalize();
}


bool CFrameGrabber::Initialize(int _nID, stParamFrameGrabber &_stParams)
{
	SetParameters(_stParams);

	if (bInit_)
	{
		Finalize();
	}
	nCamID_ = _nID;

	// for the approperiate operation, 'nCurrentFrameIndex' has to be subtraced by one
	nCurrentFrameIndex_ = stParams_.nStartFrameIndex - 1;

	/* set image foler path */
	strImageFoler_ = stParams_.strInputDir;

	if (hj::PILSNU == stParams_.nInputSource || hj::PETS == stParams_.nInputSource)
	{
		nInputType_ = HJ_READ_IMAGE;
	}
	else if (hj::KINECT == stParams_.nInputSource)
	{
		nInputType_ = HJ_READ_KINECT_DEPTH;		
	}
	else
	{
		nInputType_ = HJ_READ_IMAGE_TIME_FORMAT;
		filePos_ = -1;
		if (!hj::GetFileList(strImageFoler_, "frame_*.jpg", vecStrFileNames_))
		{
			hj::printf_debug("[ERROR] There is no matched file\n");
			return false;
		}
		std::sort(vecStrFileNames_.begin(), vecStrFileNames_.end());
		prevTime_ = hj::GetTimeFromFileName(vecStrFileNames_[0], "frame_", ".jpg");
	}	

	bRealtimeOperation_ = stParams_.bDoRealtimeOperation;
	bInit_ = true;
	bTerminate_ = false;
	
	return true;
}


bool CFrameGrabber::Finalize()
{
	if (!bInit_)
	{
		return true;
	}
	if (!matFrameBuffer_.empty()) { matFrameBuffer_.release(); }

	return true;
}

bool CFrameGrabber::Reset()
{
	if (bInit_)
	{
		Finalize();
	}
	Initialize(nCamID_, stParams_);

	return true;
}


bool CFrameGrabber::SetParameters(stParamFrameGrabber &_stParams)
{
	stParams_ = _stParams;
	return true;
}


HJ_GRAB_RESULT CFrameGrabber::GrabFrame()
{
	assert(bInit_);
	
	bool bGrabbed = false;
	HJ_GRAB_RESULT resultCode = HJ_GR_NORMAL;

	if (!matFrame_.empty())
	{
		matFramePrev_ = matFrame_.clone();
		matFrame_.release(); 
	}

	nCurrentFrameIndex_++;
	if (HJ_READ_IMAGE == nInputType_ || HJ_READ_KINECT_DEPTH == nInputType_)
	{
		bGrabbed = GrabImage(nCurrentFrameIndex_);
		if (!bGrabbed) { resultCode = HJ_GR_DATASET_ENDED; }
	}
	else if (HJ_READ_IMAGE_TIME_FORMAT == nInputType_)
	{
		long timeGap = 0;
		time_t timeCurrGrab = clock();
		if (nCurrentFrameIndex_ > stParams_.nStartFrameIndex)
		{
			timeGap = (long)(timeCurrGrab - timePrevGrab_);			
		}
		timePrevGrab_ = timeCurrGrab;

		//bGrabbed = GrabImage(timeGap);
		bGrabbed = GrabImage((long)10);
		if (!bGrabbed) { resultCode = HJ_GR_UNKNOWN_ERROR; }
	}
	else if (HJ_READ_VIDEO == nInputType_)
	{
		bGrabbed = GrabVideoFrame();
		if (!bGrabbed) { resultCode = HJ_GR_UNKNOWN_ERROR; }
	}
	
	if (matFrame_.empty() || !bGrabbed)
	{
		if (!matFramePrev_.empty())
		{
			matFrame_ = matFramePrev_.clone();
		}
	}

	return resultCode;
}


cv::Mat CFrameGrabber::GetFrame()
{
	assert(bInit_);
	return matFrame_;
}


bool CFrameGrabber::GrabImage(int _nFrameIndex)
{
	if (stParams_.bExistFrameRange && _nFrameIndex > stParams_.nEndFrameIndex)
	{
		return false;
	}

	char strImagePath[300];
	if (HJ_READ_KINECT_DEPTH == nInputType_)
	{
		sprintf_s(strImagePath, sizeof(strImagePath),
			"%s/%06d.png", strImageFoler_.c_str(), _nFrameIndex);
	}
	else
	{
		sprintf_s(strImagePath, sizeof(strImagePath),
			"%s/frame_%04d.jpg", strImageFoler_.c_str(), _nFrameIndex);
	}	
	matFrame_ = cv::imread(strImagePath, cv::IMREAD_COLOR);
	if (matFrame_.empty())
	{
		return false;
	}
	return true;
}


bool CFrameGrabber::GrabImage(long _nTimeGap)
{
	// convert _nTimeGap to formatted time gap
	long timeTemp = _nTimeGap;
	long miliseconds = 0, seconds = 0, minutes = 0, hours = 0;
	long fileTime = 0;

	if (0 <= filePos_ && 0 < _nTimeGap && bRealtimeOperation_)
	{
		timeTemp = _nTimeGap;
		// ms
		miliseconds = timeTemp % 1000;
		timeTemp = timeTemp / 1000;
		// sec
		seconds = timeTemp % 60;
		timeTemp = timeTemp / 60;
		// min
		minutes = timeTemp % 60;
		timeTemp = timeTemp / 60;
		// hours
		hours = timeTemp % 24;

		long formattedTimeGap = miliseconds;
		formattedTimeGap += seconds * 1000;
		formattedTimeGap += minutes * 100000;
		formattedTimeGap += hours * 10000000;

		// find nearest (right side) file
		long targetTime = prevTime_ + formattedTimeGap;
		bool bFound = false;
		for (filePos_++; filePos_ < (int)vecStrFileNames_.size(); filePos_++)
		{
			fileTime = hj::GetTimeFromFileName(vecStrFileNames_[filePos_], "frame_", ".jpg");
			if (fileTime > targetTime)
			{
				bFound = true;
				break;
			}
		}
		if (!bFound) { return false; }
	}
	else
	{
		filePos_++;
		if (filePos_ >= (int)vecStrFileNames_.size())
		{
			return false;
		}
		fileTime = hj::GetTimeFromFileName(vecStrFileNames_[filePos_], "frame_", ".jpg");
	}
	
	char strImagePath[300];
	sprintf_s(strImagePath, sizeof(strImagePath),
		"%s/%s", strImageFoler_.c_str(), vecStrFileNames_[filePos_].c_str());
	matFrame_ = cv::imread(strImagePath, cv::IMREAD_COLOR);
	if (matFrame_.empty())
	{
		return false;
	}
	prevTime_ = fileTime;
	return true;
}


bool CFrameGrabber::GrabVideoFrame()
{
	return true;
}


}



//()()
//('')HAANJU.YOO


