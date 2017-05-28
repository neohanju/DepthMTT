#pragma once

#include <mutex>
#include <time.h>
#include "types.hpp"

//#define HJ_MATRIX_VISION  // when MATRIX VISION's camera is used to grab the frames
#ifdef HJ_MATRIX_VISION
#include <mvIMPACT_CPP\mvIMPACT_acquire.h>
#endif


enum HJ_FRAME_INPUT_TYPE { 
	HJ_READ_IMAGE = 0,         // from dataset
	HJ_READ_IMAGE_TIME_FORMAT, // from dataset (filename with grabbed time)
	HJ_READ_VIDEO,             // from dataset, but video
	HJ_READ_KINECT_DEPTH,      // from dataset, .png file containing 512 x 424
	NUM_HJ_INPUT_TYPE          // number of input type
};

enum HJ_GRAB_RESULT {
	HJ_GR_NORMAL = 0,
	HJ_GR_HARDWARE_FAIL,
	HJ_GR_DATASET_ENDED,
	HJ_GR_UNKNOWN_ERROR,
	NUM_HJ_GR
};

namespace hj
{

class CFrameGrabber
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	CFrameGrabber();	
	~CFrameGrabber();

	bool Initialize(int _nID, stParamFrameGrabber &_stParams);
	bool Finalize();
	bool Reset();
	bool SetParameters(stParamFrameGrabber &_stParams);
	HJ_GRAB_RESULT GrabFrame();
	int  GetFrameIndex() { return nCurrentFrameIndex_; }
	cv::Mat GetFrame();

private:
	bool GrabImage(int _nFrameIndex);
	bool GrabImage(long _nTimeGap);
	bool GrabNextImage();
	bool GrabVideoFrame();	

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
private:
	bool bInit_;
	bool bRealtimeOperation_;
	bool bTerminate_;
	int  nCamID_;
	int  nCurrentFrameIndex_;
	hj::stParamFrameGrabber stParams_;	
	HJ_FRAME_INPUT_TYPE nInputType_;

	/* image buffer */
	cv::Mat matFrameBuffer_;
	cv::Mat matFrame_;
	cv::Mat matFramePrev_;

	/* dataset related */
	std::string strImageFoler_;
	time_t      timePrevGrab_;
	int         filePos_;
	long        prevTime_;
	std::vector<std::string> vecStrFileNames_;	

	/* camera related */
	int nDeviceID_;
};

}

//()()
//('')HAANJU.YOO


