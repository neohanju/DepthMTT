/******************************************************************************
* Title        : CMainController
* Author       : Haanju Yoo
* Initial Date : 2016.10.03
* Version Num. : 0.9
* Description  : managing overall tracking procedure (in a single thread)
******************************************************************************/

#pragma once

#include <atomic>
#include <process.h>
#include "DataManager.h"
#include "FrameGrabber.h"
#include "SCMTTracker.h"
//#include "Associator3D.h"
#include "Evaluator.h"

typedef enum { GDT_THREAD = 0, A_THREAD, GUI_THREAD } HJ_THREAD_TYPE;

class CMainController
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	CMainController();
	~CMainController();

	bool Initialize(std::string _strParamXMLPath);
	void Finalize();
	void Reset();
	void Run();

	//bool WakeupGDTThreads();
	//bool WakeupAssociationThread(HJ_THREAD_TYPE _threadType, unsigned int _frameIdx);	
	bool RequestGrabbing();
	bool RequestDetection();
	bool RequestTracking();

private:
	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:

private:
	hj::CDataManager  cDataManager_;	
	hj::CFrameGrabber cFrameGrabber_;	
	hj::CSCMTTracker  cMultiTracker2D_;
	//hj::CAssociator3D cAssociator3D_;	
	hj::CEvaluator    cEvaluator_;

	volatile bool bInit_;
	volatile bool bSystemRun_;
	cv::Mat matInputFrames_;
	bool bDetect_;
	bool bEvaluate_;
	
	HANDLE hGrabberThread_;
	HANDLE hDetectorThread_;
	HANDLE hTrackerThread_;
//	HANDLE hAssoicationThread_;	
//	std::atomic<int> nNumGDTResults_;
	std::atomic<unsigned int> nFrameIdx_;
	SRWLOCK lockGrabber_, lockDetector, lockTracker, lockFrameIdx_;
};

//()()
//('')HAANJU.YOO

