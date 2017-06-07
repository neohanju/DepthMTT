#include "MainController.h"
#include "haanju_fileIO.hpp"
#include "haanju_string.hpp"
#include "haanju_misc.hpp"

#define CAM_ID (0)

static volatile bool gGrabberRun;
static volatile bool gDetectorRun;
static volatile bool gTrackerRun;

//----------------------------------------------------------------
// GRABBER THREAD
//----------------------------------------------------------------
struct stGrabberThreadParams
{	
	int                 nMaxNumGrabFail;		
	CMainController*    pMainController;
	hj::CDataManager*   pDataManager;
	hj::CFrameGrabber*  pGrabber;
};
static stGrabberThreadParams gGrabberParams;

unsigned int __stdcall GrabberWork(void *data)
{
	stGrabberThreadParams *pParams = (stGrabberThreadParams*)data;	
	cv::Mat matFrame;
	int nFrameIndex;

	while (gGrabberRun && pParams->pDataManager->GetRunFlag())
	{
		// wait until the detector takes grabbed frame
		if (pParams->pDataManager->IsFrameBufferFull(CAM_ID)) 
		{ 
			Sleep(5); 
			continue;
		}

		// grabbing frame
		switch (pParams->pGrabber->GrabFrame())
		{
		case HJ_GR_NORMAL:
			matFrame = pParams->pGrabber->GetFrame();
			nFrameIndex = pParams->pGrabber->GetFrameIndex();
			hj::printf_debug("  Image is grabbed at frame %d\n", nFrameIndex);

			// save to buffer
			pParams->pDataManager->SetFrameImage(CAM_ID, matFrame, nFrameIndex);

			pParams->pMainController->RequestDetection();
			SuspendThread(GetCurrentThread());
			break;
		case HJ_GR_DATASET_ENDED:
			gGrabberRun = false;
			hj::printf_debug("  Reach the end of the dataset\n");		
			// terminate overall process
			pParams->pDataManager->SetRunFlag(false);
			pParams->pMainController->RequestDetection();
			pParams->pMainController->RequestTracking();
			break;
		default:
			hj::printf_debug("  Fail to grab a frame\n");			
			break;
		}		
	}

	hj::printf_debug("Grabber thread is terminated\n");
	pParams->pGrabber->Finalize();	

	return 0;
}


//----------------------------------------------------------------
// DETECTOR THREAD
//----------------------------------------------------------------
struct stDetectorThreadParams
{
	std::string         strDetectionDir;
	hj::DETECTION_TYPE  nDetectionType;
	//	CDetectorCrosstalk* pDetector;  // TODO: replace this with a real detector
	CMainController*    pMainController;
	hj::CDataManager*   pDataManager;
};
static stDetectorThreadParams gDetectorParams;

unsigned int __stdcall DetectorWork(void *data)
{
	stDetectorThreadParams *pParams = (stDetectorThreadParams*)data;
	cv::Mat matFrame;
	unsigned int nFrameIndex;
	hj::DetectionSet vecDetections;

	while (gDetectorRun && pParams->pDataManager->GetRunFlag())
	{
		// wait until the tracker takes detection result
		if (pParams->pDataManager->IsDetectionBufferFull(CAM_ID))
		{
			Sleep(5); 
			continue;
		}
		if (!pParams->pDataManager->IsFrameBufferFull(CAM_ID))
		{
			Sleep(5);
			continue;
		}

		pParams->pDataManager->GetFrameImage(CAM_ID, matFrame, false, &nFrameIndex);

		// virtual detector reading .txt files from a disk
		std::string strFilePath
			= pParams->strDetectionDir + "/" + hj::FormattedString("%06d.txt", nFrameIndex);
		vecDetections
			= hj::ReadDetectionResultWithTxt(
				strFilePath,
				pParams->nDetectionType);

		// get depth and patch
		cv::cvtColor(matFrame, matFrame, CV_BGR2GRAY);
		for (int detIdx = 0; detIdx < vecDetections.size(); detIdx++)
		{
			int x = (int)std::max(0.0, vecDetections[detIdx].box.x);
			int y = (int)std::max(0.0, vecDetections[detIdx].box.y);
			int w = (int)std::min((double)(matFrame.cols - x - 1), vecDetections[detIdx].box.w);
			int h = (int)std::min((double)(matFrame.rows - y - 1), vecDetections[detIdx].box.h);
			vecDetections[detIdx].patch = matFrame(cv::Rect(x, y, w, h)).clone();
		}

		hj::printf_debug("  Detections are generated at frame %d\n", nFrameIndex);
		pParams->pDataManager->SetDetectionResult(CAM_ID, vecDetections);

		pParams->pMainController->RequestTracking();
		SuspendThread(GetCurrentThread());
	}	

	hj::printf_debug("Detector thread is terminated\n");

	return 0;
}


//----------------------------------------------------------------
// TRACKER THREAD
//----------------------------------------------------------------
struct stTrackerThreadParams
{
	CMainController*    pMainController;
	hj::CDataManager*   pDataManager;
	hj::CSCMTTracker*   pTracker;
};
static stTrackerThreadParams gTrackerParams;

unsigned int __stdcall TrackerWork(void *data)
{
	stTrackerThreadParams *pParams = (stTrackerThreadParams*)data;
	cv::Mat matFrame;
	unsigned int nFrameIndex;
	hj::DetectionSet vecDetections;
	hj::CTrack2DResult cTrackResult;

	while (gTrackerRun && pParams->pDataManager->GetRunFlag())
	{
		if (!pParams->pDataManager->IsFrameBufferFull(CAM_ID))
		{
			Sleep(5); 
			continue;
		}
		if (!pParams->pDataManager->IsDetectionBufferFull(CAM_ID)) 
		{
			Sleep(5); 
			continue;
		}

		pParams->pDataManager->GetFrameImage(CAM_ID, matFrame, true, &nFrameIndex);
		pParams->pDataManager->GetDetectionResult(CAM_ID, &vecDetections);

		// do tracking
		cTrackResult = pParams->pTracker->Track(vecDetections, matFrame, nFrameIndex);
		hj::printf_debug("  Tracklets are generated at frame %d\n", nFrameIndex);

		pParams->pMainController->RequestGrabbing();
		SuspendThread(GetCurrentThread());
	}

	hj::printf_debug("Tracker thread is terminated\n");
	pParams->pTracker->Finalize();

	return 0;
}


////----------------------------------------------------------------
//// ASSOCIATION THREAD
////----------------------------------------------------------------
//struct stAssociationThreadParams
//{	
//	int nNumCams;	
//	CMainController*   pMainController;
//	hj::CDataManager*  pDataManager;
//	//hj::CAssociator3D* pAssociator;
//	hj::CEvaluator*    pEvaluator;
//};
//static stAssociationThreadParams gStAssociationThreadParam;
//
//unsigned int __stdcall AssociationWork(void *data)
//{
//	stAssociationThreadParams *pParams = (stAssociationThreadParams*)data;
//
//	std::vector<cv::Mat> vecMatFrames(pParams->nNumCams);
//	std::vector<hj::CTrack2DResult> vecTrack2DResults(pParams->nNumCams);
//	unsigned int nFrameIdx = 0;
//	unsigned int nFrameIdxRead = 0;
//	bool bEvaluate = pParams->pDataManager->GetEvaluateFlag();
//
//	hj::printf_debug("Thread for association is started\n");
//	while (gAssociationThreadRun && pParams->pDataManager->GetRunFlag())
//	{
//		for (int camIdx = 0; camIdx < pParams->nNumCams; camIdx++)
//		{
//			while (!pParams->pDataManager->IsFrameBufferFull(camIdx)
//				|| !pParams->pDataManager->IsTrack2DBufferFull(camIdx))
//			{
//				if (!pParams->pDataManager->GetRunFlag()) { break; }
//				hj::printf_debug("  >> sleep association thread\n");
//				::Sleep(5);
//			}
//			
//			/* load inputs */
//			pParams->pDataManager->GetFrameImage(camIdx, vecMatFrames[camIdx], &nFrameIdxRead);
//			pParams->pDataManager->GetTrack2DResult(camIdx, &vecTrack2DResults[camIdx]);
//			
//			// check frame indices syncronization
//			if (0 == camIdx)
//			{
//				nFrameIdx = nFrameIdxRead;
//			}
//			else if (nFrameIdx != nFrameIdxRead)
//			{
//				hj::printf_debug("  >> [ERROR] frame indices mismatch\n");
//				gAssociationThreadRun = false;
//				pParams->pMainController->TerminateProcess(-1); // assign temporary ID for 'A' thread
//				break;
//			}
//		}
//
//		//// do association
//		//hj::CTrack3DResult curResult = 
//		//	pParams->pAssociator->Run(vecTrack2DResults, vecMatFrames, nFrameIdx);
//
//		// DEBUG
//		if (14 == nFrameIdx)
//		{
//			int a = 0;
//		}
//
//		// evaluation
//		if (bEvaluate)
//		{
//			//pParams->pEvaluator->SetResult(curResult);
//		}
//
//		/* request new frame/tracklets */
//		pParams->pMainController->WakeupGDTThreads();
//		SuspendThread(GetCurrentThread());
//		hj::printf_debug("  >> suspend association thread\n");
//	}
//
//	// evaluate
//	if (bEvaluate)
//	{		
//		pParams->pEvaluator->Evaluate();
//		pParams->pEvaluator->PrintResultToConsole();
//		pParams->pEvaluator->PrintResultToFile();
//		pParams->pEvaluator->Finalize();
//	}
//
//	//pParams->pAssociator->Finalize();
//
//	hj::printf_debug("Thread for association is terminated\n");	
//
//	return 0;
//}


/////////////////////////////////////////////////////////////////////////
// CMainController MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
CMainController::CMainController()
	: bInit_(false)
	, bSystemRun_(false)	
	, bDetect_(false)
{
}


CMainController::~CMainController()
{
}


bool CMainController::Initialize(std::string _strParamXMLPath)
{
	if (bInit_) { return false; }

	/* read parameters */
	if (!cDataManager_.Initialize(_strParamXMLPath))
	{
		return false;
	}
	bDetect_   = cDataManager_.GetDetectFlag();	
	bEvaluate_ = cDataManager_.GetEvaluateFlag();	

	/* instantiate each modules */	
	cFrameGrabber_.Initialize(0, cDataManager_.GetFrameGrabberParams(CAM_ID));
	// TODO: detector
	cMultiTracker2D_.Initialize(0, cDataManager_.GetTrack2DParams(CAM_ID));
	//cAssociator3D_.Initialize(cDataManager_.GetAssociate3DParams());

	// evaluator
	if (bEvaluate_)
	{
		cEvaluator_.Initialize(cDataManager_.GetEvaluatorParams());
	}

	/* parameters */
	hj::stParamFrameGrabber grabberParams = cDataManager_.GetFrameGrabberParams(CAM_ID);
	gGrabberParams.nMaxNumGrabFail = hj::GRABBING == grabberParams.nInputSource ? 5 : 1;
	gGrabberParams.pGrabber        = &cFrameGrabber_;
	gGrabberParams.pMainController = this;	
	gGrabberParams.pDataManager    = &cDataManager_;	

	hj::stParamDetect2D detectorParams = cDataManager_.GetDetect2DParams(CAM_ID);
	gDetectorParams.nDetectionType  = detectorParams.nDetectionType;
	gDetectorParams.strDetectionDir = detectorParams.strDetectionDir;
	gDetectorParams.pMainController = this;	
	gDetectorParams.pDataManager    = &cDataManager_;

	hj::stParamTrack2D trackerParams = cDataManager_.GetTrack2DParams(CAM_ID);
	gTrackerParams.pTracker        = &cMultiTracker2D_;
	gTrackerParams.pMainController = this;
	gTrackerParams.pDataManager    = &cDataManager_;

	/* create thread */	
	gGrabberRun  = true;
	gDetectorRun = true;
	gTrackerRun  = true;
	hGrabberThread_  = (HANDLE)_beginthreadex(0, 0, &GrabberWork, &gGrabberParams, CREATE_SUSPENDED, 0);
	hDetectorThread_ = (HANDLE)_beginthreadex(0, 0, &DetectorWork, &gDetectorParams, CREATE_SUSPENDED, 0);
	hTrackerThread_  = (HANDLE)_beginthreadex(0, 0, &TrackerWork, &gTrackerParams, CREATE_SUSPENDED, 0);

	/* control flags */
	bInit_      = true;
	bSystemRun_ = true;
	cDataManager_.SetRunFlag(bSystemRun_);

	// locks
	InitializeSRWLock(&lockGrabber_);
	InitializeSRWLock(&lockDetector);
	InitializeSRWLock(&lockTracker);
	InitializeSRWLock(&lockFrameIdx_);

	return true;
}


void CMainController::Finalize()
{
	if (!bInit_) { return; }

	gGrabberRun  = false;
	WaitForSingleObject(hGrabberThread_, INFINITE);
	CloseHandle(hGrabberThread_);

	gDetectorRun = false;
	WaitForSingleObject(hDetectorThread_, INFINITE);
	CloseHandle(hDetectorThread_);

	gTrackerRun = false;
	WaitForSingleObject(hTrackerThread_, INFINITE);
	CloseHandle(hTrackerThread_);	
}

void CMainController::Reset()
{

}


void CMainController::Run()
{
	ResumeThread(hGrabberThread_);
	ResumeThread(hDetectorThread_);
	ResumeThread(hTrackerThread_);
	
	// wait until the tracker thread is terminated
	WaitForSingleObject(hTrackerThread_, INFINITE);
}


bool CMainController::RequestGrabbing()
{
	if (!gGrabberRun) { return false; }
	hj::printf_debug("  >> resume grabber thread\n");
	ResumeThread(hGrabberThread_);
	return true;
}


bool CMainController::RequestDetection()
{
	if (!gDetectorRun) { return false; }
	hj::printf_debug("  >> resume detector thread\n");
	ResumeThread(hDetectorThread_);
	return true;
}


bool CMainController::RequestTracking()
{
	if (!gTrackerRun) { return false; }
	hj::printf_debug("  >> resume tracker thread\n");
	ResumeThread(hTrackerThread_);
	return true;
}

//
//
//bool CMainController::WakeupGDTThreads()
//{
//	for (int camIdx = 0; camIdx < numCameras_; camIdx++)
//	{
//		if (!gArrGDTThreadRun[camIdx]) { return false; }
//		hj::printf_debug("  >> resume GDT thread no.%d\n", camIdx);
//		ResumeThread(vecHGDTThreads_[camIdx]);
//	}
//	return true;
//}
//
//
//bool CMainController::WakeupAssociationThread(HJ_THREAD_TYPE _threadType, unsigned int _frameIdx)
//{
//	bool bGDTFull = false;	
//	bool bTerminate = false;
//
//	//------------------------------------------		
//	AcquireSRWLockExclusive(&lockGDT_);		
//	//------------------------------------------
//	if (0 == nNumGDTResults_)
//	{
//		nFrameIdx_ = _frameIdx;
//	}
//	else if (nFrameIdx_ != _frameIdx)
//	{
//		hj::printf_debug("  >> [ERROR] Frame indices are not matched!\n");
//		bTerminate = true;
//	}
//	//------------------------------------------	
//	ReleaseSRWLockExclusive(&lockGDT_);
//	//------------------------------------------
//
//	if (bTerminate)
//	{
//		TerminateProcess(-2);  // assign temporary ID for main controller
//		hj::printf_debug("  >> resume association thread to terminate\n");
//		ResumeThread(hAssoicationThread_);
//		return true;
//	}
//
//	switch (_threadType)
//	{
//	case GDT_THREAD:
//		//------------------------------------------
//		AcquireSRWLockExclusive(&lockGDT_);
//		//------------------------------------------
//		nNumGDTResults_++;
//		if (nNumGDTResults_ >= numCameras_) { bGDTFull = true; }
//		//------------------------------------------
//		ReleaseSRWLockExclusive(&lockGDT_);
//		//------------------------------------------
//		break;
//	default:
//		break;
//	}
//	
//	if (bGDTFull)
//	{
//		//------------------------------------------
//		AcquireSRWLockExclusive(&lockGDT_);
//		//------------------------------------------
//		nNumGDTResults_ = 0;
//		//------------------------------------------
//		ReleaseSRWLockExclusive(&lockGDT_);
//		//------------------------------------------
//
//		hj::printf_debug("  >> resume association thread\n");
//		ResumeThread(hAssoicationThread_);
//		return true;
//	}
//	return false;
//}


//()()
//('')HAANJU.YOO

