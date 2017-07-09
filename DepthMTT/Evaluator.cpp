#include "Evaluator.h"
#include <numeric>
#include <limits>
#include "haanju_fileIO.hpp"
#include "haanju_string.hpp"
#include "HungarianMethod.h"


namespace hj
{

CEvaluator::CEvaluator(void)
	: bInit_(false)
{
}


CEvaluator::~CEvaluator(void)
{
}


void CEvaluator::Initialize(stParamEvaluator _stParams)
{
	if (bInit_) { return; }
	stParams_ = _stParams;
	int numFrames = stParams_.nEndFrameIndex - stParams_.nStartFrameIndex + 1;

	// measures	
	stEvaluationResult_.fMOTA = 0.0;;
	stEvaluationResult_.fRecall = 0.0;
	stEvaluationResult_.fPrecision = 0.0;
	stEvaluationResult_.nMissed = 0;
	stEvaluationResult_.nFalsePositives = 0;
	stEvaluationResult_.nIDSwitch = 0;
	stEvaluationResult_.nMostTracked = 0;
	stEvaluationResult_.nPartilalyTracked = 0;
	stEvaluationResult_.nMostLost = 0;
	stEvaluationResult_.nFragments = 0;

	// read GTs
	nNumGTObjects_ = 0;
	queueGTs_.resize(numFrames);
	for (int fIdx = stParams_.nStartFrameIndex, queuePos = 0; fIdx <= stParams_.nEndFrameIndex; fIdx++, queuePos++)
	{
		std::string strFilePath = stParams_.strGTPath + "/" + hj::FormattedString("%06d.txt", fIdx);
		std::vector<hj::CDetection> curGTObjects = hj::ReadDetectionResultWithTxt(strFilePath);
		queueGTs_[queuePos].first = fIdx;
		for (int objIdx = 0; objIdx < curGTObjects.size(); objIdx++)
		{
			// save ground truth
			stIDnRect IDnRect = { curGTObjects[objIdx].id , curGTObjects[objIdx].box };
			queueGTs_[queuePos].second.push_back(IDnRect);

			// count the number of GT objects
			if (curGTObjects[objIdx].id + 1 > nNumGTObjects_)
				nNumGTObjects_ = curGTObjects[objIdx].id + 1;
		}
	}
	bInit_ = true;
}


void CEvaluator::Finalize(void)
{
	if (!bInit_) { return; }
	queueGTs_.clear();
	queueTrackResults_.clear();
	bInit_ = false;
}


void CEvaluator::InsertResult(hj::CTrackResult &_trackResult)
{
	assert(stParams_.nStartFrameIndex <= (int)_trackResult.frameIdx
		&& stParams_.nEndFrameIndex >= (int)_trackResult.frameIdx);

	// matching between GTs and TRs (track results) with IOU
	std::pair<int, IDnRectSet> frameResult;
	frameResult.first = (int)_trackResult.frameIdx;
	frameResult.second.resize(_trackResult.objectInfos.size());	
	for (int i = 0; i < frameResult.second.size(); i++)
	{
		frameResult.second[i].id = _trackResult.objectInfos[i].id;
		frameResult.second[i].rect = _trackResult.objectInfos[i].box;
	}
	queueTrackResults_.push_back(frameResult);
}


void CEvaluator::Evaluate(void)
{
	stEvaluationResult curResult;
	curResult.fMOTA = 0.0;
	curResult.fRecall = 0.0;
	curResult.fPrecision = 0.0;
	curResult.nFalsePositives = 0;
	curResult.nMissed = 0;
	curResult.nMostTracked = 0;
	curResult.nPartilalyTracked = 0;
	curResult.nMostLost = 0;
	curResult.nIDSwitch = 0;
	curResult.nFragments = 0;


	//---------------------------------------------------
	// GT <-> TR MATCHING
	//---------------------------------------------------
	int trackResultPos = 0;
	queueGTtoTrackResult_.resize(nNumGTObjects_);
	for (int i = 0; i < queueGTs_.size(); i++)
	{
		// preparing the matching log
		int frameIdx = queueGTs_[i].first;
		for (int j = 0; j < queueGTs_[i].second.size(); j++)
		{
			int id = queueGTs_[i].second[j].id;
			queueGTtoTrackResult_[id].id = id;
			queueGTtoTrackResult_[id].queueMatchedResultID.push_back(std::make_pair(frameIdx, -1));
		}

		// find concurrent tracking result
		for (; trackResultPos < queueTrackResults_.size(); trackResultPos++)
		{
			if (queueTrackResults_[trackResultPos].first >= frameIdx)
				break;
			else if (queueTrackResults_[trackResultPos].first < frameIdx)
				continue;
		}
		if (queueTrackResults_[trackResultPos].first != frameIdx)
			continue;

		// matching
		std::vector<float> arrMatchingCost_;
		arrMatchingCost_.resize(
			queueTrackResults_[trackResultPos].second.size() * queueGTs_[i].second.size(),
			std::numeric_limits<float>::infinity());

		for (int gtIdx = 0; gtIdx < queueGTs_[i].second.size(); gtIdx++)
		{
			hj::Rect gtBox = queueGTs_[i].second[gtIdx].rect;
			for (int trackIdx = 0, costPos = gtIdx;
				trackIdx < queueTrackResults_[trackResultPos].second.size(); 
				trackIdx++, costPos += (int)queueGTs_[i].second.size())
			{
				hj::Rect trackBox = queueTrackResults_[trackResultPos].second[trackIdx].rect;
				if (!trackBox.overlap(gtBox))
					continue;
				
				// get intersection over union
				double iou = trackBox.overlappedArea(gtBox) / (trackBox.area() + gtBox.area() - trackBox.overlappedArea(gtBox));
				if (iou < stParams_.dIOU)
					continue;

				arrMatchingCost_[costPos] = 1.0f / (float)iou;
			}
		}

		// To ensure a proper operation of our Hungarian implementation, we convert infinite to the finite value
		// that is little bit (=100.0f) greater than the maximum finite cost in the original cost function.
		float maxCost = -1000.0f;
		for (int costIdx = 0; costIdx < arrMatchingCost_.size(); costIdx++)
		{
			if (!_finitef(arrMatchingCost_[costIdx])) { continue; }
			if (maxCost < arrMatchingCost_[costIdx]) { maxCost = arrMatchingCost_[costIdx]; }
		}
		maxCost = maxCost + 100.0f;
		for (int costIdx = 0; costIdx < arrMatchingCost_.size(); costIdx++)
		{
			if (_finitef(arrMatchingCost_[costIdx])) { continue; }
			arrMatchingCost_[costIdx] = maxCost;
		}

		// matching
		CHungarianMethod cHungarianMatcher;
		cHungarianMatcher.Initialize(arrMatchingCost_, 
			(unsigned int)queueTrackResults_[trackResultPos].second.size(), 
			(unsigned int)queueGTs_[i].second.size());
		stMatchInfo *curMatchInfo = cHungarianMatcher.Match();
		int numMatches = 0;
		for (size_t matchIdx = 0; matchIdx < curMatchInfo->rows.size(); matchIdx++)
		{
			if (maxCost == curMatchInfo->matchCosts[matchIdx]) { continue; }
			int trackID = queueTrackResults_[trackResultPos].second[curMatchInfo->rows[matchIdx]].id;
			int gtID = queueGTs_[i].second[curMatchInfo->cols[matchIdx]].id;
			queueGTtoTrackResult_[gtID].queueMatchedResultID.back().second = trackID;
			numMatches++;
		}
		cHungarianMatcher.Finalize();
		curResult.nFalsePositives += (int)queueTrackResults_[trackResultPos].second.size() - numMatches;
	}

	//---------------------------------------------------------
	// EVALUATING (porting from CLEAR_MOT.m)
	//---------------------------------------------------------
	int numGTs = 0;
	for (int i = 0; i < queueGTtoTrackResult_.size(); i++)
	{
		int numMissed = 0;
		int numIDSwitch = 0;
		int numFragment = 0;
		int prevID = -1;
		int prevTrackResultID = -1;
		double trackedPortion = 0.0;
		for (int j = 0; j < queueGTtoTrackResult_[i].queueMatchedResultID.size(); j++)
		{
			int trackResultID = queueGTtoTrackResult_[i].queueMatchedResultID[j].second;
			if (trackResultID < 0)
			{
				numMissed++;
				if (prevTrackResultID >= 0)
				{
					numFragment++;
				}
			}
			else if (prevID < 0)
			{
				// start interval
				prevID = trackResultID;
			}
			else if (prevID != trackResultID)
			{
				numIDSwitch++;
				prevID = trackResultID;
			}
			prevTrackResultID = trackResultID;
		}
		curResult.nMissed += numMissed;  // count missing here, because the tracking result would not exist in certain time
		curResult.nIDSwitch += numIDSwitch;
		curResult.nFragments += numFragment;

		// MT PT ML
		trackedPortion = (double)(queueGTtoTrackResult_[i].queueMatchedResultID.size() - (size_t)numMissed) / queueGTtoTrackResult_[i].queueMatchedResultID.size();
		if (trackedPortion >= 0.8)
			curResult.nMostTracked++;
		else if (trackedPortion >= 0.2)
			curResult.nPartilalyTracked++;
		else
			curResult.nMostLost++;

		numGTs += (int)queueGTtoTrackResult_[i].queueMatchedResultID.size();
	}
	double truePositive = (double)(numGTs - curResult.nMissed);
	curResult.fMOTA = 1.0 - (double)(curResult.nMissed + curResult.nFalsePositives + curResult.nIDSwitch) / numGTs;
	curResult.fRecall = truePositive / numGTs;
	curResult.fPrecision = truePositive / (truePositive + (double)curResult.nFalsePositives);

	stEvaluationResult_ = curResult;
}


void CEvaluator::PrintResultToConsole()
{
	printf(PrintResultToString().c_str());
}


void CEvaluator::PrintResultToFile(const char *strFilepathAndName)
{
	FILE *fp;
	try
	{
		if (NULL == strFilepathAndName)
		{
			char strFilePath[128] = "";
			sprintf_s(strFilePath, "%s/evaluate.txt", stParams_.strGTPath.c_str());
			fopen_s(&fp, strFilePath, "w");
		}
		else
		{
			fopen_s(&fp, strFilepathAndName, "w");
		}
		fprintf_s(fp, PrintResultToString().c_str());
		fclose(fp);
	}
	catch (long dwError)
	{
		printf("[ERROR](PrintResultToFile) cannot open file! error code %d\n", dwError);
		return;
	}
}


std::string CEvaluator::PrintResultToString()
{
	char strResult[700] = "";
	sprintf_s(strResult, "Evaluating dataset on ground plane...\n");
	sprintf_s(strResult, "%s| Recl Prcn| MT PT ML|  FP  FN  ID  FM  err| MOTA|\n", strResult);
	sprintf_s(strResult, "%s|%5.1f%5.1f|%3i%3i%3i|%4i%4i%4i%4i%5i|%5.1f|\n", strResult,
		stEvaluationResult_.fRecall * 100,
		stEvaluationResult_.fPrecision * 100,
		stEvaluationResult_.nMostTracked,
		stEvaluationResult_.nPartilalyTracked,
		stEvaluationResult_.nMostLost,
		stEvaluationResult_.nFalsePositives,
		stEvaluationResult_.nMissed,
		stEvaluationResult_.nIDSwitch,
		stEvaluationResult_.nFragments,
		stEvaluationResult_.nMissed + stEvaluationResult_.nFalsePositives + stEvaluationResult_.nIDSwitch,
		stEvaluationResult_.fMOTA * 100);

	return std::string(strResult);
}

}

//()()
//('')HAANJU.YOO


