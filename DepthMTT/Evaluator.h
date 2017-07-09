#pragma once

#include "types_3D.h"

namespace hj
{

struct stIDnRect
{
	int id;
	Rect rect;
};
typedef std::deque<stIDnRect> IDnRectSet;

struct stParamEvaluator
{
	std::string strGTPath;
	int nStartFrameIndex;
	int nEndFrameIndex;
	double dIOU = 0.5;
};

struct stGTMatchingResult
{
	int id;
	std::deque<std::pair<int, int>> queueMatchedResultID;  // frame index, matched TR id
};

struct stEvaluationResult
{
	double fMOTA;
	double fRecall;
	double fPrecision;
	int nMissed;
	int nFalsePositives;
	int nIDSwitch;
	int nMostTracked;
	int nPartilalyTracked;
	int nMostLost;
	int nFragments;
};

class CEvaluator
{
public:
	CEvaluator(void);
	~CEvaluator(void);

	void Initialize(stParamEvaluator _stParams);
	void Finalize(void);
	void InsertResult(hj::CTrackResult &trackResult);
	void Evaluate(void);
	
	stEvaluationResult GetEvaluationResult(void) { return stEvaluationResult_; }
	void PrintResultToConsole();
	void PrintResultToFile(const char *strFilepathAndName = NULL);
	std::string PrintResultToString();

private:
	bool bInit_;
	int nNumGTObjects_;
	stParamEvaluator stParams_;
	stEvaluationResult stEvaluationResult_;
	std::deque<std::pair<int, IDnRectSet>> queueGTs_;
	std::deque<std::pair<int, IDnRectSet>> queueTrackResults_;
	std::deque<stGTMatchingResult> queueGTtoTrackResult_;
};

}

//()()
//('')HAANJU.YOO


