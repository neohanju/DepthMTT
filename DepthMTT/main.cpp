/**************************************************************************
* Title        : Online Multi-Camera Multi-Target Tracking Algorithm
* Author       : Haanju Yoo
* Initial Date : 2013.08.29 (ver. 0.9)
* Version Num. : 1.0 (since 2016.09.06)
* Description  :
*	The implementation of the paper named "Online Scheme for Multiple
*	Camera Multiple Target Tracking Based on Multiple Hypothesis 
*	Tracking" at IEEE transactions on Circuit and Systems for Video 
*	Technology (TCSVT).
***************************************************************************
                                            ....
                                           W$$$$$u
                                           $$$$F**+           .oW$$$eu
                                           ..ueeeWeeo..      e$$$$$$$$$
                                       .eW$$$$$$$$$$$$$$$b- d$$$$$$$$$$W
                           ,,,,,,,uee$$$$$$$$$$$$$$$$$$$$$ H$$$$$$$$$$$~
                        :eoC$$$$$$$$$$$C""?$$$$$$$$$$$$$$$ T$$$$$$$$$$"
                         $$$*$$$$$$$$$$$$$e "$$$$$$$$$$$$$$i$$$$$$$$F"
                         ?f"!?$$$$$$$$$$$$$$ud$$$$$$$$$$$$$$$$$$$$*Co
                         $   o$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                 !!!!m.*eeeW$$$$$$$$$$$f?$$$$$$$$$$$$$$$$$$$$$$$$$$$$$U
                 !!!!!! !$$$$$$$$$$$$$$  T$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                  *!!*.o$$$$$$$$$$$$$$$e,d$$$$$$$$$$$$$$$$$$$$$$$$$$$$$:
                 "eee$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$C
                b ?$$$$$$$$$$$$$$**$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$!
                Tb "$$$$$$$$$$$$$$*uL"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
                 $$o."?$$$$$$$$F" u$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                  $$$$en '''    .e$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
                   $$$B*  =*"?.e$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$F
                    $$$W"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                     "$$$o#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                    R: ?$$$W$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" :!i.
                     !!n.?$???""''.......,''''''"""""""""""''   ...+!!!
                      !* ,+::!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*'
                      "!?!!!!!!!!!!!!!!!!!!~ !!!!!!!!!!!!!!!!!!!~'
                      +!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!?!'
                    .!!!!!!!!!!!!!!!!!!!!!' !!!!!!!!!!!!!!!, !!!!
                   :!!!!!!!!!!!!!!!!!!!!!!' !!!!!!!!!!!!!!!!! '!!:
                .+!!!!!!!!!!!!!!!!!!!!!~~!! !!!!!!!!!!!!!!!!!! !!!.
               :!!!!!!!!!!!!!!!!!!!!!!!!!.':!!!!!!!!!!!!!!!!!:: '!!+
               "~!!!!!!!!!!!!!!!!!!!!!!!!!!.~!!!!!!!!!!!!!!!!!!!!.'!!:
                   ~~!!!!!!!!!!!!!!!!!!!!!!! ;!!!!~' ..eeeeeeo.'+!.!!!!.
                 :..    '+~!!!!!!!!!!!!!!!!! :!;'.e$$$$$$$$$$$$$u .
                 $$$$$$beeeu..  '''''~+~~~~~" ' !$$$$$$$$$$$$$$$$ $b
                 $$$$$$$$$$$$$$$$$$$$$UU$U$$$$$ ~$$$$$$$$$$$$$$$$ $$o
                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$. $$$$$$$$$$$$$$$~ $$$u
                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$! $$$$$$$$$$$$$$$ 8$$$$.
                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$X $$$$$$$$$$$$$$'u$$$$$W
                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$! $$$$$$$$$$$$$".$$$$$$$:
                 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  $$$$$$$$$$$$F.$$$$$$$$$
                 ?$$$$$$$$$$$$$$$$$$$$$$$$$$$$f $$$$$$$$$$$$' $$$$$$$$$$.
                  $$$$$$$$$$$$$$$$$$$$$$$$$$$$ $$$$$$$$$$$$$  $$$$$$$$$$!
                  "$$$$$$$$$$$$$$$$$$$$$$$$$$$ ?$$$$$$$$$$$$  $$$$$$$$$$!
                   "$$$$$$$$$$$$$$$$$$$$$$$$Fib ?$$$$$$$$$$$b ?$$$$$$$$$
                     "$$$$$$$$$$$$$$$$$$$$"o$$$b."$$$$$$$$$$$  $$$$$$$$'
                    e. ?$$$$$$$$$$$$$$$$$ d$$$$$$o."?$$$$$$$$H $$$$$$$'
                   $$$W.'?$$$$$$$$$$$$$$$ $$$$$$$$$e. "??$$$f .$$$$$$'
                  d$$$$$$o "?$$$$$$$$$$$$ $$$$$$$$$$$$$eeeeee$$$$$$$"
                  $$$$$$$$$bu "?$$$$$$$$$ 3$$$$$$$$$$$$$$$$$$$$*$$"
                 d$$$$$$$$$$$$$e. "?$$$$$:'$$$$$$$$$$$$$$$$$$$$8
         e$$e.   $$$$$$$$$$$$$$$$$$+  "??f "$$$$$$$$$$$$$$$$$$$$c
        $$$$$$$o $$$$$$$$$$$$$$$F"          '$$$$$$$$$$$$$$$$$$$$b.0
       M$$$$$$$$U$$$$$$$$$$$$$F"              ?$$$$$$$$$$$$$$$$$$$$$u
       ?$$$$$$$$$$$$$$$$$$$$F                   "?$$$$$$$$$$$$$$$$$$$$u
        "$$$$$$$$$$$$$$$$$$"                       ?$$$$$$$$$$$$$$$$$$$$o
          "?$$$$$$$$$$$$$F                            "?$$$$$$$$$$$$$$$$$$
             "??$$$$$$$F                                 ""?3$$$$$$$$$$$$F
                                                       .e$$$$$$$$$$$$$$$$'
                                                      u$$$$$$$$$$$$$$$$$
                                                     '$$$$$$$$$$$$$$$$"
                                                      "$$$$$$$$$$$$F"
                                                        ""?????""

**************************************************************************/

#include <sstream>
#include <iostream>
#include "opencv2\highgui\highgui.hpp"
#include "haanju_utils.hpp"
#include "Evaluator.h"


#define MTT_S_04  // <- modify this to select the test sequences

#ifdef MTT_S_01
	#define DATASET_PATH (".\\data\\MTT_S_01")
	#define START_FRAME_INDEX (0)
	#define END_FRAME_INDEX (407)
#endif
#ifdef MTT_S_02
	#define DATASET_PATH (".\\data\\MTT_S_02")
	#define START_FRAME_INDEX (0)
	#define END_FRAME_INDEX (431)
#endif
#ifdef MTT_S_03
	#define DATASET_PATH (".\\data\\MTT_S_03")
	#define START_FRAME_INDEX (0)
	#define END_FRAME_INDEX (438)
#endif
#ifdef MTT_S_04
	#define DATASET_PATH (".\\data\\MTT_S_04")
	#define START_FRAME_INDEX (0)
	#define END_FRAME_INDEX (693)
#endif
#ifdef MTT_S_05
#define DATASET_PATH (".\\data\\MTT_S_05")
#define START_FRAME_INDEX (0)
#define END_FRAME_INDEX (235)
#endif


int main(int argc, char** argv)
{	
	std::string strDatasetPath = std::string(DATASET_PATH);
	std::string strFilePath;  // <- temporary file path for this and that
	cv::Mat matCurFrame;
	hj::DetectionSet curDetections;	


	//---------------------------------------------------
	// TRACKER INITIATION
	//---------------------------------------------------
	hj::CTrackResult trackResult;     // <- The tracking result will be saved here
	hj::stParamTrack trackParams;     // <- Contains whole parameters of tracking module. Using default values is recommended.
	trackParams.nImageWidth = 512;
	trackParams.nImageHeight = 424;
	trackParams.dImageRescale = 0.5;  // <- Heavy influence on the speed of the algorithm.
	trackParams.bVisualize = true;
	trackParams.bVideoRecord = true;  // <- To recoder the result visualization.
	trackParams.strVideoRecordPath = strDatasetPath;
	hj::CDepthMTTracker cTracker;      // <- The instance of a multi-target tracker.
	cTracker.Initialize(trackParams);


	//---------------------------------------------------
	// EVALUATION MODULE INITIATION
	//---------------------------------------------------
	hj::stParamEvaluator evalParams;
	evalParams.strGTPath = strDatasetPath;
	evalParams.nStartFrameIndex = START_FRAME_INDEX;
	evalParams.nEndFrameIndex = END_FRAME_INDEX;
	evalParams.dIOU = 0.5;
	hj::CEvaluator evaluator;
	evaluator.Initialize(evalParams);
	std::string strEvalFilePath = evalParams.strGTPath + "_evaluation_result.txt";


	//---------------------------------------------------
	// PROC. TIME LOGGING
	//---------------------------------------------------
	FILE *fp;
	std::string strProcTimeFilePath = strDatasetPath + "_procTime.csv";
	fopen_s(&fp, strProcTimeFilePath.c_str(), "w");

	
	//---------------------------------------------------
	// MAIN LOOP FOR TRACKING
	//---------------------------------------------------
	for (int fIdx = START_FRAME_INDEX; fIdx <= END_FRAME_INDEX; fIdx++)
	{
		// Grab frame image
		strFilePath = strDatasetPath + "\\" + hj::FormattedString("%06d.png", fIdx);
		matCurFrame = cv::imread(strFilePath, cv::IMREAD_GRAYSCALE);
		if (matCurFrame.empty())
		{
			std::cerr << "No such a file " << strFilePath << std::endl;
			fclose(fp);
			return -1;
		}

		// Read detections
		strFilePath = strDatasetPath + "/" + hj::FormattedString("%06d.txt", fIdx);
		curDetections = hj::CDepthMTTracker::ReadDetectionResultWithTxt(strFilePath);

		// Track targets between consecutive frames
		trackResult = cTracker.Track(curDetections, matCurFrame, fIdx);
		evaluator.InsertResult(trackResult);
		fprintf_s(fp, "%d,%lld\n", fIdx, (long long)trackResult.procTime);
	}
	fclose(fp);  // proc. time logging


	//---------------------------------------------------
	// EVALUATION
	//---------------------------------------------------
	evaluator.Evaluate();
	evaluator.PrintResultToConsole();
	evaluator.PrintResultToFile(strEvalFilePath.c_str());


	return 0;
}


//()()
//('')HAANJU.YOO
