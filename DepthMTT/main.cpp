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
#include "types.hpp"
#include "DepthMTTracker.h"
#include "haanju_string.hpp"
#include "haanju_fileIO.hpp"
#include "Evaluator.h"

#define MTT_S_05
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
	std::string strFilePath;
	cv::Mat matCurFrame;
	hj::DetectionSet curDetections;
	hj::CTrackResult trackResult;

	// init tracker
	hj::stParamTrack trackParams;
	trackParams.nImageWidth = 512;
	trackParams.nImageHeight = 424;
	trackParams.bVisualize = true;
	trackParams.bVideoRecord = false;
	trackParams.strVideoRecordPath = strDatasetPath;
	hj::DepthMTTracker cTracker;
	cTracker.Initialize(trackParams);

	// init evaluator
	hj::stParamEvaluator evalParams;
	evalParams.strGTPath = strDatasetPath;
	evalParams.nStartFrameIndex = START_FRAME_INDEX;
	evalParams.nEndFrameIndex = END_FRAME_INDEX;
	evalParams.dIOU = 0.5;
	hj::CEvaluator evaluator;
	evaluator.Initialize(evalParams);

	for (int fIdx = START_FRAME_INDEX; fIdx <= END_FRAME_INDEX; fIdx++)
	{
		// get frame image
		strFilePath = strDatasetPath + "\\" + hj::FormattedString("%06d.png", fIdx);
		matCurFrame = cv::imread(strFilePath, cv::IMREAD_GRAYSCALE);
		if (matCurFrame.empty())
		{
			std::cerr << "No such a file " << strFilePath << std::endl;
			return -1;
		}

		// get detections
		strFilePath = strDatasetPath + "/" + hj::FormattedString("%06d.txt", fIdx);
		curDetections = hj::ReadDetectionResultWithTxt(strFilePath, hj::DEPTH_HEAD);

		// track
		trackResult = cTracker.Track(curDetections, matCurFrame, fIdx);
		evaluator.InsertResult(trackResult);
	}

	evaluator.Evaluate();
	evaluator.PrintResultToConsole();
	evaluator.PrintResultToFile();

	return 0;
}

//()()
//('')HAANJU.YOO
