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

int main(int argc, char** argv)
{	
	//---------------------------------------------------
	// ARGUMENTS PARSING
	//---------------------------------------------------	
	int fIdxStart = 0, fIdxEnd = 100;
	if (argc < 4)
	{
		printf("[WARNING] Insufficient number of parameters\n");
		printf("- Dataset path\n");
		printf("- Index of starting frame\n");
		printf("- Index of ending frame\n");
		return -1;
	}

	// base path of the dataset
	std::string strDatasetPath = std::string(argv[1]);

	// starting frame index
	std::istringstream ssFrameIdxStart(argv[2]);
	if (!(ssFrameIdxStart >> fIdxStart))
	{
		std::cerr << "Invalid starting frame index: " << argv[2] << std::endl;
		return -1;
	}

	// ending frame index
	std::istringstream ssFrameIdxEnd(argv[3]);
	if (!(ssFrameIdxEnd >> fIdxEnd))
	{
		std::cerr << "Invalid ending frame index: " << argv[3] << std::endl;
		return -1;
	}
	

	//---------------------------------------------------
	// MAIN SECTION
	//---------------------------------------------------	
	std::string strFilePath;
	cv::Mat matCurFrame;
	hj::DetectionSet curDetections;
	hj::DepthMTTracker cTracker;

	// parameters
	hj::stParamTrack trackParams;
	trackParams.nImageWidth = 512;
	trackParams.nImageHeight = 424;
	trackParams.bVisualize = true;
	cTracker.Initialize(trackParams);

	for (int fIdx = fIdxStart; fIdx < fIdxEnd; fIdx++)
	{
		// get frame image
		strFilePath = strDatasetPath + "/" + hj::FormattedString("%06d.png", fIdx);
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
		cTracker.Track(curDetections, matCurFrame, fIdx);
	}

	return 0;
}

//()()
//('')HAANJU.YOO
