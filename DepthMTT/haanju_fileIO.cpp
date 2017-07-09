#include <Windows.h>

#include "haanju_fileIO.hpp"


/************************************************************************
 Method Name: IsDirectoryExists
 Description:
	- Check wheter the directory exists or not
 Input Arguments:
	- dirName: directory path
 Return Values:
	- true: exists / false: non-exists
************************************************************************/
bool hj::CreateDirectoryForWindows(const std::string &dirName)
{
	std::wstring wideStrDirName = L"";
	wideStrDirName.assign(dirName.begin(), dirName.end());
	if (CreateDirectory(wideStrDirName.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError()) { return true; }
	return false;
}


/************************************************************************
 Method Name: GetFileList
 Description:
	- Get file name list with '_fileFormat' at '_dirPath'
 Input Arguments:
	- _dirPath   : directory path
	- _fileFormat: filename format used in 'dir' command
	- _outputVecFileNameList: name list of found files
 Return Values:
	- true: exists / false: non-exists
************************************************************************/
bool hj::GetFileList(const std::string _dirPath, const std::string _fileFormat, std::vector<std::string> &_outputVecFileNameList)
{
	HANDLE dir;
	WIN32_FIND_DATA fileData;
	std::string strDirFormat_ = _dirPath + "/" + _fileFormat;
#ifdef UNICODE
	std::wstring strDirFormat = L"";
	strDirFormat.assign(strDirFormat_.begin(), strDirFormat_.end());
#else
	std::string strDirFormat(strDirFormat_);
#endif

	// check existance
	if (INVALID_HANDLE_VALUE == (dir = FindFirstFile(strDirFormat.c_str(), &fileData)))
	{
		return false;
	}

	// read files
	_outputVecFileNameList.clear();
	do
	{
#ifdef UNICODE
		std::wstring strFileName_w(fileData.cFileName);
		std::string  strFileName = "";
		strFileName.assign(strFileName_w.begin(), strFileName_w.end());
#else
		std::string strFileName(fileData.cFileName);
#endif
		_outputVecFileNameList.push_back(strFileName);
	} while (0 != FindNextFile(dir, &fileData) || ERROR_NO_MORE_FILES != GetLastError());

	return true;
}


/************************************************************************
 Method Name: printLog
 Description:
	- print out log file
 Input Arguments:
	- filename: file path
	- strLog  : log string
 Return Values:
	- none
************************************************************************/
void hj::printLog(const char *filename, std::string strLog)
{
	try
	{
		FILE *fp;
		fopen_s(&fp, filename, "a");
		fprintf(fp, strLog.c_str());
		fclose(fp);
	}
	catch (DWORD dwError)
	{
		printf("[ERROR] cannot open logging file! error code %d\n", dwError);
		return;
	}
}


/************************************************************************
 Method Name: ReadDetectionResultWithTxt
 Description:
	-
 Input Arguments:
	-
 Return Values:
	- Track3D*:
************************************************************************/
std::vector<hj::CDetection> hj::ReadDetectionResultWithTxt(std::string _strFilePath, DETECTION_TYPE _detectionType)
{
	std::vector<hj::CDetection> vec_result;
	int num_detection = 0;
	float x, y, w, h, depth, id;

	FILE *fid;
	try {		
		fopen_s(&fid, _strFilePath.c_str(), "r");
		if (NULL == fid) { return vec_result; }

		switch (_detectionType)
		{
		case hj::DEPTH_HEAD:
		default:
			// read # of detections
			fscanf_s(fid, "%d\n", &num_detection);
			vec_result.reserve(num_detection);

			// read box infos
			for (int detect_idx = 0; detect_idx < num_detection; detect_idx++)
			{
				fscanf_s(fid, "%f %f %f %f %f %f\n", &id, &depth, &x, &y, &w, &h);
				CDetection cur_detection;
				cur_detection.id = (int)id;
				cur_detection.box = Rect((double)x, (double)y, (double)w, (double)h);				
				vec_result.push_back(cur_detection);
			}
			break;
		}
		fclose(fid);
	}
	catch (DWORD dwError) {
		printf("[ERROR] file open error with detection result reading: %d\n", dwError);
	}
	return vec_result;
}


//()()
//('')HAANJU.YOO


