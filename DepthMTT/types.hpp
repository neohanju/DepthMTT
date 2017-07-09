/******************************************************************************
* Title        : types
* Author       : Haanju Yoo
* Initial Date : 2013.08.29 (ver. 0.9)
* Version Num. : 1.0 (since 2016.09.16)
* Description  :
*	Including basic types for overall tracking operation. Types, which are only
*	used in 3D association, are in an additional file named 'types_3D.hpp'.
******************************************************************************
*               .__                           __.
*                \ '\~~---..---~~~~~~--.---~~| /
*                 '~-.   '                   .~         _____
*                     ~.                .--~~    .---~~~    /
*                      / .-.      .-.      |  <~~        __/
*                     |  |_|      |_|       \  \     .--'
*                    /-.      -       .-.    |  \_   \_
*                    \-'   -..-..-    '-'    |    \__  \_
*                     '.                     |     _/  _/
*                     ~-                .,-\   _/  _/
*                      /                 -~~~~\ /_  /_
*                     |               /   |    \  \_  \_
*                     |   /          /   /      | _/  _/
*                     |  |          |   /    .,-|/  _/
*                     )__/           \_/    -~~~| _/
*                       \                      /  \
*                        |           |        /_---'
*                        \    .______|      ./
*                        (   /        \    /
*                        '--'          /__/
*
******************************************************************************/

#ifndef __HAANJU_TYPES_HPP__
#define __HAANJU_TYPES_HPP__

#define NOMINMAX

#include <list>
#include <deque>
#include <time.h>
#include <opencv2\imgproc\imgproc.hpp>

//#include "cameraModel.h"

#define MAX_NUM_SAME_THREAD (10)
#define HJ_LIDAR_RESOLUTION (360*2)

namespace hj
{

enum DETECTION_TYPE { FULLBODY = 0, HEAD, PARTS, DEPTH_HEAD };
enum INPUT_SOURCE   { GRABBING = 0, PILSNU, DKU, PETS, KINECT };

/////////////////////////////////////////////////////////////////////////
// GEOMETRY TYPES
/////////////////////////////////////////////////////////////////////////
class Point2D
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	// constructors
	Point2D() : x(0), y(0) {}
	Point2D(double x, double y) : x(x), y(y) {}
	Point2D(cv::Point2f a) : x((double)a.x), y((double)a.y) {}

	// operators
	Point2D& operator=(const Point2D &a)     { x = a.x; y = a.y; return *this; }
	Point2D& operator=(const cv::Point &a)   { x = a.x; y = a.y; return *this; }
	Point2D& operator=(const cv::Point2f &a) { x = (double)a.x; y = (double)a.y; return *this; }
	Point2D& operator+=(const Point2D &a)    { x = x + a.x; y = y + a.y; return *this; }
	Point2D& operator-=(const Point2D &a)    { x = x - a.x; y = y - a.y; return *this; }
	Point2D& operator+=(const double s)      { x = x + s; y = y + s; return *this; }
	Point2D& operator-=(const double s)      { x = x - s; y = y - s; return *this; }
	Point2D& operator*=(const double s)      { x = x * s; y = y * s; return *this; }
	Point2D& operator/=(const double s)      { x = x / s; y = y / s; return *this; }
	Point2D  operator+(const Point2D &a)     { return Point2D(x + a.x, y + a.y); }
	Point2D  operator+(const double s)       { return Point2D(x + s, y + s); }
	Point2D  operator-(const Point2D &a)     { return Point2D(x - a.x, y - a.y); }
	Point2D  operator-(const double s)       { return Point2D(x - s, y - s); }
	Point2D  operator-()                     { return Point2D(-x, -y); }
	Point2D  operator*(const double s)       { return Point2D(x * s, y * s); }
	Point2D  operator/(const double s)       { return Point2D(x / s, y / s); }
	bool     operator==(const Point2D &a)    { return (x == a.x && y == a.y); }
	bool     operator==(const cv::Point &a)  { return (x == a.x && y == a.y); }

	// methods
	double  norm_L2()             { return std::sqrt(x * x + y * y); }
	double  dot(const Point2D &a) { return x * a.x + y * a.y; }
	Point2D scale(double scale)   { return Point2D(x * scale, y * scale); }
	bool    onView(const unsigned int width, const unsigned int height)
	{
		if (x < 0) { return false; }
		if (x >= (double)width) { return false; }
		if (y < 0) { return false; }
		if (y >= (double)height) { return false; }
		return true;
	}	

	// data conversion
	cv::Point cv() { return cv::Point((int)x, (int)y); }

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	// data
	double x;
	double y;
};


class Point3D
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	// constructors
	Point3D() : x(0), y(0), z(0) {}
	Point3D(double x, double y, double z) :x(x), y(y), z(z) {}

	// operators
	Point3D& operator=(const Point3D &a)      { x = a.x; y = a.y; z = a.z; return *this; }
	Point3D& operator=(const cv::Point3d &a)  { x = a.x; y = a.y; z = a.z; return *this; }
	Point3D& operator+=(const Point3D &a)     { x = x + a.x; y = y + a.y; z = z + a.z; return *this; }
	Point3D& operator-=(const Point3D &a)     { x = x - a.x; y = y - a.y; z = z - a.z; return *this; }
	Point3D& operator+=(const double s)       { x = x + s; y = y + s; z = z + s; return *this; }
	Point3D& operator-=(const double s)       { x = x - s; y = y - s; z = z - s; return *this; }
	Point3D& operator*=(const double s)       { x = x * s; y = y * s; z = z * s; return *this; }
	Point3D& operator/=(const double s)       { x = x / s; y = y / s; z = z / s; return *this; }
	Point3D  operator+(const Point3D &a)      { return Point3D(x + a.x, y + a.y, z + a.z); }
	Point3D  operator+(const double s)        { return Point3D(x + s, y + s, z + s); }
	Point3D  operator-(const Point3D &a)      { return Point3D(x - a.x, y - a.y, z - a.z); }
	Point3D  operator-(const double s)        { return Point3D(x - s, y - s, z - s); }
	Point3D  operator-()                      { return Point3D(-x, -y, -z); }
	Point3D  operator*(const double s)        { return Point3D(x * s, y * s, z * s); }
	Point3D  operator/(const double s)        { return Point3D(x / s, y / s, z / s); }
	bool     operator==(const Point3D &a)     { return (x == a.x && y == a.y && z == a.z); }
	bool     operator==(const cv::Point3d &a) { return (x == a.x && y == a.y && z == a.z); }

	// methods
	double norm_L2() { return std::sqrt(x * x + y * y + z * z); }
	double dot(const Point3D &a) { return x * a.x + y * a.y + z * a.z; }

	// data conversion
	cv::Point3d cv() { return cv::Point3d(x, y, z); }

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	// data
	double x;
	double y;
	double z;
};

typedef std::pair<Point2D, Point2D> Line2D;
typedef std::pair<Point3D, Point3D> Line3D;
struct FOV
{
	Point3D corner[4];
};

class Rect
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	// constructors
	Rect() : x(0), y(0), w(0), h(0) {}
	Rect(double x, double y, double w, double h) :x(x), y(y), w(w), h(h) {}

	// operators
	Rect& operator=(const Rect &a)      { x = a.x; y = a.y; w = a.w; h = a.h; return *this; }
	Rect& operator=(const cv::Rect &a)  { x = a.x; y = a.y; w = a.width; h = a.height; return *this; }
	bool  operator==(const Rect &a)     const { return (x == a.x && y == a.y && w == a.w && h == a.h); }
	bool  operator==(const cv::Rect &a) const { return (x == a.x && y == a.y && w == a.width && h == a.height); }
	Rect& operator*=(const double s)    { x = x * s; y = y * s; w = w * s; h = h * s; return *this; }
	Rect operator*(const double s)      const { return Rect(x * s, y * s, w * s, h * s); }

	// methods	
	Point2D bottomCenter() const { return Point2D(x + std::ceil(w / 2.0), y + h); }
	Point2D topCenter()    const { return Point2D(x + std::ceil(w / 2.0), y); }
	Point2D center()       const { return Point2D(x + std::ceil(w / 2.0), y + std::ceil(h / 2.0)); }
	Point2D reconstructionPoint() const
	{
		return this->bottomCenter();
		//switch(PSN_INPUT_TYPE)
		//{
		//case 1:
		//	return this->bottomCenter();
		//	break;
		//default:
		//	return this->center();
		//	break;
		//}
	}
	Rect cropWithSize(const double width, const double height) const
	{
		double newX = std::max(0.0, x);
		double newY = std::max(0.0, y);
		double newW = std::min(width - newX - 1, w);
		double newH = std::min(height - newY - 1, h);
		return Rect(newX, newY, newW, newH);
	}
	Rect scale(double scale) const
	{
		return Rect(x * scale, y * scale, w * scale, h * scale);
	}
	double area() { return w * h; }
	bool contain(const Point2D &a)     const { return (a.x >= x && a.x < x + w && a.y >= y && a.y < y + h); }
	bool contain(const cv::Point2f &a) const { return ((double)a.x >= x && (double)a.x < x + w && (double)a.y >= y && (double)a.y < y + h); }
	bool overlap(const Rect &a) const
	{
		return (std::max(x + w, a.x + a.w) - std::min(x, a.x) < w + a.w) && (std::max(y + h, a.y + a.h) - std::min(y, a.y) < h + a.h) ? true : false;
	}
	double distance(const Rect &a) const
	{
		Point3D descriptor1 = Point3D(x + w / 2.0, y + h / 2.0, w);
		Point3D descriptor2 = Point3D(a.x + a.w / 2.0, a.y + a.h / 2.0, a.w);
		return (descriptor1 - descriptor2).norm_L2() / std::min(w, a.w);
	}
	double overlappedArea(const Rect &a) const
	{
		double overlappedWidth = std::min(x + w, a.x + a.w) - std::max(x, a.x);
		if (0.0 >= overlappedWidth) { return 0.0; }
		double overlappedHeight = std::min(y + h, a.y + a.h) - std::max(y, a.y);
		if (0.0 >= overlappedHeight) { return 0.0; }
		return overlappedWidth * overlappedHeight;
	}

	// conversion
	cv::Rect cv() const { return cv::Rect((int)x, (int)y, (int)w, (int)h); }

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	double x;
	double y;
	double w;
	double h;
};


/////////////////////////////////////////////////////////////////////////
// OBJECTS
/////////////////////////////////////////////////////////////////////////
class CDetection
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	// constructors
	CDetection() : box(0.0, 0.0, 0.0, 0.0), depth(0.0), score(0.0) {}
	CDetection(const CDetection &c) { *this = c; };

	double GetDepth()
	{
		assert(!patch.empty());
		int numPixels = 0;
		double totalDepths = 0.0;
		double windowSize = patch.cols / 4;
		
		for (int col = std::max(0, int(patch.cols / 2 - windowSize));
			col < std::min(int(patch.cols / 2 + windowSize), patch.cols);
			col++)
		{
			for (int row = std::max(0, int(patch.rows / 2 - windowSize));
				row < std::min(int(patch.rows / 2 + windowSize), patch.rows);
				row++)
			{
				numPixels++;
				totalDepths += (double)patch.at<unsigned char>(row, col);
			}
		}
		depth = totalDepths / (double)numPixels;

		return depth;
	}

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	Rect   box;
	double depth;
	double score;
	cv::Mat patch;
	int id; // for GT
};
typedef std::vector<CDetection> DetectionSet;


class CObjectInfo
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	// constructors
	CObjectInfo() : id(0), box(0.0, 0.0, 0.0, 0.0), depth(0.0) {}

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:
	unsigned int id;
	Rect         box;
	double       depth;
};


/////////////////////////////////////////////////////////////////////////
// RESULTS
/////////////////////////////////////////////////////////////////////////
class CTrackResult
{
	//----------------------------------------------------------------
	// METHODS
	//----------------------------------------------------------------
public:
	// constructors
	CTrackResult() : frameIdx(0), timeStamp(0) {}

	//----------------------------------------------------------------
	// VARIABLES
	//----------------------------------------------------------------
public:	
	unsigned int frameIdx;
	unsigned int timeStamp;
	std::vector<CObjectInfo> objectInfos;

	std::vector<Rect> vecDetectionRects;
	std::vector<Rect> vecTrackerRects;
	cv::Mat matMatchingCost;
};


/////////////////////////////////////////////////////////////////////////
// BUFFER RELATED
/////////////////////////////////////////////////////////////////////////
class CCircularIndex
{
public:
	CCircularIndex() : size_(0), currentIndex_(0) {}
	~CCircularIndex() {}
	// manupulation
	void setSize(int size) { size_ = size; currentIndex_ = 0; }
	void operator++(void)  { if (size_ && ++currentIndex_ >= size_) { currentIndex_ -= size_; } }
	void operator--(void)  { if (size_ && --currentIndex_ < 0) { currentIndex_ += size_; } }
	void operator++(int)   { ++*this; }
	void operator--(int)   { --*this; }
	// position access
	int size() { return size_; }
	int next()
	{
		int nextPos = currentIndex_ + (1 & size_);
		if (nextPos >= size_) { nextPos -= size_; }
		return nextPos;
	}
	int current() { return currentIndex_; }
	int previous()
	{
		int prevPos = currentIndex_ - (1 & size_);
		if (prevPos < 0) { prevPos += size_; }
		return prevPos;
	}
private:
	int size_;
	int currentIndex_;
};

class CFIFOIndicator
{
public:
	CFIFOIndicator::CFIFOIndicator(int size)
	{
		size_ = size;
		this->end();
	};
	CFIFOIndicator::~CFIFOIndicator() {};

	int current() { return pos_; }
	int next() { return ++pos_; }
	int previous() { return --pos_; }
	int end() { return pos_ = size_ - 1; }
private:
	int size_;
	int pos_;
};

class CMatFIFOBuffer
{
public:
	CMatFIFOBuffer() : bInit_(false) {}
	CMatFIFOBuffer(int bufferSize) : bInit_(true), bufferSize_(bufferSize) {}
	~CMatFIFOBuffer() { clear(); }

	bool set(int bufferSize)
	{
		if (bInit_) { this->clear(); }
		try
		{
			bufferSize_ = bufferSize;
		}
		catch (int e)
		{
			printf("An execption occured in CircularBuffer::set. Exeption number is %d.\n", e);
			return false;
		}

		return bInit_ = true;
	}

	bool clear()
	{
		if (!bInit_) { return true; }
		try
		{
			for (int bufferIdx = 0; bufferIdx < buffer_.size(); bufferIdx++)
			{
				this->remove(bufferIdx);
			}
		}
		catch (int e)
		{
			printf("An execption occured in CircularBuffer::clear. Exeption number is %d.\n", e);
			return false;
		}
		bufferSize_ = 0;
		bInit_ = false;

		return true;
	}

	bool insert(cv::Mat newMat)
	{
		//assert(!bInit_
		//	&& newMat.rows == elementSize_.height
		//	&& newMat.cols == elementSize_.width
		//	&& newMat.type() == elementType_);

		try
		{
			cv::Mat newBufferMat = newMat.clone();
			buffer_.push_back(newBufferMat);

			// circulation
			if (bufferSize_ < buffer_.size())
			{
				buffer_.pop_front();
			}
		}
		catch (int e)
		{
			printf("An execption occured in CircularBuffer::insert. Exeption number is %d.\n", e);
			return false;
		}

		return true;
	}

	bool insert_resize(cv::Mat newMat, cv::Size resizeParam)
	{
		//assert(!bInit_
		//	&& newMat.rows == elementSize_.height
		//	&& newMat.cols == elementSize_.width
		//	&& newMat.type() == elementType_);

		try
		{
			cv::Mat newBufferMat;
			cv::resize(newMat, newBufferMat, resizeParam);
			buffer_.push_back(newBufferMat);

			// circulation
			if (bufferSize_ < buffer_.size())
			{
				buffer_.pop_front();
			}
		}
		catch (int e)
		{
			printf("An execption occured in CircularBuffer::insert. Exeption number is %d.\n", e);
			return false;
		}

		return true;
	}

	bool remove(int pos)
	{
		assert(bInit_ && pos < bufferSize_);
		if (pos >= buffer_.size())
		{
			return true;
		}
		if (!buffer_[pos].empty())
		{
			buffer_[pos].release();
		}
		return true;
	}

	size_t size()
	{
		return bufferSize_;
	}

	size_t num_elements()
	{
		return buffer_.size();
	}

	cv::Mat front()
	{
		assert(bInit_);
		return buffer_.front();
	}

	cv::Mat back()
	{
		assert(bInit_);
		return buffer_.back();
	}

	cv::Mat get(int pos)
	{
		assert(bInit_ && pos < bufferSize_);
		return buffer_[pos];
	}

	int get_back_idx()
	{
		return (int)buffer_.size();
	}

	CFIFOIndicator get_indicator()
	{
		CFIFOIndicator newIndicator(bufferSize_);
		return newIndicator;
	}

	/* iterators */
	typedef std::deque<cv::Mat>::iterator iterator;
	typedef std::deque<cv::Mat>::const_iterator const_iterator;
	typedef std::deque<cv::Mat>::reverse_iterator reverse_iterator;
	typedef std::deque<cv::Mat>::const_reverse_iterator const_reverse_iterator;

	iterator begin() { return buffer_.begin(); }
	iterator end()   { return buffer_.end();  }
	const_iterator begin() const { return buffer_.begin(); }
	const_iterator end()   const { return buffer_.end(); }
	reverse_iterator rbegin() { return buffer_.rbegin(); }
	reverse_iterator rend()   { return buffer_.rend(); }
	const_reverse_iterator rbegin() const { return buffer_.rbegin(); }
	const_reverse_iterator rend()   const { return buffer_.rend(); }

private:
	bool bInit_;
	int  bufferSize_;
	std::deque<cv::Mat> buffer_;
};


/////////////////////////////////////////////////////////////////////////
// PARAMETERS
/////////////////////////////////////////////////////////////////////////
struct stViewInformation
{	
	//------------------------------------------------
	// METHODS
	//------------------------------------------------
	stViewInformation()
		: nNumCameras(0)
	{};
	~stViewInformation() {};

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
	int nNumCameras;
	std::vector<int> vecCamIDs;
};

struct stParamTrack
{
	//------------------------------------------------
	// METHODS
	//------------------------------------------------
	stParamTrack()		
		: nImageWidth(0)
		, nImageHeight(0)		
		, dImageRescale(1.0)
		, dImageRescaleRecover(1.0)
		, dDetectionMinHeight(1400.0)
		, dDetectionMaxHeight(2300.0)
		, nMaxTrackletLength(5)
		, nMinNumFeatures(4)
		, nMaxNumFeatures(100)
		, nBackTrackingLength(4)
		, dFeatureTrackWindowSizeRatio(1.0)
		, dMaxBoxDistance(1.0)
		, dMinBoxOverlapRatio(0.3)
		, dMaxBoxCenterDiffRatio(0.5)
		, dMinOpticalFlowMajorityRatio(0.5)
		, dMaxDetectionDistance(500.0)
		, dMaxHeightDifference(400.0)
		, bVisualize(false)
		, bVideoRecord(false)
		, strVideoRecordPath("")
	{};
	~stParamTrack() {};

	//------------------------------------------------
	// VARIABLES
	//------------------------------------------------
	int nImageWidth;
	int nImageHeight;

	/* speed-up */
	double dImageRescale;
	double dImageRescaleRecover;

	/* detection validation  */
	double dDetectionMinHeight;
	double dDetectionMaxHeight;

	/* bi-directional tracking */
	int    nMaxTrackletLength;
	int    nMinNumFeatures;
	int    nMaxNumFeatures;
	int    nBackTrackingLength;	
	double dFeatureTrackWindowSizeRatio;
	double dMaxBoxDistance;
	double dMinBoxOverlapRatio;
	double dMaxBoxCenterDiffRatio;
	double dMinOpticalFlowMajorityRatio;

	/* matching score related */
	double dMaxDetectionDistance;
	double dMaxHeightDifference;

	/* visualization for debugging */
	bool   bVisualize;

	/* video recording for result visualization */
	bool        bVideoRecord;
	std::string strVideoRecordPath;
};


}


#endif

//()()
//('')HAANJU.YOO


