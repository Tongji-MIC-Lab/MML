#ifndef DENSETRACK_H_
#define DENSETRACK_H_
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>

#include "TBC.h"

using namespace cv;

extern const float scale_stride;


#ifndef _RectInfo_
#define _RectInfo_
typedef struct {
	int x;       // top left corner
	int y;
	int width;
	int height;
}RectInfo;
#endif

typedef struct {
    int width;   // resolution of the video
    int height;
    int length;  // number of frames
}SeqInfo;

typedef struct {
    int length;  // length of the trajectory
    int gap;     // initialization gap for feature re-sampling 
}TrackInfo;

typedef struct {
    int nBins;   // number of bins for vector quantization
    bool isHof; 
    int nxCells; // number of cells in x direction
    int nyCells; 
    int ntCells;
    int dim;     // dimension of the descriptor
    int height;  // size of the block for computing the descriptor
    int width;
}DescInfo; 

// integral histogram for the descriptors
typedef struct {
    int height;
    int width;
    int nBins;
    float* desc;
}DescMat;

class Track
{
public:
    vector<Point2f> point;
    vector<Point2f> disp;
    vector<float> tbcf;

    int index;

    Track(const Point2f& point_, const TrackInfo& trackInfo)
    {
        index = 0;
        disp.resize(trackInfo.length);
        point.resize(trackInfo.length+1);
        point[0] = point_;
        tbcf.resize(trackInfo.length*d_cov);
    }

    void addPoint(const Point2f& point_)
    {
        index++;
        point[index] = point_;
    }
};

#endif /*DENSETRACK_H_*/
