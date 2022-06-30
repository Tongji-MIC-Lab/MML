#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;

class CBinFile
{
private:
	string filepath;
	FILE *hTmp;
	off_t fLen;
	off_t tRows;
	off_t dims;
public:
    CBinFile();
	~CBinFile(void);

private:
	bool bWrite;
public:
	bool InitWrite(string strfile,long dims);
	bool writeLines(const float *data,int num);
	void wlastLine();

private:
    bool bReadBin;
public:
    off_t InitReadBin(string strfile);
    size_t readBin(char* pbuf);
    off_t GetDim();
};

