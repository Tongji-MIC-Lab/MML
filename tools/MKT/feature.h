#ifndef CFEATURE_H
#define CFEATURE_H
#include "vector"
#include "Track.h"
#include "tinfo.h"
#include "BinFile.h"

using namespace std;

class CFeature
{
private:
    int length;
    int ntCells;
    CBinFile hInfo;
    CBinFile hTbcF;

public:
    CFeature(){}
     ~CFeature();
    void Initial(string videoId, string resultPath, int dInfo, int nt, int len);
    void writeFeature(float mean_x, float mean_y, int iScale, int width, int height, int frame_num, list<Track>::iterator& iTrack, SeqInfo &seqInfo);
};

#endif // CFEATURE_H
