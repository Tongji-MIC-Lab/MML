#include "feature.h"
#include "TBC.h"

void avgDesc(vector<float>& desc, const int dim, const int ntCells,const int length, vector<float>& vec)
{
    CV_Assert(desc.size() == dim*length);
    int tStride = cvFloor(length / ntCells);
    float norm = 1. / float(tStride);
    int pos = 0;
    vec.resize(dim*ntCells);
    for (int i = 0; i < ntCells; i++)
    {
        for (int t = 0; t < tStride; t++)
            for (int j = 0; j < dim; j++)
                vec[i * dim + j] += desc[pos++];
    }
    for (int i=0; i<vec.size(); i++)
        vec[i] = vec[i]*norm;
}

void CFeature::Initial(string videoId, string resultPath, int dInfo,int nt,int len)
{
    this->ntCells = nt;
    this->length = len;

    char tmpFile[200] = { 0 };
    sprintf(tmpFile, "%s/TINFO/%s", resultPath.c_str(), videoId.c_str());
    if (!hInfo.InitWrite(tmpFile,dInfo))
        exit(-1);

    sprintf(tmpFile, "%s/TBC/%s", resultPath.c_str(), videoId.c_str());
    if (!hTbcF.InitWrite(tmpFile,d_cov*nt))
        exit(-1);

}

CFeature::~CFeature()
{

}


void CFeature::writeFeature(float mean_x, float mean_y,int iScale, int width, int height,
                            int frame_num, list<Track>::iterator& iTrack, SeqInfo& seqInfo)
{
    TInfo ti;
    ti.iScale = iScale;
    ti.x = min<float>(max<float>(mean_x / float(width), 0), 0.999);
    ti.y = min<float>(max<float>(mean_y / float(height), 0), 0.999);
    ti.t = min<float>(max<float>((frame_num - this->length/2.0)/float(seqInfo.length), 0), 0.999);
    hInfo.writeLines((float*)&ti,sizeof(TInfo)/sizeof(float));

    vector<float> vec;
    avgDesc(iTrack->tbcf,d_cov,this->ntCells,this->length,vec);
    hTbcF.writeLines(&(vec[0]),vec.size());
}
