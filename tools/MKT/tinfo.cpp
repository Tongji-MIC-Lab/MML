#include "tinfo.h"
#include <string.h>

FInfo::FInfo()
{

}

FInfo::~FInfo()
{

}

int FInfo::Init(string fp)
{
    int tRows = 0;
    FILE * hTmp = fopen(fp.c_str(), "rb");
    if (hTmp)
    {
        char line[20] = { 0 };
        fseeko(hTmp, -16, SEEK_END);
        __off_t  fLen = ftello(hTmp);
        fgets(line, 20, hTmp);
        int dims = atoi(line + 10);
        line[10] = 0;
        tRows = atoi(line);
        if (tRows < 0 || dims <= 0 || dims > 10000 || tRows > 9000000000)
        {
            printf("[ERROR] [FInfo::Init] tRows=%d dims=%d\n",tRows,dims);
            exit(0);
        }
        if (dims * tRows != fLen / sizeof(float))
        {
            printf("[ERROR] [FInfo::Init] Check [%s] file failure.\ntRows=%d dims=%d fLen=%d\n", fp.c_str(), (int) tRows, (int) dims, (int) fLen);
            exit(0);
        }
        fseeko(hTmp, 0, SEEK_SET);
        if (4 != dims)
        {
            printf("[ERROR] [FInfo::Init] Dimension error %s.\n", fp.c_str());
            exit(0);
        }

        float *pTmp = new float[dims * tRows];
        fread(pTmp,sizeof(float),dims * tRows,hTmp);
        for (int i=0;i<tRows;i++)
        {
            TInfo ti;
            memcpy(&ti,pTmp + dims*i,sizeof(TInfo));
            _vecInfo.push_back(ti);
        }
        delete[]pTmp;
        fclose(hTmp);
    }
    else
    {
        printf("[ERROR] Fail to open infomation file:%s\n", fp.c_str());
        exit(0);
    }
    return tRows;
}

void FInfo::GetInfo(int i,TInfo& v)
{
    if (i<_vecInfo.size())
        v = _vecInfo[i];
    else
        v = TInfo();
}
