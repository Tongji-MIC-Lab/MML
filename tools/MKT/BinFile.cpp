#include "BinFile.h"

CBinFile::CBinFile()
{
    hTmp = 0;
    fLen = dims = tRows = 0;
    bReadBin = false;
    bWrite = false;
}

CBinFile::~CBinFile(void)
{
    if (bWrite && hTmp)
        this->wlastLine();
	if (hTmp)
		fclose(hTmp);
}

bool CBinFile::InitWrite(string strfile, long dims)
{
	bWrite = false;
	if (hTmp)
		fclose(hTmp);
	hTmp = fopen(strfile.c_str(), "wb");
	this->dims = dims;
	this->fLen = this->tRows = 0;
    if (hTmp)
        bWrite = true;
    else
        printf("[ERROR] open %s fail.\n",strfile.c_str());
	return bWrite;
}

bool CBinFile::writeLines(const float *data, int num)
{
	bool bReturn = false;
	CV_Assert(0 == num % this->dims);
	if (bWrite && hTmp)
	{
		size_t tw = fwrite(data, sizeof(float), num, hTmp);
		CV_Assert(tw == num);
		fLen += tw * sizeof(float);

		int tr = num / this->dims;
		tRows += tr;
	}
	return bReturn;
}

void CBinFile::wlastLine()
{
	CV_Assert(fLen / dims / sizeof(float) == tRows);
	if (bWrite && hTmp)
	{
		fprintf(hTmp, "%10d%6d", tRows, dims);
        fclose(hTmp);
        hTmp = 0;
	}
}

off_t CBinFile::InitReadBin(string strfile)
{
    if (hTmp)
        fclose(hTmp);
    fLen = -1;
    bReadBin = false;
    hTmp = fopen(strfile.c_str(), "rb");
    if (hTmp)
    {
        char line[20] =
        { 0 };
        fseeko(hTmp, -16, SEEK_END);
        fLen = ftello(hTmp);
        fgets(line, 20, hTmp);
        dims = atoi(line + 10);
        line[10] = 0;
        tRows = atoi(line);
        if (tRows < 0 || dims <= 0 || dims > 10000 || tRows > 9000000000)
        {
            printf("[ERROR][InitReadBin] %s row=%d dims=%d.\n",strfile.c_str(),tRows,dims);
            exit(0);
        }
        if (0 == tRows)
        {
            printf("[INFO][InitReadBin] 0 row. %s \n",strfile.c_str());
            return 0;
        }

        if (dims == fLen / tRows / sizeof(float))
        {
            bReadBin = true;
        }
        else
        {
            printf("[ERROR] Check [%s] file failure.\ntRows=%d dims=%d fLen=%d\n", strfile.c_str(), (int) tRows, (int) dims, (int) fLen);
            exit(0);
        }
        fseeko(hTmp, 0, SEEK_SET);
    }
    return fLen;
}

size_t CBinFile::readBin(char* pbuf)
{
    if (bReadBin && hTmp && fLen > 0)
    {
        size_t tr = fread(pbuf, 1, fLen, hTmp);
        CV_Assert(tr == fLen);
        return tr;
    }
    else
        return -1;
}

off_t CBinFile::GetDim()
{
    return this->dims;
}


