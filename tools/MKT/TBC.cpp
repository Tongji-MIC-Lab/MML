#include "TBC.h"
#include "stdio.h"

void logcvsvd(const cv::Mat &m,cv::Mat &lm)
{
    CV_Assert(m.rows == m.cols);
    CV_Assert(m.type() == CV_32F);

    const double c_sig = -138.1551;
    const int N = m.rows;

    cv::Mat S,U,VT;
    cv::SVDecomp(m,S,U,VT,SVD::FULL_UV);

    cv::Mat mVal = cv::Mat::zeros(N,N,CV_32F);
    for (int r=0;r<N;r++)
    {
        double td = S.ptr<float>(0)[r];
        if (td==0)
        {
            mVal.ptr<float>(r)[r] = c_sig;
        }
        else
        {
            mVal.ptr<float>(r)[r] = log(td);
        }
    }
    lm = U * mVal * VT;
}


cv::Mat Cov2Vec(const cv::Mat &m)
{
    CV_Assert(m.type() == CV_32F);
    cv::Mat mvec;
    mvec.create(1,d_cov,CV_32F);
    int pos = 0;
    for (int r=0;r<m.rows;r++)
    {
        for (int c=0;c<m.cols;c++)
        {
            if (r<=c)
            {
                if ((0==r&&0==c)||(0==r&&1==c)||(1==r&&0==c)||(1==r&&1==c))
                    continue;

                mvec.ptr<float>(0)[pos] = m.ptr<float>(r)[c];

                pos++;
            }
        }
    }
    return mvec;
}

static int sgn(double x)
{
	if (x>0) return(1);
	if (x<0) return(-1);
	return(0);
}


CTBC::CTBC(void)
{
	m_bInit = false;
}


CTBC::~CTBC(void)
{
}

void CTBC::Initial(cv::Mat &op)
{
    m_width = op.cols;
    m_height = op.rows;
    cv::Mat mOf[2];
    cv::split(op,mOf);
    Mat udX, udY, vdX, vdY;
    Sobel(mOf[0], udX, CV_32F, 1, 0, 1);
    Sobel(mOf[0], udY, CV_32F, 0, 1, 1);
    Sobel(mOf[1], vdX, CV_32F, 1, 0, 1);
    Sobel(mOf[1], vdY, CV_32F, 0, 1, 1);

    cv::Mat mF[c_d];
    mF[0].create(m_height,m_width,CV_32F);
    mF[1].create(m_height,m_width,CV_32F);
    mF[2] = mOf[0];
    mF[3] = mOf[1];
    cv::magnitude(mOf[0],mOf[1],mF[4]);
    mF[5].create(m_height,m_width,CV_32F);
    mF[6] = udX;
    mF[7] = udY;
    mF[8] = vdX;
    mF[9] = vdY;
    cv::magnitude(udX,udY,mF[10]);
    cv::magnitude(vdX,vdY,mF[11]);
    mF[12].create(m_height,m_width,CV_32F);
    mF[13].create(m_height,m_width,CV_32F);

    for (int r=0;r<m_height;r++)
    {
        for (int c=0;c<m_width;c++)
        {
            mF[0].ptr<float>(r)[c] = c;
            mF[1].ptr<float>(r)[c] = r;

            mF[5].ptr<float>(r)[c] = cv::fastAtan2(abs(mOf[1].ptr<float>(r)[c]),abs(mOf[0].ptr<float>(r)[c]));
            mF[12].ptr<float>(r)[c] = cv::fastAtan2(abs(udY.ptr<float>(r)[c]),abs(udX.ptr<float>(r)[c]));
            mF[13].ptr<float>(r)[c] = cv::fastAtan2(abs(vdY.ptr<float>(r)[c]),abs(vdX.ptr<float>(r)[c]));
        }
    }

    for (int i=0;i<c_d;i++)
    {
        cv::integral(mF[i],mP[i]);
    }

    cv::Mat mMul[c_d][c_d];
    for (int i=0;i<c_d;i++)
    {
        for (int j=i;j<c_d;j++)
        {
            if (j>=i)
            {
                cv::multiply(mF[i],mF[j],mMul[i][j]);
                cv::integral(mMul[i][j],mQ[i][j]);
            }
        }
    }
    m_bInit = true;
}

bool CTBC::GetCovMatInMat(RectInfo rt,cv::Mat &mCov)
{
    if (m_bInit)
    {
        if (rt.x < 0)
            rt.x =0;
        if (rt.y <0)
            rt.y = 0;
        if (rt.x + rt.width>m_width)
            rt.width = m_width-rt.x;
        if (rt.y + rt.height>m_height)
            rt.height = m_height-rt.y;

        float S = rt.width*rt.height;

        cv::Mat mTmp = cv::Mat::zeros(c_d,c_d,CV_32F);
        for (int i=0;i<c_d;i++)
        {
            for (int j=0;j<c_d;j++)
            {
                if (j>=i)
                {
                    mTmp.ptr<float>(i)[j] = 1/(S-1)*(GetIntegal(mQ[i][j],rt)-1/S*GetIntegal(mP[i],rt)*GetIntegal(mP[j],rt));
                    mTmp.ptr<float>(j)[i] = mTmp.ptr<float>(i)[j];
                }
            }
        }

        logcvsvd(mTmp,mCov);
    }
    return true;
}

bool CTBC::GetCovMatInVec(RectInfo rt,vector<float> &mDes, const int index)
{
	bool bReturn = false;
	if (m_bInit)
    {
        cv::Mat mCov;
        if (this->GetCovMatInMat(rt,mCov))
        {
            cv::Mat mVec = Cov2Vec(mCov);
            CV_Assert(d_cov == mVec.cols);
            CV_Assert(mVec.type() == CV_32F);

            vector<float> vecTmp;
            vecTmp.resize(d_cov);
            float sum = 0;
            for (int i=0;i<d_cov;i++)
            {
                vecTmp[i] = mVec.ptr<float>(0)[i];
                sum += abs(vecTmp[i]);
            }

            for (int i=0;i<d_cov;i++)
            {
                vecTmp[i] = vecTmp[i]/sum;
                mDes[index*d_cov+i] = sgn(vecTmp[i])*sqrt(abs(vecTmp[i]));
            }
            bReturn = true;
        }
        else
        {
            bReturn = false;
        }
	}
	return bReturn;
}

float CTBC::GetIntegal(const cv::Mat& m, RectInfo rt)
{
	float fReturn = 0;
	if (m_bInit)
	{
		CV_Assert(m.type()==CV_32F || m.type() == CV_64F);

		CV_Assert(rt.x+rt.width<m.cols);
		CV_Assert(rt.y+rt.height<m.rows);
		CV_Assert(rt.x>=0);
		CV_Assert(rt.y>=0);

		switch(m.type())
		{
		case CV_32F:
			{
				fReturn = m.ptr<float>(rt.y+rt.height)[rt.x+rt.width]+m.ptr<float>(rt.y)[rt.x]
				- m.ptr<float>(rt.y+rt.height)[rt.x] - m.ptr<float>(rt.y)[rt.x+rt.width];
				break;
			}
		case CV_64F:
			{
				fReturn = m.ptr<double>(rt.y+rt.height)[rt.x+rt.width]+m.ptr<double>(rt.y)[rt.x]
				- m.ptr<double>(rt.y+rt.height)[rt.x] - m.ptr<double>(rt.y)[rt.x+rt.width];
				break;
			}
		}
	}
	return fReturn;
}
