#ifndef _MANIFOLDS_
#define _MANIFOLDS_

#include <opencv2/opencv.hpp>
#include <string>
using namespace std;
using namespace cv;

#define c_d 14
#define d_cov ((c_d * c_d + c_d)/2 - 3)

#ifndef _RectInfo_
#define _RectInfo_
typedef struct {
    int x;
	int y;
	int width;
	int height;
}RectInfo;
#endif

class CTBC
{
private:
	bool m_bInit;
    cv::Mat mP[c_d];
    cv::Mat mQ[c_d][c_d];
	int m_width;
	int m_height;
public:
    CTBC(void);
    ~CTBC(void);

    void Initial(cv::Mat &op);
    bool GetCovMatInVec(RectInfo rt,vector<float> &mDes, const int index);
private:
    bool GetCovMatInMat(RectInfo rt,cv::Mat &mCov);
    float GetIntegal(const cv::Mat& m, RectInfo rt);
};
#endif//_MANIFOLDS_

