#ifndef TINFO_H
#define TINFO_H
#include <string>
#include <vector>
using namespace std;

#ifndef TInfo
typedef struct _tinfo_
{
    float iScale;
    float x;
    float y;
    float t;
}TInfo;
#endif//TInfo

class FInfo
{
private:
    vector<TInfo> _vecInfo;
public:
    FInfo();
    ~FInfo();
    int Init(string fp);
    void GetInfo(int i,TInfo& v);
};

#endif // TINFO_H
