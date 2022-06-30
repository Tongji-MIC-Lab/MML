#include "ObjList.h"
#include "common.h"

extern int MotionKeypointTrack(string video, string videoId, string resultPath);

void MKT(int argc, char ** argv)
{
    if (5 != argc)
    {
        printf("Usage: [Video path] [Result path] [Video list] [Extensions]\n");
        return;
    }

    const char * vedioPath = argv[1];
    string resPath = argv[2];
    const char * txtlist = argv[3];
    const char * ext = argv[4];
    CObjList objList;
    if (false == objList.Init(txtlist))
    {
        printf("[ERROR] Initial text file error.\n");
        exit(0);
    }
    yi::makeDir(resPath);
    yi::makeDir(resPath + "/TINFO");
    yi::makeDir(resPath + "/TBC");

    const unsigned int nVedio = objList._vec.size();
#pragma omp parallel for schedule(dynamic)
    for (unsigned int i = 0; i < nVedio; i++)
    {
        string vid = objList._vec[i].id;
        printf("[%d / %d] Start %s\n", i, nVedio,vid.c_str());
        string vfile = yi::format("%s/%s%s",vedioPath,vid.c_str(),ext);
        if (MotionKeypointTrack(vfile, vid, resPath))
            printf("[%d / %d] End %s\n", i+1, nVedio,vid.c_str());
        else
            printf("[ERROR] %s failure.\n", vid.c_str());
    }
}

int main(int argc, char** argv)
{
    printf("MKT V1.0.\r\n");
    MKT(argc, argv);
	return 0;
}
