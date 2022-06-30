#include "Track.h"
#include "OpticalFlow.h"
#include "feature.h"
#include "vfc.h"
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
using namespace std;


// parameters
const float scale_stride = sqrt(2);
const int patch_size = 32;
const float epsilon = 0.05;
const float min_flow = 0.4;

const float min_var = sqrt(3);
const float max_var = 50;
const float max_dis = 20;

DescMat* InitDescMat(int height, int width, int nBins)
{
    DescMat* descMat = (DescMat*)malloc(sizeof(DescMat));
    descMat->height = height;
    descMat->width = width;
    descMat->nBins = nBins;

    long size = height*width*nBins;
    descMat->desc = (float*)malloc(size*sizeof(float));
    memset(descMat->desc, 0, size*sizeof(float));
    return descMat;
}

void ReleDescMat(DescMat* descMat)
{
    free(descMat->desc);
    free(descMat);
}

void InitDescInfo(DescInfo* descInfo, int nBins, bool isHof, int size, int nxy_cell, int nt_cell)
{
    descInfo->nBins = nBins;
    descInfo->isHof = isHof;
    descInfo->nxCells = nxy_cell;
    descInfo->nyCells = nxy_cell;
    descInfo->ntCells = nt_cell;
    descInfo->dim = nBins*nxy_cell*nxy_cell;
    descInfo->height = size;
    descInfo->width = size;
}

void InitSeqInfo(SeqInfo* seqInfo, char* video)
{
    VideoCapture capture;
    capture.open(video);

    if(!capture.isOpened())
        fprintf(stderr, "Could not initialize capturing..\n");

    // get the number of frames in the video
    int frame_num = 0;
    while(true) {
        Mat frame;
        capture >> frame;

        if(frame.empty())
            break;

        if(frame_num == 0) {
            seqInfo->width = frame.cols;
            seqInfo->height = frame.rows;
        }

        frame_num++;
    }
    seqInfo->length = frame_num;
}

// get the rectangle for computing the descriptor
void GetRect(const Point2f& point, RectInfo& rect, const int width, const int height, const DescInfo& descInfo)
{
    int x_min = descInfo.width/2;
    int y_min = descInfo.height/2;
    int x_max = width - descInfo.width;
    int y_max = height - descInfo.height;

    rect.x = min<int>(max<int>(cvRound(point.x) - x_min, 0), x_max);
    rect.y = min<int>(max<int>(cvRound(point.y) - y_min, 0), y_max);
    rect.width = descInfo.width;
    rect.height = descInfo.height;
}

// compute integral histograms for the whole image
void BuildDescMat(const Mat& xComp, const Mat& yComp, float* desc, const DescInfo& descInfo)
{
    float maxAngle = 360.f;
    int nDims = descInfo.nBins;
    // one more bin for hof
    int nBins = descInfo.isHof ? descInfo.nBins-1 : descInfo.nBins;
    const float angleBase = float(nBins)/maxAngle;

    int step = (xComp.cols+1)*nDims;
    int index = step + nDims;
    for(int i = 0; i < xComp.rows; i++, index += nDims) {
        const float* xc = xComp.ptr<float>(i);
        const float* yc = yComp.ptr<float>(i);

        // summarization of the current line
        vector<float> sum(nDims);
        for(int j = 0; j < xComp.cols; j++) {
            float x = xc[j];
            float y = yc[j];
            float mag0 = sqrt(x*x + y*y);
            float mag1;
            int bin0, bin1;

            // for the zero bin of hof
            if(descInfo.isHof && mag0 <= min_flow) {
                bin0 = nBins; // the zero bin is the last one
                mag0 = 1.0;
                bin1 = 0;
                mag1 = 0;
            }
            else {
                float angle = fastAtan2(y, x);
                if(angle >= maxAngle) angle -= maxAngle;

                // split the mag to two adjacent bins
                float fbin = angle * angleBase;
                bin0 = cvFloor(fbin);
                bin1 = (bin0+1)%nBins;

                mag1 = (fbin - bin0)*mag0;
                mag0 -= mag1;
            }

            sum[bin0] += mag0;
            sum[bin1] += mag1;

            for(int m = 0; m < nDims; m++, index++)
                desc[index] = desc[index-step] + sum[m];
        }
    }
}

// get a descriptor from the integral histogram
void GetDesc(const DescMat* descMat, RectInfo& rect, DescInfo descInfo, vector<float>& desc, const int index)
{
    int dim = descInfo.dim;
    int nBins = descInfo.nBins;
    int height = descMat->height;
    int width = descMat->width;

    int xStride = rect.width/descInfo.nxCells;
    int yStride = rect.height/descInfo.nyCells;
    int xStep = xStride*nBins;
    int yStep = yStride*width*nBins;

    // iterate over different cells
    int iDesc = 0;
    vector<float> vec(dim);
    for(int xPos = rect.x, x = 0; x < descInfo.nxCells; xPos += xStride, x++)
    for(int yPos = rect.y, y = 0; y < descInfo.nyCells; yPos += yStride, y++) {
        // get the positions in the integral histogram
        const float* top_left = descMat->desc + (yPos*width + xPos)*nBins;
        const float* top_right = top_left + xStep;
        const float* bottom_left = top_left + yStep;
        const float* bottom_right = bottom_left + xStep;

        for(int i = 0; i < nBins; i++) {
            float sum = bottom_right[i] + top_left[i] - bottom_left[i] - top_right[i];
            vec[iDesc++] = max<float>(sum, 0) + epsilon;
        }
    }

    float norm = 0;
    for(int i = 0; i < dim; i++)
        norm += vec[i];
    if(norm > 0) norm = 1./norm;

    int pos = index*dim;
    for(int i = 0; i < dim; i++)
        desc[pos++] = sqrt(vec[i]*norm);
}

// check whether a trajectory is valid or not
bool IsValid(vector<Point2f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length)
{
    int size = track.size();
    float norm = 1./size;
    for(int i = 0; i < size; i++) {
        mean_x += track[i].x;
        mean_y += track[i].y;
    }
    mean_x *= norm;
    mean_y *= norm;

    for(int i = 0; i < size; i++) {
        float temp_x = track[i].x - mean_x;
        float temp_y = track[i].y - mean_y;
        var_x += temp_x*temp_x;
        var_y += temp_y*temp_y;
    }
    var_x *= norm;
    var_y *= norm;
    var_x = sqrt(var_x);
    var_y = sqrt(var_y);

    // remove static trajectory
    if(var_x < min_var && var_y < min_var)
        return false;
    // remove random trajectory
    if( var_x > max_var || var_y > max_var )
        return false;

    float cur_max = 0;
    for(int i = 0; i < size-1; i++) {
        track[i] = track[i+1] - track[i];
        float temp = sqrt(track[i].x*track[i].x + track[i].y*track[i].y);

        length += temp;
        if(temp > cur_max)
            cur_max = temp;
    }

    if(cur_max > max_dis && cur_max > length*0.7)
        return false;

    track.pop_back();
    norm = 1./length;
    // normalize the trajectory
    for(int i = 0; i < size-1; i++)
        track[i] *= norm;

    return true;
}

bool IsCameraMotion(vector<Point2f>& disp)
{
    float disp_max = 0;
    float disp_sum = 0;
    for(int i = 0; i < disp.size(); ++i) {
        float x = disp[i].x;
        float y = disp[i].y;
        float temp = sqrt(x*x + y*y);

        disp_sum += temp;
        if(disp_max < temp)
            disp_max = temp;
    }

    if(disp_max <= 1)
        return false;

    float disp_norm = 1./disp_sum;
    for (int i = 0; i < disp.size(); ++i)
        disp[i] *= disp_norm;

    return true;
}

int InitPry(const Mat& frame, vector<float>& scales, vector<Size>& sizes)
{
    int scale_num = 8;
    int rows = frame.rows, cols = frame.cols;
    float min_size = min<int>(rows, cols);

    int nlayers = 0;
    while(min_size >= patch_size) {
        min_size /= scale_stride;
        nlayers++;
    }

    if(nlayers == 0) nlayers = 1; // at least 1 scale

    scale_num = min<int>(scale_num, nlayers);

    scales.resize(scale_num);
    sizes.resize(scale_num);

    scales[0] = 1.;
    sizes[0] = Size(cols, rows);

    for(int i = 1; i < scale_num; i++) {
        scales[i] = scales[i-1] * scale_stride;
        sizes[i] = Size(cvRound(cols/scales[i]), cvRound(rows/scales[i]));
    }
    return scale_num;
}

void BuildPry(const vector<Size>& sizes, const int type, vector<Mat>& grey_pyr)
{
    int nlayers = sizes.size();
    grey_pyr.resize(nlayers);

    for(int i = 0; i < nlayers; i++)
        grey_pyr[i].create(sizes[i], type);
}

static void MyWarpPerspective(Mat& prev_src, Mat& src, Mat& dst, Mat& M0, int flags = INTER_LINEAR,
                             int borderType = BORDER_CONSTANT, const Scalar& borderValue = Scalar())
{
    int width = src.cols;
    int height = src.rows;
    dst.create( height, width, CV_8UC1 );

    Mat mask = Mat::zeros(height, width, CV_8UC1);
    const int margin = 5;

    const int BLOCK_SZ = 32;
    short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];

    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    double M[9];
    Mat matM(3, 3, CV_64F, M);
    M0.convertTo(matM, matM.type());
    if( !(flags & WARP_INVERSE_MAP) )
         invert(matM, matM);

    int x, y, x1, y1;

    int bh0 = min(BLOCK_SZ/2, height);
    int bw0 = min(BLOCK_SZ*BLOCK_SZ/bh0, width);
    bh0 = min(BLOCK_SZ*BLOCK_SZ/bw0, height);

    for( y = 0; y < height; y += bh0 ) {
    for( x = 0; x < width; x += bw0 ) {
        int bw = min( bw0, width - x);
        int bh = min( bh0, height - y);

        Mat _XY(bh, bw, CV_16SC2, XY);
        Mat matA;
        Mat dpart(dst, Rect(x, y, bw, bh));

        for( y1 = 0; y1 < bh; y1++ ) {

            short* xy = XY + y1*bw*2;
            double X0 = M[0]*x + M[1]*(y + y1) + M[2];
            double Y0 = M[3]*x + M[4]*(y + y1) + M[5];
            double W0 = M[6]*x + M[7]*(y + y1) + M[8];
            short* alpha = A + y1*bw;

            for( x1 = 0; x1 < bw; x1++ ) {

                double W = W0 + M[6]*x1;
                W = W ? INTER_TAB_SIZE/W : 0;
                double fX = max((double)INT_MIN, min((double)INT_MAX, (X0 + M[0]*x1)*W));
                double fY = max((double)INT_MIN, min((double)INT_MAX, (Y0 + M[3]*x1)*W));

                double _X = fX/double(INTER_TAB_SIZE);
                double _Y = fY/double(INTER_TAB_SIZE);

                if( _X > margin && _X < width-1-margin && _Y > margin && _Y < height-1-margin )
                    mask.at<uchar>(y+y1, x+x1) = 1;

                int X = saturate_cast<int>(fX);
                int Y = saturate_cast<int>(fY);

                xy[x1*2] = saturate_cast<short>(X >> INTER_BITS);
                xy[x1*2+1] = saturate_cast<short>(Y >> INTER_BITS);
                alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (X & (INTER_TAB_SIZE-1)));
            }
        }

        Mat _matA(bh, bw, CV_16U, A);
        remap( src, dpart, _XY, _matA, interpolation, borderType, borderValue );
    }
    }

    for( y = 0; y < height; y++ ) {
        const uchar* m = mask.ptr<uchar>(y);
        const uchar* s = prev_src.ptr<uchar>(y);
        uchar* d = dst.ptr<uchar>(y);
        for( x = 0; x < width; x++ ) {
            if(m[x] == 0)
                d[x] = s[x];
        }
    }
}

void ComputeMatch(const cv::Mat& pre_grey,const cv::Mat& flow,const vector<KeyPoint>& prev_kpts, const vector<KeyPoint>& kpts,
    const Mat& prev_desc, const Mat& desc, vector<Point2f>& prev_pts, vector<Point2f>& pts)
{
    prev_pts.clear();
    pts.clear();

    if(prev_kpts.size() == 0 || kpts.size() == 0)
        return;

    Mat mask = windowedMatchingMask(kpts, prev_kpts, 25, 25);
    BFMatcher desc_matcher(NORM_L2);
    vector<DMatch> matches;
    desc_matcher.match(desc, prev_desc, matches, mask);

    //VFC filter SURF
    {
        vector<Point2f> X;
        vector<Point2f> Y;
        for (unsigned int i = 0; i < matches.size(); i++) {
            int idx1 = matches[i].trainIdx;
            int idx2 = matches[i].queryIdx;

            if (idx1<prev_kpts.size())
                X.push_back(prev_kpts[idx1].pt);
            else
                printf("prev_kpts:%d\n",idx1);

            if (idx2<kpts.size())
                Y.push_back(kpts[idx2].pt);
            else
                printf("kpts:%d\n",idx2);
        }

        VFC myvfc;
        myvfc.setData(X, Y);
        myvfc.optimize();
        vector<int> matchIdx = myvfc.obtainCorrectMatch();
        for (unsigned int i=0;i<matchIdx.size();i++)
        {
            int idx = matchIdx[i];
            if (idx>=X.size() || idx >=Y.size())
            {
                printf("ComputeMatch:%d\n",idx);
            }
            else
            {
                prev_pts.push_back(X[idx]);
                pts.push_back(Y[idx]);
            }
        }
    }

    //optical flow sample points
    {
        const int width = flow.cols;
        const int height = flow.rows;
        vector<Point2f> X;
        vector<Point2f> Y;

        const int MAX_COUNT = 1000;
        goodFeaturesToTrack(pre_grey, X, MAX_COUNT, 0.001, 3);

        for(int i = 0; i < X.size(); i++) {
            int x = min<int>(max<int>(cvRound(X[i].x), 0), width-1);
            int y = min<int>(max<int>(cvRound(X[i].y), 0), height-1);

            const float* f = flow.ptr<float>(y);
            Y.push_back(Point2f(x+f[2*x], y+f[2*x+1]));
        }

        VFC myvfc;
        myvfc.setData(X, Y);
        myvfc.optimize();
        vector<int> matchIdx = myvfc.obtainCorrectMatch();
        for (unsigned int i=0;i<matchIdx.size();i++)
        {
            int idx = matchIdx[i];
            if (idx>=X.size() || idx >=Y.size())
            {
                printf("ComputeMatch:%d\n",idx);
            }
            else
            {
                prev_pts.push_back(X[idx]);
                pts.push_back(Y[idx]);
            }
        }
    }
}

void MotionSample(int iScale, const vector<KeyPoint>& inkpts,
                  const Mat& mflow, vector<Point2f>& points,int frame_num,const cv::Mat &frame)
{
    const int min_distance = 2;
    cv::Mat mcur[2];
    cv::split(mflow,mcur);
    cv::Mat mm(mflow.rows,mflow.cols,CV_32F);
    float fsum = 0;
    int icount = 0;
    const float c_m = 0.01;
    for (int r=0;r<mm.rows;r++)
    {
        for (int c=0;c<mm.cols;c++)
        {
            float ft = sqrt(mcur[0].ptr<float>(r)[c]*mcur[0].ptr<float>(r)[c]
                    + mcur[1].ptr<float>(r)[c]*mcur[1].ptr<float>(r)[c]);

            if (ft<c_m)
                ft = 0;

            mm.ptr<float>(r)[c] = ft;

            if (0 != ft)
            {
                fsum += ft;
                icount ++;
            }
        }
    }
    const float fmean = fsum / icount;
    const float c_threshold = fmean;

    const int width = mm.cols/min_distance;
    const int height = mm.rows/min_distance;
    vector<int> counters(width*height);
    int x_max = min_distance*width;
    int y_max = min_distance*height;

    for(unsigned int i = 0; i < points.size(); i++) {
        Point2f point = points[i];
        int x = cvFloor(point.x);
        int y = cvFloor(point.y);
        if(x >= x_max || y >= y_max)
            continue;
        x /= min_distance;
        y /= min_distance;
        counters[y*width+x]++;
    }
    points.clear();

    const double ptScale = 1 / pow(scale_stride, iScale);
    for(unsigned int i = 0; i < inkpts.size(); i++)
    {
        Point2f ptf;
        ptf.x = inkpts[i].pt.x * ptScale;
        ptf.y = inkpts[i].pt.y * ptScale;

        if(ptf.x >= x_max || ptf.y >= y_max)
            continue;
        int sx = int(ptf.x) / min_distance;
        int sy = int(ptf.y) / min_distance;
        if (counters[sy*width+sx] > 0)
            continue;

        if(mm.ptr<float>(int(ptf.y))[int(ptf.x)] > c_threshold)
        {
            points.push_back(ptf);
            counters[sy*width+sx] ++;
        }
    }

}


int MotionKeypointTrack(string video, string videoId, string resultPath)
{
    int scale_num = 8;
    int nxy_cell = 2;
    int nt_cell = 3;
    int init_gap = 1;
    int track_length = 15;

	VideoCapture capture;
	capture.open(video);

    if(!capture.isOpened())
    {
        printf("[ERROR] [%s] open error.\n",video.c_str());
		return -1;
	}

	int frame_num = 0;
    TrackInfo trackInfo = {track_length, init_gap};
    DescInfo tbcInfo;
    InitDescInfo(&tbcInfo, 0, false, patch_size, nxy_cell, nt_cell);

    CFeature cf;
    cf.Initial(videoId,resultPath,4,nt_cell,trackInfo.length);

	SeqInfo seqInfo;
    InitSeqInfo(&seqInfo, (char*)video.c_str());

    SurfFeatureDetector detector_surf(200);
	SurfDescriptorExtractor extractor_surf(true, true);
    vector<Point2f> prev_pts_surf, pts_surf;
    vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;

	Mat image, prev_grey, grey;

    vector<float> fscales(0);
    vector<Size> sizes(0);
    vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0), flow_warp_pyr(0);
    vector<Mat> prev_poly_pyr(0), poly_pyr(0), poly_warp_pyr(0);
    vector<list<Track> > xyScaleTracks;
    int init_counter = 0;
    while(true)
    {
		Mat frame;
        Mat tmpframe;
        if (false == capture.read(tmpframe))
            break;
        if (tmpframe.empty())
        {
            printf("[ERROR] %s frame is empty\n", videoId.c_str());
            return -1;
        }
        if (tmpframe.cols > 860)
        {
            Size sizef;
            sizef.width = 800;
            sizef.height = (int) (800.0 / tmpframe.cols * tmpframe.rows);
            resize(tmpframe, frame, sizef, 0, 0, INTER_LINEAR);
        }
        else
            frame = tmpframe;
        tmpframe.release();

        if(frame_num == 0)
        {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);

            scale_num = InitPry(frame, fscales, sizes);

			BuildPry(sizes, CV_8UC1, prev_grey_pyr);
			BuildPry(sizes, CV_8UC1, grey_pyr);
			BuildPry(sizes, CV_32FC2, flow_pyr);
			BuildPry(sizes, CV_32FC2, flow_warp_pyr);

			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_warp_pyr);

			xyScaleTracks.resize(scale_num);

			frame.copyTo(image);
			cvtColor(image, prev_grey, CV_BGR2GRAY);

			for(int iScale = 0; iScale < scale_num; iScale++) {
				if(iScale == 0)
					prev_grey.copyTo(prev_grey_pyr[0]);
				else
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);
			}
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);
            detector_surf.detect(prev_grey, prev_kpts_surf);
			extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);
			frame_num++;
			continue;
		}

		init_counter++;
		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

        my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
        my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

        detector_surf.detect(grey, kpts_surf);
		extractor_surf.compute(grey, kpts_surf, desc_surf);
        ComputeMatch(prev_grey,flow_pyr[0],prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);

        Mat H = Mat::eye(3, 3, CV_64FC1);
        if(pts_surf.size() > 50)
        {
            vector<unsigned char> match_mask;
            Mat temp = findHomography(prev_pts_surf, pts_surf, RANSAC, 1, match_mask);
            if(countNonZero(Mat(match_mask)) > 25)
                H = temp;
        }

		Mat H_inv = H.inv();
		Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);
        MyWarpPerspective(prev_grey, grey, grey_warp, H_inv);

		my::FarnebackPolyExpPyr(grey_warp, poly_warp_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_warp_pyr, flow_warp_pyr, 10, 2);

        grey.copyTo(grey_pyr[0]);
        for(int iScale = 1; iScale < scale_num; iScale++)
            resize(grey_pyr[iScale-1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

        if (frame_num == 1)
        {
            for(int iScale = 0; iScale < scale_num; iScale++)
            {
                vector<Point2f> points(0);
                MotionSample(iScale,kpts_surf,flow_warp_pyr[iScale],points,frame_num,frame);

                list<Track>& tracks = xyScaleTracks[iScale];
                for(unsigned int i = 0; i < points.size(); i++)
                    tracks.push_back(Track(points[i], trackInfo));
            }
        }

        if (frame_num>=2)
        {
            for(int iScale = 0; iScale < scale_num; iScale++)
            {
                int width = grey_pyr[iScale].cols;
                int height = grey_pyr[iScale].rows;

                CTBC tbcf;
                tbcf.Initial(flow_warp_pyr[iScale]);

                list<Track>& tracks = xyScaleTracks[iScale];
                for (list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();)
                {
                    int index = iTrack->index;
                    Point2f prev_point = iTrack->point[index];
                    int x = min<int>(max<int>(cvRound(prev_point.x), 0), width-1);
                    int y = min<int>(max<int>(cvRound(prev_point.y), 0), height-1);

                    Point2f point;
                    point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];
                    point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];

                    if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height)
                    {
                        iTrack = tracks.erase(iTrack);
                        continue;
                    }

                    iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];
                    iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];

                    RectInfo rect;
                    GetRect(prev_point, rect, width, height, tbcInfo);
                    tbcf.GetCovMatInVec(rect, iTrack->tbcf, index);
                    iTrack->addPoint(point);

                    if(iTrack->index >= trackInfo.length)
                    {
                        vector<Point2f> trajectory(trackInfo.length+1);
                        for(int i = 0; i <= trackInfo.length; ++i)
                            trajectory[i] = iTrack->point[i]*fscales[iScale];

                        vector<Point2f> displacement(trackInfo.length);
                        for (int i = 0; i < trackInfo.length; ++i)
                            displacement[i] = iTrack->disp[i]*fscales[iScale];

                        float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
                        if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement))
                            cf.writeFeature(mean_x, mean_y, iScale, width, height, frame_num,iTrack,seqInfo);

                        iTrack = tracks.erase(iTrack);
                        continue;
                    }
                    ++iTrack;
                }

                if(init_counter != trackInfo.gap)
                    continue;

                vector<Point2f> points(0);
                for(list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
                    points.push_back(iTrack->point[iTrack->index]);

                MotionSample(iScale,kpts_surf,flow_warp_pyr[iScale],points,frame_num,frame);
                for(unsigned int i = 0; i < points.size(); i++)
                    tracks.push_back(Track(points[i], trackInfo));
            }
        }

		init_counter = 0;
		grey.copyTo(prev_grey);
        for(unsigned int i = 0; i < scale_num; i++)
        {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}

		prev_kpts_surf = kpts_surf;
		desc_surf.copyTo(prev_desc_surf);

		frame_num++;
	}
    return 1;
}
