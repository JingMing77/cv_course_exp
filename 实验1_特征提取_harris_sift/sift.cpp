#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <random>

using namespace cv;
using namespace std;

#define OCTAVE 3

// SIFT
void fSIFT(Mat src, Mat dst, vector<Point2f>& points)
{
    Mat src1 = src.clone();
    //定义SIFT特征检测类对象
    auto sift = SiftFeatureDetector::create();
    //定义KeyPoint变量
    vector<KeyPoint>keyPoints;
    //特征点检测
    sift->detect(src1, keyPoints);
    for (int i = 0; i < keyPoints.size(); i++) {
        points.push_back(keyPoints[i].pt);
    }
    //绘制特征点(关键点)
    drawKeypoints(src1, keyPoints, dst,
        Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

// 构建高斯金字塔
void BuildGaussianPyramid(Mat src, Mat pyr[], int Octaves = 3, int layers = 1)
{
    double* sigma = new double[Octaves * layers];
    double k = pow(2, 1.0 / 3);
    Mat pyr0 = src.clone();
    pyr[0] = pyr0;
    // 计算尺度sigma
    for (int i = 0; i < Octaves; i++)
    {
        for (int j = 0; j < layers; j++)
        {
            // sigma[i* layers+j] = 1.6 * k^(j+s*(i-1));
            sigma[i * layers + j] = 1.6 * pow(2, (j + layers * (i - 1)) / 3);
        }
    }
    // 降采样
    int index = 0;
    for (int i = 0; i < Octaves - 1; i++)
    {
        pyrDown(pyr[i], pyr[i + layers], Size((pyr[i].cols + 1) / 2, (pyr[i].rows + 1) / 2));
    }
    for (int i = 0; i < Octaves; i++)
    {
        for (int j = 0; j < layers; j++)
        {
            GaussianBlur(pyr[i], pyr[i * layers + j], Size(5, 5), sigma[i * layers + j], sigma[i * layers + j]);

        }
    }
}

// 旋转
void imrotate(Mat src, Mat& dst, double angle)
{
    int w = src.cols;
    int h = src.rows;
    Point2f pt(w / 2., h / 2.);
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(src, dst, r, Size(w, h));
}
// 幂次变换
void power_transform(Mat src, Mat& dst, double gama, double scale = 1.0)
{
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)    
        {
            for (int c = 0; c < 3; c++) {
                double r = (double)src.at<Vec3b>(i, j)[c] / 255;
                dst.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(scale * pow(r, gama) * 255);
            }
        }
    }
}

//mu高斯函数的偏移，sigma高斯函数的标准差
double generateGaussianNoise(double mu, double sigma)
{
    //定义小值，numeric_limits<double>::min()是函数,返回编译器允许的double型数最小值
    const double epsilon = 1E-6;
    static double z0, z1;
    static bool flag = false;
    flag = !flag;
    //flag为假构造高斯随机变量x
    if (!flag) return z1 * sigma + mu;
    //构造随机变量
    double u1, u2;
    do
    {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);
    //flag为真构造高斯随机变量x
    z0 = sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(2 * CV_PI * u2);
    return z0 * sigma + mu;
}
// 添加高斯噪声
void addGaussianNoise(Mat src, Mat& dst)
{
    int rows = src.rows, cols = src.cols;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int c = 0; c < 3; c++) {
                //添加高斯噪声
                int val = dst.at<Vec3b>(i, j)[c] + generateGaussianNoise(2, 0.2) * 32;
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                dst.at<Vec3b>(i, j)[c] = (uchar)val;
            }
        }
    }
}
// 添加椒盐噪声
void addPeperSaltNoise(Mat src, Mat& dst)
{
    default_random_engine generator;
    uniform_int_distribution<int>randomRow(0, src.rows - 1);
    uniform_int_distribution<int>randomCol(0, src.cols - 1);

    int i, j;
    for (int k = 0; k < 500; k++)
    {
        //随机生成图形设置
        i = randomCol(generator);
        j = randomRow(generator);
        for (int c = 0; c < 3; c++) {
            dst.at<Vec3b>(j, i)[c] = 255;
        }
    }
    for (int k = 0; k < 500; k++)
    {
        //随机生成图形设置
        i = randomCol(generator);
        j = randomRow(generator);
        for (int c = 0; c < 3; c++) {
            dst.at<Vec3b>(j, i)[c] = 0;
        }
    }
}

// 计算查准率P和查全率R
void calPR(vector<Point2f> basepoints, vector<Point2f> keypoints,
    double& P, double& R, int type, int scale = 1, double alpha = 0, int height = 0, int width = 0)
{
    int TP = 0;
    int i, j;
    if (type == 1)
    {   //对于尺度缩放后的图像，将角点坐标进行相应放大，在原图对应位置查找8像素邻域内是否有角点，
        //若有则该角点视为真正值（TP），否则视为假正值（FP）

        for (i = 0; i < keypoints.size(); i++) {
            int xx = keypoints[i].x * scale;
            int yy = keypoints[i].y * scale;
            // circle(ccc, Point(xx, yy), 5, Scalar(0, 0, 255), 2, 8, 0);
            for (j = 0; j < basepoints.size(); j++) {
                if (abs(basepoints[j].x - xx) <= 5 && abs(basepoints[j].y - yy) <= 5)
                { // 查找8像素邻域内是否有角点
                    TP++;
                    break;
                }
            }
        }
        P = (1.0 * TP) / keypoints.size();
        R = (1.0 * TP) / basepoints.size();
    }
    else if (type == 2)
    {   //对于旋转后的图像，将角点坐标通过旋转变换映射到原图，映射后的坐标不一定为整数，
        //则对于映射后不是整数坐标的角点，在原图对应位置查找8像素邻域内是否有角点，
        //若有则该角点视为真正值，否则视为假正值；
        //对于映射后是整数坐标的角点，若原图此位置是角点，则视为真正值，否则视为假正值。
        float cosa = cos(alpha / 180 * CV_PI), sina = -sin(alpha / 180 * CV_PI);
        Point2f center = Point2f(1.0 * width / 2, 1.0 * height / 2);
        for (i = 0; i < keypoints.size(); i++) {
            //顺时针旋转alpha度
            float xx = cosa * (keypoints[i].x - center.x) + sina * (keypoints[i].y - center.y) + center.x;
            float yy = cosa * (keypoints[i].y - center.y) - sina * (keypoints[i].x - center.x) + center.y;

            if (xx < 0 || xx >= width) {
                if (xx < 0)    xx = 0;
                else    xx = width - 1;
            }
            if (yy < 0 || yy >= height) {
                if (yy < 0)    yy = 0;
                else    yy = height - 1;
            }

            int x = (int)xx;
            int y = (int)yy;
            //  circle(ccc, Point(x, y), 5, Scalar(0, 0, 255), 2, 8, 0);
            if (abs(xx - x) < 1E-5 && abs(yy - y) < 1E-5) {// 是整数坐标
                for (j = 0; j < basepoints.size(); j++) {
                    if (basepoints[j].x == x && basepoints[j].y == y)
                    {
                        TP++;
                        break;
                    }
                }
            }
            else {// 不是整数坐标
                for (j = 0; j < basepoints.size(); j++) {
                    if (abs(basepoints[j].x - x) <= 2 && abs(basepoints[j].y - y) <= 2)
                    { // 查找8像素邻域内是否有角点
                        TP++;
                        break;
                    }
                }
            }
        }
        waitKey(0);
        P = (1.0 * TP) / keypoints.size();
        R = (1.0 * TP) / basepoints.size();
    }

    else
    {   //对于幂次变换和加噪声的图像，变换前后图像尺寸不变
        //则对处理后图像的每个角点坐标，若原图该位置是角点，则视为真正值，否则为假正值

        for (i = 0; i < keypoints.size(); i++) {
            for (j = 0; j < basepoints.size(); j++) {
                if (abs(basepoints[j].x - keypoints[i].x) < 1 && abs(basepoints[j].y - keypoints[i].y) < 1)
                {
                    TP++;
                    break;
                }
            }
        }
        P = (1.0 * TP) / keypoints.size();
        R = (1.0 * TP) / basepoints.size();
    }
    if (P > 1)  P = 1;
    if (R > 1)  R = 1;
}

// 主函数
int main()
{
    const Mat src = imread("exp1.jpg", 1);
    if (src.empty())
    {
        printf("could not load image...\n");
        return -1;
    }
    imshow("原图", src);

    // 对原图进行SIFT角点检测
    Mat dst_SIFT = src.clone();
    vector<Point2f> dst_point;
    fSIFT(src, dst_SIFT, dst_point);
    imshow("结果", dst_SIFT);
    imwrite("1.jpg", dst_SIFT);
    waitKey(0);

    // 构建高斯金字塔进行SIFT角点检测
    Mat pyr[OCTAVE];
    Mat pyr_SIFT;
    vector<Point2f> pyr_point;
    double P[OCTAVE], R[OCTAVE];
    BuildGaussianPyramid(src, pyr);
    for (int i = 0; i < OCTAVE; i++) {
        pyr_SIFT = pyr[i].clone();
        fSIFT(pyr[i], pyr_SIFT, pyr_point);
        calPR(dst_point, pyr_point, P[i], R[i], 1, pow(2, i));
        pyr_point.clear();
        cout << "第" << i << "层的P = " << P[i] << ", R = " << R[i] << endl;
        imshow("结果", pyr_SIFT);
        imwrite("oct.jpg", pyr_SIFT);
        waitKey(0);
    }
    destroyAllWindows();

    // 进行30、45、60度旋转，对旋转图像进行SIFT角点检测
    Mat rotate30 = Mat::zeros(src.size(), CV_8UC1);
    Mat rotate45 = Mat::zeros(src.size(), CV_8UC1);
    Mat rotate60 = Mat::zeros(src.size(), CV_8UC1);
    vector<Point2f> rotate30_point;
    vector<Point2f> rotate45_point;
    vector<Point2f> rotate60_point;
    double P30, P45, P60, R30, R45, R60;
    imrotate(src, rotate30, 30);
    imrotate(src, rotate45, 45);
    imrotate(src, rotate60, 60);
    Mat rotate30_SIFT = rotate30.clone();
    Mat rotate45_SIFT = rotate45.clone();
    Mat rotate60_SIFT = rotate60.clone();
    fSIFT(rotate30, rotate30_SIFT, rotate30_point);
    fSIFT(rotate45, rotate45_SIFT, rotate45_point);
    fSIFT(rotate60, rotate60_SIFT, rotate60_point);
    calPR(dst_point, rotate30_point, P30, R30, 2, 1, 30, src.rows, src.cols);
    calPR(dst_point, rotate45_point, P45, R45, 2, 1, 45, src.rows, src.cols);
    calPR(dst_point, rotate60_point, P60, R60, 2, 1, 60, src.rows, src.cols);
    cout << "旋转30度 P = " << P30 << ", R = " << R30 << endl;
    cout << "旋转45度 P = " << P45 << ", R = " << R45 << endl;
    cout << "旋转60度 P = " << P60 << ", R = " << R60 << endl;
    imshow("30度", rotate30_SIFT);    waitKey(0);
    imshow("45度", rotate45_SIFT);    waitKey(0);
    imshow("60度", rotate60_SIFT);    waitKey(0);
    imwrite("20.jpg", rotate30_SIFT);
    imwrite("40.jpg", rotate45_SIFT);
    imwrite("60.jpg", rotate60_SIFT);
    destroyAllWindows();

    // 幂次变换后进行SIFT角点检测
    Mat lighter = src.clone();
    Mat darker = src.clone();
    vector<Point2f> lighter_point;
    vector<Point2f> darker_point;
    power_transform(src, lighter, 0.5);
    power_transform(src, darker, 1.5);
    Mat lighter_SIFT = lighter.clone();
    Mat darker_SIFT = darker.clone();
    double P_lighter, R_lighter, P_darker, R_darker;
    fSIFT(lighter, lighter_SIFT, lighter_point);
    fSIFT(darker, darker_SIFT, darker_point);
    calPR(dst_point, lighter_point, P_lighter, R_lighter, 3);
    calPR(dst_point, darker_point, P_darker, R_darker, 3);
    cout << "变亮后 P = " << P_lighter << ", R = " << R_lighter << endl;
    cout << "变暗后 P = " << P_darker << ", R = " << R_darker << endl;
    imshow("变亮", lighter_SIFT);    waitKey(0);
    imshow("变暗", darker_SIFT);    waitKey(0);
    imwrite("变亮.jpg", lighter_SIFT);
    imwrite("变暗.jpg", darker_SIFT);
    destroyAllWindows();

    // 增加高斯与椒盐噪声，对噪声图像进行SIFT角点检测
    Mat GaussianNoise = src.clone();
    Mat PeperSaltNoise = src.clone();
    vector<Point2f> GaussianNoise_point;
    vector<Point2f> PeperSaltNoise_point;
    double P_GaussianNoise, R_GaussianNoise, P_PeperSaltNoise, R_PeperSaltNoise;
    addGaussianNoise(src, GaussianNoise);
    addPeperSaltNoise(src, PeperSaltNoise);
    Mat GaussianNoise_SIFT = GaussianNoise.clone();
    Mat PeperSaltNoise_SIFT = PeperSaltNoise.clone();
    fSIFT(GaussianNoise, GaussianNoise_SIFT, GaussianNoise_point);
    fSIFT(PeperSaltNoise, PeperSaltNoise_SIFT, PeperSaltNoise_point);
    calPR(dst_point, GaussianNoise_point, P_GaussianNoise, R_GaussianNoise, 3);
    calPR(dst_point, PeperSaltNoise_point, P_PeperSaltNoise, R_PeperSaltNoise, 3);
    cout << "高斯噪声 P = " << P_GaussianNoise << ", R = " << R_GaussianNoise << endl;
    cout << "椒盐噪声 P = " << P_PeperSaltNoise << ", R = " << R_PeperSaltNoise << endl;
    imshow("高斯噪声", GaussianNoise_SIFT);
    imshow("椒盐噪声", PeperSaltNoise_SIFT);
    imwrite("高斯噪声.jpg", GaussianNoise_SIFT);
    imwrite("椒盐噪声.jpg", PeperSaltNoise_SIFT);
    waitKey(0);
    destroyAllWindows();

    return 0;
}