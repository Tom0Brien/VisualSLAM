#ifndef IMAGEFEATURES_H
#define IMAGEFEATURES_H 

#include <opencv2/core.hpp>
#include "cameraModel.hpp"

struct Marker {
    int id;
    std::vector<cv::Point2f> corners;
    Eigen::MatrixXd rMCc;
    Eigen::MatrixXd Rcm;
};

struct Feature {
    int x;
    int y;
    float score;
};



int detectAndDrawHarris(cv::Mat img, cv::Mat & imgout, int maxNumFeatures);
int detectAndDrawShiAndTomasi(cv::Mat img, cv::Mat & imgout, int maxNumFeatures);
int detectAndDrawArUco(cv::Mat img, cv::Mat & imgout, std::vector<Marker> & detected_markers,const CameraParameters & param);
int detectAndDrawORB(cv::Mat img, cv::Mat & imgout, int maxNumFeatures);


#endif