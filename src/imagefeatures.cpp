#include <iostream>
#include <string>
#include <cstdlib>
#include <iomanip>

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>

#include "imagefeatures.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/eigen.hpp>
#include <stdlib.h>


std::vector<Feature> detected_features;

bool sortByScore(Feature const& lhs, Feature const& rhs) {
        return lhs.score > rhs.score;
}

bool sortById(Marker const& lhs, Marker const& rhs) {
        return lhs.id < rhs.id;
}

int detectAndDrawArUco(cv::Mat img, cv::Mat & imgout, std::vector<Marker> & detected_markers,const CameraParameters & param){

     std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    parameters->cornerRefinementWinSize = 10;
    parameters->minCornerDistanceRate = 0.03;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::detectMarkers(img, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.166, param.Kc, param.distCoeffs, rvecs, tvecs);


    //Add all markers to a struct
    for(int i = 0; i < markerCorners.size();i++){
        //Convert to rotation matrix
        cv::Mat Rcv;
        cv:Rodrigues(rvecs[i],Rcv);
        Eigen::MatrixXd R(3,3);
        Eigen::MatrixXd rMCc(3,1);
        //Convert to eigen type
        cv::cv2eigen(Rcv,R);
        //Convert to euler angles
        cv::cv2eigen(tvecs[i], rMCc);
        Marker temp = {markerIds[i],markerCorners[i],rMCc,R};
        detected_markers.push_back(temp);
    }

    //Sort struct by id
    std::sort(detected_markers.begin(),detected_markers.end(),&sortById);

    //Show markers detected
    imgout = img;
    cv::aruco::drawDetectedMarkers(imgout, markerCorners, markerIds);
    for (int i = 0; i < rvecs.size(); ++i) {
        auto rvec = rvecs[i];
        auto tvec = tvecs[i];
        cv::aruco::drawAxis(imgout, param.Kc, param.distCoeffs, rvec, tvec, 0.1);
    }
    return 0;
}

int detectAndDrawORB(cv::Mat img, cv::Mat & imgout, int maxNumFeatures, cv::Mat & descriptors, std::vector<cv::KeyPoint> & keypoints){
    //Create orb detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create(
        maxNumFeatures,         // nfeatures
        1.2f,                   // scaleFactor
        8,                      // nlevels
        75,                     // edgeThreshold
        0,                      // firstLevel
        2,                      // WTA_K
        cv::ORB::HARRIS_SCORE,  // scoreType
        75,                     // patchSize
        20                      // fastThreshold
    );
    // Detect the position of the Oriented FAST corner point.
    orb->detect(img, keypoints);
    // Calculate the BRIEF descriptor according to the position of the corner point
    orb->compute(img, keypoints, descriptors);
    //Draw keypoints on output image
    // drawKeypoints(img, keypoints, imgout, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    for(int i = 0; i < keypoints.size(); i++){
        cv::circle(img, cv::Point(keypoints[i].pt.x,keypoints[i].pt.y), 2, cv::Scalar(rand() % 255, rand() % 255, rand() % 255), 1, 8, 0);
    }
    imgout = img;
    return 0;

}

