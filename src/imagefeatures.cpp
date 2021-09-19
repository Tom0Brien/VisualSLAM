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



std::vector<Feature> detected_features;

bool sortByScore(Feature const& lhs, Feature const& rhs) {
        return lhs.score > rhs.score;
}

bool sortById(Marker const& lhs, Marker const& rhs) {
        return lhs.id < rhs.id;
}

// int detectAndDrawHarris(cv::Mat img, cv::Mat & imgout, int maxNumFeatures){
//     // Print some stuff
//     std::cout << "Using harris feature detector" << std::endl;
//     std::cout << "Width : " << img.cols << std::endl;
//     std::cout << "Height: " << img.rows << std::endl;
//     std::cout << "Features requested: " << maxNumFeatures << std::endl;
//     // Initialize variables
//     int blockSize = 2;
//     int apertureSize = 3;
//     double k = 0.04;
//     int thresh = 175;
//     int max_thresh = 255;
//     int num_features_detected = 0;   
//     const char* corners_window = "Corners detected";

//     // Convert to gray
//     cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);
//     cv::Mat img_gray;
//     cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
//     cv::cornerHarris(img_gray, dst, blockSize, apertureSize, k);

//     //Normalize
//     cv::Mat dst_norm, dst_norm_scaled;
//     cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
//     cv::convertScaleAbs(dst_norm, dst_norm_scaled);
//     for(int i = 0; i < dst_norm.rows ; i++)
//     {
//         for(int j = 0; j < dst_norm.cols; j++)
//         {
//             if((int) dst_norm.at<float>(i,j) > thresh)
//             {
//                 num_features_detected++;
//                 features new_feature = {i,j,dst.at<float>(i,j)};
//                 detected_features.push_back(new_feature);
//                 cv::circle(imgout, cv::Point(j,i), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
//             }
//         }
//     }

//     std::cout << "Features detected: " << num_features_detected << std::endl;    
//     //Sort struct
//     std::sort(detected_features.begin(), detected_features.end(), &sortByScore);
//     for(int i = 0; i < maxNumFeatures; i++){
//         std::cout << "Idx: " << i << "    at point:" << "(" << detected_features[i].x << "," << detected_features[i].y << ")" << "    Harris score: " << detected_features[i].score << std::endl;
//         cv::circle(imgout, cv::Point(detected_features[i].y,detected_features[i].x), 5, cv::Scalar(0, 0, 255), 5, 8, 0);
//         cv::putText(imgout,"Id="+std::to_string(i),cv::Point(detected_features[i].y,detected_features[i].x),cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(100, 255, 255),2);
//     }
//     cv::putText(imgout,"Ids for the "+std::to_string(maxNumFeatures)+" most responsive features found in the image",cv::Point(10,15),cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 100, 200),2);


//     // Show image and detected
//     cv::namedWindow(corners_window);
//     cv::imshow(corners_window, imgout);
//     int wait = cv::waitKey(0);
//     return 0;
    
// }
// int detectAndDrawShiAndTomasi(cv::Mat img, cv::Mat & imgout, int maxNumFeatures){   
//     // Print some stuff
//     std::cout << "Using Shi & Tomasi corner detector" << std::endl;
//     std::cout << "Width : " << img.cols << std::endl;
//     std::cout << "Height: " << img.rows << std::endl;
//     std::cout << "Features requested: " << maxNumFeatures << std::endl;
//     // Initialize variables
//     int blockSize = 3;
//     double k = 0.05;
//     float thresh = 0.35;
//     int num_features_detected = 0;   
//     const char* corners_window = "Corners detected";
    

//     // Convert to gray
//     cv::Mat min_eigen_values     = cv::Mat::zeros(img.size(), CV_32FC1);
//     cv::Mat img_gray, normalized, normalized_scaled;
//     cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
//     cv::cornerMinEigenVal(img_gray, min_eigen_values, blockSize, k);
//     // cv::normalize(min_eigen_values, normalized, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
//     // cv::convertScaleAbs(normalized, normalized_scaled);

//     for(int i = 0; i < min_eigen_values.rows ; i++)
//     {
//         for(int j = 0; j < min_eigen_values.cols; j++)
//         {
//             if((float) min_eigen_values.at<float>(i,j) > thresh)
//             {
//                 num_features_detected++;
//                 features new_feature = {i,j,min_eigen_values.at<float>(i,j)};
//                 detected_features.push_back(new_feature);
//                 cv::circle(imgout, cv::Point(j,i), 5, cv::Scalar(0, 255, 0), 2, 8, 0);
//             }
//         }
//     }

//     std::cout << "Features detected: " << num_features_detected << std::endl;    
//     //Sort struct
//     std::sort(detected_features.begin(), detected_features.end(), &sortByScore);
//     for(int i = 0; i < maxNumFeatures; i++){
//         std::cout << "Idx: " << i << "    at point:" << "(" << detected_features[i].x << "," << detected_features[i].y << ")" << "    Min eigen val: " << detected_features[i].score << std::endl;
//         cv::circle(imgout, cv::Point(detected_features[i].y,detected_features[i].x), 5, cv::Scalar(0, 0, 255), 5, 8, 0);
//         cv::putText(imgout,"Id="+std::to_string(i),cv::Point(detected_features[i].y,detected_features[i].x),cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 100, 200),2);
//         }

//     cv::putText(imgout,"Ids for the "+std::to_string(maxNumFeatures)+" most responsive features found in the image",cv::Point(10,10),cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 100, 200),2);


//     // Show image and detected
//     cv::namedWindow(corners_window);
//     cv::imshow(corners_window, imgout);
//     int wait = cv::waitKey(0);
//     return 0;
    
// }
int detectAndDrawArUco(cv::Mat img, cv::Mat & imgout, std::vector<Marker> & detected_markers,const CameraParameters & param){
    std::cout << "Using Marker detector" << std::endl;
    std::cout << "Width : " << img.cols << std::endl;
    std::cout << "Height: " << img.rows << std::endl;

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::detectMarkers(img, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.166, param.Kc, param.distCoeffs, rvecs, tvecs);

    for(int i = 0; i < markerCorners.size(); i++){
        std::cout << "tvecs" << tvecs[i] << std::endl;
        std::cout << "rvecs" << rvecs[i] << std::endl;
    }

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

    //Print out sorted markers
    for(int i = 0; i < markerCorners.size();i++){
        std::cout << "ID: " << detected_markers[i].id << "   with corners: " 
        << detected_markers[i].corners[0] << "," << detected_markers[i].corners[1] << ","
        << detected_markers[i].corners[2] << ","<< detected_markers[i].corners[3] << std::endl;
    }

    //Show markers detected
    // const char* detected_markers = ;
    imgout = img;
    cv::aruco::drawDetectedMarkers(imgout, markerCorners, markerIds);
    for (int i = 0; i < rvecs.size(); ++i) {
        auto rvec = rvecs[i];
        auto tvec = tvecs[i];
        cv::aruco::drawAxis(imgout, param.Kc, param.distCoeffs, rvec, tvec, 0.1);
    }
    // cv::imshow("Corners detected", imgout);
    // int wait = cv::waitKey(1);
    return 0;
}
int detectAndDrawORB(cv::Mat img, cv::Mat & imgout, int maxNumFeatures){
    
    std::cout << "Using ORB feature detector" << std::endl;
    std::cout << "Width : " << img.cols << std::endl;
    std::cout << "Height: " << img.rows << std::endl;
    std::cout << "Features requested: " << maxNumFeatures << std::endl;

    //Create orb detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create(
        maxNumFeatures,         // nfeatures
        1.2f,                   // scaleFactor
        8,                      // nlevels
        31,                     // edgeThreshold
        0,                      // firstLevel
        2,                      // WTA_K
        cv::ORB::HARRIS_SCORE,  // scoreType
        31,                     // patchSize
        20                      // fastThreshold 
    );

    //Create array to store keypoints
    std::vector<cv::KeyPoint> keypoints;

    //Create descriptors?
    cv::Mat descriptors;

    // Detect the position of the Oriented FAST corner point.
    orb->detect(img, keypoints);

    // Calculate the BRIEF descriptor according to the position of the corner point
    orb->compute(img, keypoints, descriptors);

    //Print some stuff
    std::cout << "Descriptor Width:" << descriptors.cols << std::endl;
    std::cout << "Descriptor Height:" << descriptors.rows << std::endl;
    for(int i = 0;i < descriptors.rows;i++){
        std::cout << "Keypoint " << i << " description:" << descriptors.row(i) << std::endl; 
    }

    //Draw keypoints on output image
    drawKeypoints(img, keypoints, imgout, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    const char* orb_features = "Corners detected";
    cv::namedWindow(orb_features);
    cv::imshow(orb_features,imgout);
    int wait = cv::waitKey(0);
    return 0;
    
}

