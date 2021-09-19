
// #include <filesystem>
// #include <iostream>
// #include <cassert>
// #include <cmath>
// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <bitset>

// #include <Eigen/Core>

// #include <opencv2/calib3d.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/core/eigen.hpp>
// #include <opencv2/core/utility.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/videoio.hpp>

// #include "calibrate.h"

// static void write(cv::FileStorage& fs, const std::string&, const CameraParameters& x)
// {
//     x.write(fs);
// }

// void read(const cv::FileNode& node, CameraParameters& x, const CameraParameters& default_value){

//     if(node.empty()){
//         x = default_value;
//         std::cout << "Expected to be able to read a param structure" << std::endl;
//         assert(0);
//     }
//     else
//         x.read(node);
// }

// void CameraParameters::print() const
// {

//     std::bitset<8*sizeof(flag)> bitflag(flag);
//     std::cout   << "Calibration data:" << std::endl;
//     std::cout   << std::setw(30) << "cameraMatrix : " << std::endl << Kc         << std::endl;
//     std::cout   << std::setw(30) << "distCoeffs : "   << std::endl << distCoeffs << std::endl;
//     std::cout   << std::setw(30) << "flag : "                      << bitflag    << std::endl;
//     std::cout   << std::setw(30) << "imageSize : "    << std::endl << imageSize  << std::endl;
//     std::cout   << std::endl;
// }

// // Write serialization for this struct
// void CameraParameters::write(cv::FileStorage& fs) const
// {
//     fs  << "{"
//         << "camera_matrix"           << Kc
//         << "distortion_coefficients" << distCoeffs
//         << "flag"                    << flag
//         << "imageSize"               << imageSize
//         << "}";
// }

// // Read serialization for this struct
// void CameraParameters::read(const cv::FileNode& node)
// {
//     node["camera_matrix"]           >> Kc;
//     node["distortion_coefficients"] >> distCoeffs;
//     node["flag"]                    >> flag;
//     node["imageSize"]               >> imageSize;
// }

// // ------------------------------------------------------------
// // Export calibration data from a file
// // ------------------------------------------------------------
// void exportCalibrationData(const std::filesystem::path & calibrationFilePath, const CameraParameters & param){

//     // TODO
//     cv::FileStorage fs(calibrationFilePath.string(), cv::FileStorage::WRITE);
//     fs << "CalibrationData" << param;
//     fs.release();
// }

// // ------------------------------------------------------------
// // Import calibration data from a file
// // ------------------------------------------------------------
// void importCalibrationData(const std::filesystem::path & calibrationFilePath, CameraParameters & param){

//     cv::FileStorage fs;

//     fs.open(calibrationFilePath.string(), cv::FileStorage::READ);
//     assert(fs.isOpened());
//     fs["CalibrationData"] >> param;
// }

// void calibrateCameraFromVideo(const std::filesystem::path &videoPath, const std::filesystem::path &dataPath, Settings& s)
// {
//     // Calibration chessboard parameters
//     const int gridWidth = 10;
//     const int gridHeight = 7;
//     const double squareSize = 0.022;

//     // TODO
//     // - Open input video at videoPath
//     cv::VideoCapture cap(videoPath.string());

//     // Check if camera opened successfully
//     if(!cap.isOpened()){
//         std::cout << "Error opening video stream or file" << std::endl;
//         return;
//     }

//     // - Extract relevant frames containing the chessboard
//     cv::Mat view;
//     std::vector<std::vector<cv::Point2f> > rQOi_set;
//     cv::Size imageSize;
//     int num_frames = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
//     int count = 0;
//     while(cap.isOpened()){
//         cap >> view;
//         if(view.empty() || count == 120)          // If there are no more images stop the loop
//         {
//             break;
//         }
//         imageSize = view.size();  // Format input image.
//         std::vector<cv::Point2f> rQOi;

//         if (detectChessBoard(view, rQOi)){
//             rQOi_set.push_back(rQOi);
//             std::cout << "Found chess board corners in frame: " << count <<std::endl;
//             count += 1;
//         }
//         else{
//             std::cout << "Found no chess board corners in frame: " << count <<std::endl;
//         }

//         std::cout   << std::setw(30) << "No. of frames used : "            << rQOi_set.size() << " out of " << num_frames << " frames." <<std::endl;
//         std::cout   << std::setw(30) << "inputImageSize : "             << imageSize << std::endl;

//         //skip 10 frames
//         for(int j = 0; j < 10; j++) {
//             cap >> view;
//         }
//     }
//     // - Perform camera calibration
//     CameraParameters param;
//     runCalibration(s,rQOi_set,imageSize,param);
//     // - Write the camera matrix and lens distortion parameters to XML file at dataPath
//     exportCalibrationData(dataPath,param);
//     // - Visualise the camera calibration results

// }

// // ------------------------------------------------------------
// // detectChessBoard in a single frame
// // ------------------------------------------------------------
// bool detectChessBoard(const cv::Mat & view, std::vector<cv::Point2f> & rQOi ){

//     // TODO: Copy from Lab3
//     cv::Size patternsize(10,7);
//     cv::Mat img_gray;
//     cv::cvtColor(view, img_gray, cv::COLOR_BGR2GRAY);
//     bool patternfound = findChessboardCorners(img_gray, patternsize, rQOi);
//     return patternfound;
// }

// // ------------------------------------------------------------
// // run calibration on a set of detected points
// // ------------------------------------------------------------
// void runCalibration(const Settings & s, const std::vector<std::vector<cv::Point2f>> & rQOi_set,  const cv::Size & imageSize, CameraParameters & param){

//     std::vector<cv::Point3f> rPNn_base;
//     std::vector<std::vector<cv::Point3f>> rPNn;
//     generateCalibrationGrid(s, rPNn_base);

//     for(int i = 0; i < rQOi_set.size(); i++) {
//         rPNn.push_back(rPNn_base);
//     }

    
//     double rms=-1;
//     if( !rQOi_set.empty() ){
//         std::cout << "here" << std::endl;
//         // ------------------------------------------------------------
//         // Run camera calibration
//         // ------------------------------------------------------------
//         // TODO: Copy from Lab3
//         cv::Mat cameraMatrix,distCoeffs;
//         distCoeffs = cv::Mat::zeros(5,1,CV_64F);
//         std::vector<cv::Mat> R, rNCc;
//         int flag = cv::CALIB_THIN_PRISM_MODEL | cv::CALIB_RATIONAL_MODEL;
//         double rms = cv::calibrateCamera(rPNn, rQOi_set, imageSize, cameraMatrix, distCoeffs, R, rNCc, flag);
//         std::cout << "Calibration rms: " << rms << std::endl;
//         // ------------------------------------------------------------
//         // Write parameters to the data struct 
//         // ------------------------------------------------------------
//         param.Kc = cameraMatrix;
//         param.distCoeffs = distCoeffs;
//         param.flag = flag;
//         param.imageSize = imageSize;


//     } else{
//         std::cerr << "No imagePoints found" << std::endl;
//         assert(0);
//     }
// }

// // ------------------------------------------------------------
// // Define camera calibration grid
// // ------------------------------------------------------------
// void generateCalibrationGrid(const Settings & s, std::vector<cv::Point3f> & rPNn_grid){
//     // Defining the world coordinates for 3D points
//     for(int i=0; i < s.boardSize.height; i++) {
//         for(int j=0; j < s.boardSize.width; j++){
//             rPNn_grid.push_back(cv::Point3f(j*s.squareSize,i*s.squareSize,0));
//         }
//     }
// }
