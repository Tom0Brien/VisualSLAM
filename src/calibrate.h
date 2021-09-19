// #ifndef CALIBRATE_H
// #define CALIBRATE_H

// #include <filesystem>

// #include <Eigen/Core>

// #include <opencv2/calib3d.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/core/eigen.hpp>
// #include <opencv2/core/utility.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/videoio.hpp>

// #include "settings.h"


// struct CameraParameters{
//     cv::Mat Kc;                 // Camera Matrix
//     cv::Mat distCoeffs;         // Distortion coefficients
//     int flag            = 0;    // Calibration flag
//     double fieldOfView  = 150;  // Describe the arc of the view cone
//     cv::Size imageSize;         // Image size
//     // Read and write methods required for class serialisation
//     void read(const cv::FileNode & node);
//     void write(cv::FileStorage& fs) const;
//     // Convenience function
//     void print() const;
// };


// void calibrateCameraFromVideo(const std::filesystem::path &videoPath, const std::filesystem::path &dataPath, Settings& s);
// bool detectChessBoard(const cv::Mat & view, std::vector<cv::Point2f> & rQOi );
// void runCalibration(const Settings & s, const std::vector<std::vector<cv::Point2f>> & rQOi_set,  const cv::Size & imageSize, CameraParameters & param);
// void generateCalibrationGrid(const Settings & s, std::vector<cv::Point3f> & rPNn_grid);
// void importCalibrationData(const std::filesystem::path & calibrationFilePath, CameraParameters & param);
// #endif