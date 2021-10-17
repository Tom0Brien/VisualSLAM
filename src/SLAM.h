#ifndef SLAM_H
#define SLAM_H

#include <filesystem>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/aruco.hpp>

#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>

#include "settings.h"
#include "cameraModel.hpp"
#include "rotation.hpp"
#include "gaussian.hpp"


#include <iostream>
#include <chrono>


#include "imagefeatures.h"
#include "plot.h"
#include "cameraModel.hpp"
#include "model.hpp"


void runSLAMFromVideo(const std::filesystem::path &videoPath, const std::filesystem::path &cameraDataPath, const CameraParameters & param, Settings& s, int scenario, int interactive, const std::filesystem::path &outputDirectory);
cv::Mat getPlotFrame(const PlotHandles &handles);
void plot(cv::Mat & imgout, Eigen::MatrixXd & muPlot, Eigen::MatrixXd & SPlot, SlamParameters & slamparam, CameraParameters & param, int interactive, cv::VideoWriter & video);
bool pixelDistance(std::vector<Landmark> & landmarks, cv::KeyPoint keypoint, double pixel_distance_thresh);
bool pixelDistance(std::vector<Landmark> & landmarks, cv::KeyPoint keypoint, double pixel_distance_thresh, int j);
void removeBadLandmarks(Eigen::VectorXd & mup, Eigen::MatrixXd & Sp, std::vector<Landmark> & landmark, int j);
void removeRow(Eigen::VectorXd & vector, unsigned int rowToRemove);
void removeColumn(Eigen::MatrixXd & matrix, unsigned int colToRemove);
void  combineDescriptors(cv::Mat & landmark_descriptors, std::vector<Landmark> & landmarks);
Eigen::MatrixXd generateOpticalRay(const CameraParameters & param, Eigen::MatrixXd & pixel);
#endif