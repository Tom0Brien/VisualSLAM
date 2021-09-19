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

#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>

#include "settings.h"
#include "cameraModel.hpp"
#include "rotation.hpp"
#include "gaussian.hpp"


void runSLAMFromVideo(const std::filesystem::path &videoPath, const std::filesystem::path &cameraDataPath, const CameraParameters & param, Settings& s, int scenario, int interactive, const std::filesystem::path &outputDirectory);
#endif