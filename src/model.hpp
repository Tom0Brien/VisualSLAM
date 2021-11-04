#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Core>


#include "imagefeatures.h"
#include "cameraModel.hpp"
#include "rotation.hpp"
#include "gaussian.hpp"

struct SlamParameters
{
    CameraParameters camera_param;
    std::vector<int> landmarks_seen;
    int n_landmark;
    Eigen::MatrixXd position_tune;
    Eigen::MatrixXd orientation_tune;
    double measurement_noise;
    std::vector<Landmark> landmarks;
    std::vector<Duck> ducks;

    int max_landmarks; // Maximum number of landmarks
    double kappa; // Initial covariance for new landmarks
    double optical_ray_length; // Length of point on optical ray that landmarks are initialized
    int max_features; // Maximum number of features the feature detector obtains
    int max_bad_frames; // Maximum number of frames before a landmark is deleted
    double feature_thresh; // Threshold to initialize a new landmark
    double initial_pixel_distance_thresh; // Distance a landmark needs to be from all other landmarks to be initialized
    double update_pixel_distance_thresh; // Distance a landmark needs to be from all other landmarks to be updated
    double initial_width_thresh; // Distance a landmark needs to be from border of screen to be initialized
    double initial_height_thresh; // Distance a landmark needs to be from border of screen to be initialized
    int camera_states;
};

struct SlamProcessModel
{
    // double operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f, Eigen::MatrixXd & SQ);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f, Eigen::MatrixXd & SQ, Eigen::MatrixXd &dfdx);
};

struct ArucoLogLikelihood
{
    double operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param);
    double operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g);
    double operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g, Eigen::MatrixXd &H);
};

struct PointLogLikelihood
{
    double operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param);
    double operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g);
    double operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g, Eigen::MatrixXd &H);
};

struct DuckLogLikelihood
{
    double operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param);
    double operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g);
    double operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g, Eigen::MatrixXd &H);
};

struct arucoLogLikelihoodAnalytical
{
    double operator()(const Eigen::VectorXd y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g);
    double operator()(const Eigen::VectorXd y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g, Eigen::MatrixXd &H);
};

// double slamLogLikelihoodAnalytical(const Eigen::VectorXd y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g, Eigen::MatrixXd &H);

#endif