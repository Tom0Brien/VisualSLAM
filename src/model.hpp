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
    double position_tune;
    double orientation_tune;
    double measurement_noise;
    std::vector<Landmark> landmarks;
};

struct SlamProcessModel
{
    // double operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f, Eigen::MatrixXd & SQ);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f, Eigen::MatrixXd & SQ, Eigen::MatrixXd &dfdx);
};

// struct SlamMeasurementModel
// {
//     void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & h);
//     void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & h, Eigen::MatrixXd & SR);
//     void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & h, Eigen::MatrixXd & SR, Eigen::MatrixXd & dhdx);
// };

struct ArucoLogLikelihood
{
    double operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g, Eigen::MatrixXd &H);
};

struct PointLogLikelihood
{
    double operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g, Eigen::MatrixXd &H);
};

struct arucoLogLikelihoodAnalytical
{
    double operator()(const Eigen::VectorXd y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g, Eigen::MatrixXd &H);
};

// double slamLogLikelihoodAnalytical(const Eigen::VectorXd y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g, Eigen::MatrixXd &H);

#endif