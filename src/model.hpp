#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Core>


#include "imagefeatures.h"
#include "cameraModel.hpp"

struct SlamParameters
{
    CameraParameters camera_param;
    std::vector<int> landmarks_seen;
};

struct SlamProcessModel
{
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f, Eigen::MatrixXd & SQ);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f, Eigen::MatrixXd & SQ, Eigen::MatrixXd & dfdx);
};

// struct SlamMeasurementModel
// {
//     void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & h);
//     void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & h, Eigen::MatrixXd & SR);
//     void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & h, Eigen::MatrixXd & SR, Eigen::MatrixXd & dhdx);
// };

struct SlamLogLikelihood
{
    double operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param);
    double operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g);
    double operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g, Eigen::MatrixXd &H);
};


#endif