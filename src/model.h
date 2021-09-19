#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Core>

#include "imagefeatures.h"

struct SlamParameters
{

};

struct SlamProcessModel
{
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f, Eigen::MatrixXd & SQ);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f, Eigen::MatrixXd & SQ, Eigen::MatrixXd & dfdx);
};

struct SlamMeasurementModel
{
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & h);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & h, Eigen::MatrixXd & SR);
    void operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & h, Eigen::MatrixXd & SR, Eigen::MatrixXd & dhdx);
};


#endif