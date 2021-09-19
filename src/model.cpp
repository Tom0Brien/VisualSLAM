
#include "model.h"
#include "rotation.hpp"
#include <iostream>

// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// 
// SlamProcessModel
// 
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
void SlamProcessModel::operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f)
{
    // TODO: mean function, x = [nu;eta;landmarks]
    Eigen::MatrixXd J(6,6);
    J.setZero();
    double phi = x(3);
    double theta = x(4);
    double psi = x(5);
    Eigen::MatrixXd Rz; rotz(psi, Rz);
    Eigen::MatrixXd Ry; rotz(theta, Ry);
    Eigen::MatrixXd Rx; rotz(phi, Rx);
    Eigen::MatrixXd Rnc(3,3);
    Rnc = Rz*Ry*Rx;
    Eigen::MatrixXd T(3,3);
    T << 1, std::sin(phi)*std::tan(theta),std::cos(phi)*std::tan(theta),
        0,std::cos(phi),-std::sin(phi),
        0,std::sin(phi)/std::cos(theta),std::cos(phi)/std::cos(theta);
    J.block(0,0,3,3) = Rnc;
    J.block(3,3,3,3) = T;

    f.resize(x.rows(),1);   
    f.setZero();
    Eigen::MatrixXd nu(6,1);
    nu = x.segment(6,6);
    f.block(6,0,6,1) = J*nu;

    
}

void SlamProcessModel::operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f, Eigen::MatrixXd & SQ)
{
    operator()(x,u,param,f);
    
    SQ = 0.1*Eigen::MatrixXd::Identity(x.rows(),x.rows());
}

void SlamProcessModel::operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f, Eigen::MatrixXd & SQ, Eigen::MatrixXd & dfdx)
{
    operator()(x,u,param,f,SQ);
    dfdx.resize(x.rows(),x.rows());
    dfdx.setZero();
}


// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// 
// SlamMeasurementModel
// 
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------

void SlamMeasurementModel::operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & h)
{
    // TODO: mean function
    int nx = 12;
    h = Eigen::MatrixXd::Zero(x.rows()-nx,1);
    h.block(0,0,x.rows()-nx,1) = x.block(nx,0,x.rows()-nx,1);
}

void SlamMeasurementModel::operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & h, Eigen::MatrixXd & SR)
{
    operator()(x,u,param,h);
    int nx = 12;
    SR.resize(x.rows()-nx,x.rows()-nx);
    double tune = 0.1;
    SR = tune*Eigen::MatrixXd::Identity(x.rows()-nx,x.rows()-nx);

}

void SlamMeasurementModel::operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & h, Eigen::MatrixXd & SR, Eigen::MatrixXd & dhdx)
{
    operator()(x,u,param,h,SR);
    int nx = 12;
    dhdx.resize(x.rows()-nx, x.rows()-nx);
    dhdx = Eigen::MatrixXd::Identity(x.rows()-nx,x.rows()-nx);


}







