#include "measurementPointLandmark.hpp"
#include <iostream>


#include <Eigen/Core>
#include <Eigen/QR>

// #include <autodiff/forward/dual.hpp>
// #include <autodiff/forward/dual/eigen.hpp>



void MeasurementPointLandmarkBundle::operator()(const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd & rQOi){
    assert(x.cols() == 1);
    const int nCameraStates     = 12;
    const int featureDim        = 3;
    int nLandmarkStates         = x.rows() - nCameraStates;
    assert(nLandmarkStates%3 == 0);
    int nLandmarks              = nLandmarkStates / featureDim;

    // TODO
    Eigen::VectorXd rQOi_temp;
    MeasurementPointLandmarkSingle h;
    std::cout << "FUCK FUCK FUCK 1" << std::endl;
    rQOi.resize(nLandmarks*2,1);
    for(int j; j < nLandmarks; j++){
        h(j,x,param,rQOi_temp);
        rQOi.segment(j*2,2) = rQOi_temp;
    }

}
void MeasurementPointLandmarkBundle::operator()(const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR){
    assert(x.cols() == 1);
    const int nCameraStates     = 12;
    const int featureDim        = 3;
    int nLandmarkStates         = x.rows() - nCameraStates;
    assert(nLandmarkStates%3 == 0);
    int nLandmarks              = nLandmarkStates / featureDim;
    // TODO
    operator()(x,param,rQOi);
    std::cout << "FUCK FUCK FUCK 2" << std::endl;
    SR.derived().resize(nLandmarks*2,nLandmarks*2);
    SR.setIdentity();
    SR = 0.01 * SR;
}
void MeasurementPointLandmarkBundle::operator()(const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR, Eigen::MatrixXd & C){
    assert(x.cols() == 1);
    const int nCameraStates     = 12;
    const int featureDim        = 3;
    std::cout << "x.rows()" << x.rows() << std::endl;
    int nLandmarkStates         = x.rows() - nCameraStates;
    assert(nLandmarkStates%3 == 0);
    int nLandmarks              = nLandmarkStates / featureDim;
    // TODO
    Eigen::VectorXd rQOi_temp;
    Eigen::MatrixXd J_temp;
    J_temp.resize(2,nCameraStates+nLandmarkStates);
    C.resize(nLandmarks*2, nCameraStates+nLandmarkStates);
    C.setZero();

    MeasurementPointLandmarkSingle h;
    rQOi.resize(nLandmarks*2,1);

    for(int j=0; j < nLandmarks; j++){
        h(j,x,param,rQOi_temp,SR,J_temp);
        rQOi.segment(j*2,2) = rQOi_temp;

        std::cout << "J_temp" << J_temp << std::endl;
        std::cout << "J_temp.rows()" << J_temp.rows() << std::endl;
        std::cout << "J_temp.cols()" << J_temp.cols() << std::endl;
        C.derived().block(j*2,0,2,nCameraStates+nLandmarkStates) = J_temp;
    }

    SR.derived().resize(nLandmarks*2,nLandmarks*2);
    SR.setIdentity();
    SR = 0.01 * SR;
}
