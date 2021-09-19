#ifndef ROTATION_HPP
#define ROTATION_HPP

#include <Eigen/Core>
#include <iostream>

using std::cos;
using std::sin;
using std::sqrt;
using std::pow;
using std::atan2;

template<typename Scalar>
void rot2rpy(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>  & Rnc, Eigen::Matrix<Scalar, Eigen::Dynamic, 1> & Thetanc){
    assert(Rnc.rows() == 3);
    assert(Rnc.cols() == 3);
    Thetanc.resize(3,1);
    Thetanc << atan2(Rnc(2,1), Rnc(2,2)), 
               atan2(-Rnc(2,0),sqrt(pow(Rnc(2,1),2)+pow(Rnc(2,2),2))), atan2(Rnc(1,0),Rnc(0,0));
}

template<typename Scalar>
void rotx(const Scalar & x, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> & Rnc){
    
    // // TODO:
    Rnc.resize(3,3);
    Rnc << 1,0,0,
    0,cos(x),-sin(x),
    0,sin(x),cos(x);
}

template<typename Scalar>
void roty(const Scalar & x, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> & Rnc){
    
    // // TODO:
    Rnc.resize(3,3);
    Rnc << cos(x), 0,sin(x),
           0,1,0,
          -sin(x), 0, cos(x);
}

template<typename Scalar>
void rotz(const Scalar & x, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> & Rnc){
    
    // TODO:
    Rnc.resize(3,3);
    Rnc << cos(x), -sin(x), 0,
            sin(x), cos(x), 0,
            0, 0, 1;
}

template<typename Scalar>
void rpy2rot(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> & Thetanc, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> & Rnc){
    // TODO: 
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Rx;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Ry;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Rz;
    rotz(Thetanc(2),Rz);
    roty(Thetanc(1),Ry);
    rotx(Thetanc(0),Rx);
    Rnc.resize(3,3);
    Rnc = Rz*Ry*Rx;
}




#endif