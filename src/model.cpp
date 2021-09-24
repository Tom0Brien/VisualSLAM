
#include "model.hpp"
#include <iostream>

#include<opencv2/opencv.hpp>

// #include <autodiff/forward/dual.hpp>
// #include <autodiff/forward/dual/eigen.hpp>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

using namespace autodiff;

using std::sqrt;

void SlamProcessModel::operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f)
{
    // TODO: mean function, x = [nu;eta;landmarks]
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> J(6,6);
    J.setZero();

    // Rnc
    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanc(3,1);
    Thetanc = x.block(9,0,3,1);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnc(3,3);
    rpy2rot(Thetanc,Rnc);

    // Kinematic transform T(nu)
    double phi   = Thetanc(0);
    double theta = Thetanc(1);
    double psi   = Thetanc(2);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> T(3,3);
    using std::tan;
    using std::cos;
    using std::sin;
    T << 1, sin(phi)*tan(theta),cos(phi)*tan(theta),
        0,cos(phi),-sin(phi),
        0,sin(phi)/cos(theta),cos(phi)/cos(theta);

    J.block(0,0,3,3) = Rnc;
    J.block(3,3,3,3) = T;

    f.resize(x.rows(),1);
    f.setZero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> nu(6,1);
    nu = x.block(0,0,6,1);
    f.block(6,0,6,1) = J*nu;


}

void SlamProcessModel::operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f, Eigen::MatrixXd & SQ)
{
    operator()(x,u,param,f);
    SQ.resize(x.rows(),x.rows());
    SQ.setZero();
    double position_tune = 1e3;
    double orientation_tune = 1;
    double landmark_tune = 0;
    SQ.block(0,0,3,3) = position_tune*Eigen::MatrixXd::Identity(3,3);
    SQ.block(3,3,3,3) = orientation_tune*Eigen::MatrixXd::Identity(3,3);
}


void SlamProcessModel::operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f, Eigen::MatrixXd & SQ, Eigen::MatrixXd &dfdx)
{
    int nx = x.rows();
    int nj = (nx - 12)/6;
    operator()(x,u,param,f,SQ);
    dfdx.resize(nx,12+6*nj);
    dfdx.setZero();

    Eigen::MatrixXd J(6,6);
    J.setZero();

    Eigen::MatrixXd eta(6,1);
    eta = x.block(6,0,6,1);
    Eigen::MatrixXd nu(6,1);
    nu  = x.block(0,0,6,1);


    double x1       = eta(0);
    double x2       = eta(1);
    double x3       = eta(2);
    double phi      = eta(3);
    double theta    = eta(4);
    double psi      = eta(5);

    double x1dot    = nu(0);
    double x2dot    = nu(1);
    double x3dot    = nu(2);
    double phidot   = nu(3);
    double thetadot = nu(4);
    double psidot   = nu(5);

    // Rnc
    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanc(3,1);
    Thetanc = x.block(9,0,3,1);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnc(3,3);
    rpy2rot(Thetanc,Rnc);

    Eigen::MatrixXd T(3,3);
    using std::tan;
    using std::cos;
    using std::sin;

    T << 1, sin(phi)*tan(theta),cos(phi)*tan(theta),
        0,cos(phi),-sin(phi),
        0,sin(phi)/cos(theta),cos(phi)/cos(theta);

    J.block(0,0,3,3) = Rnc;
    J.block(3,3,3,3) = T;

    dfdx.block(6,0,6,6) = J;

    Eigen::MatrixXd dfdnu(6,6);
    dfdnu.setZero();

    using std::sin;
    using std::cos;
    using std::tan;
    using std::pow;

    dfdnu(0,3) = x2dot*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)) + x3dot*(cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(theta));
    dfdnu(1,3) = -x2dot*(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta)) - x3dot*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta));
    dfdnu(2,3) = x2dot*cos(phi)*cos(theta) - x3dot*cos(theta)*sin(phi);
    dfdnu(3,3) = thetadot*cos(phi)*tan(theta) - psidot*sin(phi)*tan(theta);
    dfdnu(4,3) = -psidot*cos(phi) - thetadot*sin(phi);
    dfdnu(5,3) = (thetadot*cos(phi))/cos(theta) - (psidot*sin(phi))/cos(theta);

    dfdnu(0,4) = x3dot*cos(phi)*cos(psi)*cos(theta) - x1dot*cos(psi)*sin(theta) + x2dot*cos(psi)*cos(theta)*sin(phi);
    dfdnu(1,4) = x3dot*cos(phi)*cos(theta)*sin(psi) - x1dot*sin(psi)*sin(theta) + x2dot*cos(theta)*sin(phi)*sin(psi);
    dfdnu(2,4) = - x1dot*cos(theta) - x3dot*cos(phi)*sin(theta) - x2dot*sin(phi)*sin(theta);
    dfdnu(3,4) = psidot*cos(phi)*(pow(tan(theta),2) + 1) + thetadot*sin(phi)*(pow(tan(theta),2) + 1);
    dfdnu(4,4) = 0;
    dfdnu(5,4) = (psidot*cos(phi)*sin(theta))/pow(cos(theta),2) + (thetadot*sin(phi)*sin(theta))/pow(cos(theta),2);

    dfdnu(0,5) = x3dot*(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta)) - x2dot*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta)) - x1dot*cos(theta)*sin(psi);
    dfdnu(1,5) = x3dot*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)) - x2dot*(cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(theta)) + x1dot*cos(psi)*cos(theta);

    dfdx.block(6,6,6,6) = dfdnu;

}

// Templated version of SlamLogLikelihood
// Note: templates normally should live in a template header (.hpp), but
//       since all instantiations of this template are used only in this
//       compilation unit, its definition can live here
template <typename Scalar>
double slamLogLikelihood(const Eigen::Matrix<Scalar,Eigen::Dynamic,1> y, const Eigen::Matrix<Scalar,Eigen::Dynamic,1> & x, const Eigen::Matrix<Scalar,Eigen::Dynamic,1> & u, const SlamParameters & param)
{

    // Variables
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> rJNn(3,1);
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> eta(3,1);
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> Thetanj(3,1);
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Rnj(3,3);
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> rJcNn(3,1);
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> measurement_pixel(2,1);
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> state_pixel(2,1);

    // Corner local measurements
    Scalar length = 0.166;
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> rJcJj(3,4);
    rJcJj << -length/2,length/2,length/2,-length/2,length/2,length/2,-length/2,-length/2,0,0,0,0;

    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> SR = 2*Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(2,2);

    Scalar cost = 0;
    eta = x.segment(6,6);
    //For each landmark seenauto t_start = std::chrono::high_resolution_clock::now();
    for(int l = 0; l < param.landmarks_seen.size(); l++) {
        // *** State Predicted Landmark Location *** //
        rJNn = x.segment(12+param.landmarks_seen[l]*6,3);
        Thetanj = x.segment(12+3+param.landmarks_seen[l]*6,3);
        rpy2rot(Thetanj,Rnj);
        // For each corner of the landmark
        for(int c = 0; c < 4; c++) {
            // *** State Predicted Corner Pixel *** //
            rJcNn = Rnj*rJcJj.col(c) + rJNn;
            int w2p_flag = worldToPixel(rJcNn,eta,param.camera_param,state_pixel);
            // *** Measurement Corner Pixel ***//
            measurement_pixel = y.segment(8*l + 2*c,2);
            // Sum up log gausian for each measurement
            if(w2p_flag == 0) {
                cost += logGaussian(measurement_pixel,state_pixel, SR);
            }
        }
    }
    return cost;
}



// double SlamLogLikelihood::operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param)
// {
//     // Evaluate log N(y;h(x),R)
//     return slamLogLikelihood(y, x, u, param);
// }


// double SlamLogLikelihood::operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g)
// {
//     Eigen::Matrix<autodiff::dual,Eigen::Dynamic,1> xdual = x.cast<autodiff::dual>();
//     autodiff::dual fdual;
//     auto t_start = std::chrono::high_resolution_clock::now();
//     g = autodiff::gradient(slamLogLikelihood<autodiff::dual>, wrt(xdual), at(y,xdual,u,param), fdual);
//     auto t_end = std::chrono::high_resolution_clock::now();
//     double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
//     // std::cout << "Time taken for gradient calc [s]: " << elapsed_time_ms/1000 << std::endl;
//     return val(fdual);
// }

// double SlamLogLikelihood::operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g, Eigen::MatrixXd &H)
// {
//     Eigen::Matrix<autodiff::dual2nd,Eigen::Dynamic,1> xdual = x.cast<autodiff::dual2nd>();
//     autodiff::dual2nd fdual;
//     auto t_start = std::chrono::high_resolution_clock::now();
//     H = autodiff::hessian(slamLogLikelihood<autodiff::dual2nd>, wrt(xdual), at(y,xdual,u,param), fdual, g);
//     auto t_end = std::chrono::high_resolution_clock::now();
//     double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
//     std::cout << "Time taken for hessian calc [s]: " << elapsed_time_ms/1000 << std::endl;
//     std::cout << "length of y: " << y.size() << std::endl;
//     std::cout << "length of x: " << x.size() << std::endl;
//     return val(fdual);
// }


double SlamLogLikelihood::operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g, Eigen::MatrixXd &H)
{
    Eigen::Matrix<autodiff::var,Eigen::Dynamic,1>  xvar = x.cast<var>();
    var fvar = slamLogLikelihood(y,x,u,param);
    H = hessian(fvar, xvar, g);
    return val(fvar);
}







