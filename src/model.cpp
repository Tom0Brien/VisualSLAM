
#include "model.hpp"
#include <iostream>



// #include <autodiff/reverse/var.hpp>
// #include <autodiff/reverse/var/eigen.hpp>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

// #include <autodiff/forward.hpp>
// #include <autodiff/forward/eigen.hpp>

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
    double phi = Thetanc(0);
    double theta =Thetanc(1);
    double psi = Thetanc(2);

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
    double tune = 0.05;
    SQ.resize(x.rows(),x.rows());
    SQ.setIdentity();
    double camera_tune = 1;
    double landmark_tune = 0;
    SQ.block(0,0,6,6) = camera_tune*Eigen::MatrixXd::Identity(6,6);
    SQ.block(6,6,x.rows()-6,x.rows()-6) = landmark_tune*Eigen::MatrixXd::Identity(x.rows()-6,x.rows()-6);
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

    // std::cout << "dfdx: "  << dfdx  << std::endl;
    // std::cout << "dfdx.rows(): "  << dfdx.rows()  << std::endl;
    // std::cout << "dfdx.cols(): "  << dfdx.cols()  << std::endl;

}

// Templated version of SlamLogLikelihood
// Note: templates normally should live in a template header (.hpp), but
//       since all instantiations of this template are used only in this
//       compilation unit, its definition can live here
template <typename Scalar>
static Scalar slamLogLikelihood(const Eigen::Matrix<Scalar,Eigen::Dynamic,1> y, const Eigen::Matrix<Scalar,Eigen::Dynamic,1> & x, const Eigen::Matrix<Scalar,Eigen::Dynamic,1> & u, const SlamParameters & param)
{

    // y are corner pixels [nj*2*4, 1]
    // std::cout << "y : " << y << std::endl;
    // Evaluate log N(y;h(x),R)
    int nx = 12;

    // Corner local measurements
    Scalar length = 0.166;
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> rJcJj(3,4);
    rJcJj.block(0,0,3,1) << -length/2,  length/2, 0;
    rJcJj.block(0,1,3,1) <<  length/2,  length/2, 0;
    rJcJj.block(0,2,3,1) <<  length/2, -length/2, 0;
    rJcJj.block(0,3,3,1) << -length/2, -length/2, 0;
    // std::cout << "local marker corners : " << rJcJj << std::endl;

    // TODO: upper Cholesky factor of measurement covariance (pixel)
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> SR(2,2);
    double tune = 5;
    SR = tune*Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Identity(2,2);
    // SR(0,0) = 1920;
    // SR(1,1) = 1080;

    Scalar cost = 0;
    //For each landmark seen
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> eta = x.block(6,0,6,1);
    for(int j = 0; j < param.landmarks_seen.size(); j++) {
        // For each corner of the landmark
        // *** State Predicted Landmark Location *** //
        Eigen::Matrix<Scalar,Eigen::Dynamic,1> rJNn = x.block(nx+6*param.landmarks_seen[j],0,3,1);
        // std::cout << " rJNn : " << rJNn << std::endl;
        Eigen::Matrix<Scalar,Eigen::Dynamic,1> Thetanj = x.block(nx+3+6*param.landmarks_seen[j],0,3,1);
        // std::cout << " Thetanj : " << Thetanj << std::endl;
        Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Rnj;
        rpy2rot(Thetanj,Rnj);
        // std::cout << " Rnj : " << Rnj << std::endl;
        for(int c = 0; c < 4; c++) {
            // Calculate the corner coords from our state into world space and convert to pixel to compare against the measurements
            // *** State Predicted Corner Pixel *** //
            Eigen::Matrix<Scalar,Eigen::Dynamic,1> rJcNn = Rnj*rJcJj.block(0,c,3,1)+ rJNn;
            // std::cout << "State corner of marker location : " << rJcNn << std::endl;
            // std::cout << " eta : " << eta << std::endl;
            Eigen::Matrix<Scalar,Eigen::Dynamic,1> rQOi;
            worldToPixel(rJcNn,eta,param.camera_param,rQOi);
            // std::cout << " rQOi : " << rQOi << std::endl;

            // *** Measurement Corner Pixel ***//
            Eigen::Matrix<Scalar,Eigen::Dynamic,1> rQOj = y.block(8*j+2*c, 0, 2, 1);
            // std::cout << " rQOj : " << rQOj << std::endl;

            // std::cout << "Measurement Corner Pixel : " << rQOj << std::endl;
            // std::cout << "State Predicted Corner Pixel : " << rQOi << std::endl;

            // Sum up log gausian for each measurement
            cost += logGaussian(rQOj,rQOi, SR);
        }
    }
  //std::cout << " cost : " << cost << std::endl;

    return cost;
}


double SlamLogLikelihood::operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param)
{
    // Evaluate log N(y;h(x),R)
    return slamLogLikelihood(y, x, u, param);
}


double SlamLogLikelihood::operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g)
{
    Eigen::Matrix<autodiff::dual,Eigen::Dynamic,1> xdual = x.cast<autodiff::dual>();
    autodiff::dual fdual;
    g = autodiff::gradient(slamLogLikelihood<autodiff::dual>, wrt(xdual), at(y,xdual,u,param), fdual);
    return val(fdual);
}

double SlamLogLikelihood::operator()(const Eigen::VectorXd & y, const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd &g, Eigen::MatrixXd &H)
{
    Eigen::Matrix<autodiff::dual2nd,Eigen::Dynamic,1> xdual = x.cast<autodiff::dual2nd>();
    autodiff::dual2nd fdual;
    H = autodiff::hessian(slamLogLikelihood<autodiff::dual2nd>, wrt(xdual), at(y,xdual,u,param), fdual, g);
    return val(fdual);
}







