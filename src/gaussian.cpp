#include <Eigen/Core>
#include <cassert>
#include <iostream>

#include "gaussian.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// 
// conditionGaussianOnMarginal
// 
// -------------------------------------------------------------------------------- ------
// --------------------------------------------------------------------------------------

void conditionGaussianOnMarginal(
    const Eigen::VectorXd & muyx, 
    const Eigen::MatrixXd & Syx, 
    const Eigen::VectorXd & y,
    Eigen::VectorXd & muxGy, 
    Eigen::MatrixXd & SxGy)
{
    // TODO: Copy from Lab 4
    int ny = y.rows();
    int nx = Syx.cols() - ny;  
    Eigen::MatrixXd S1  = Syx.topLeftCorner(ny,ny);
    Eigen::MatrixXd S2 =  Syx.topRightCorner(ny,nx);
    Eigen::MatrixXd S3 =  Syx.bottomRightCorner(nx,nx); 
    Eigen::MatrixXd mux =  muyx.tail(nx);
    Eigen::MatrixXd muy = muyx.head(ny);
    muxGy = mux + S2.transpose()*(S1.triangularView<Eigen::Upper>().transpose().solve(y - muy));
    SxGy = S3;
}



// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// 
// gaussianConfidenceEllipse
// 
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------

void gaussianConfidenceEllipse3Sigma(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S, Eigen::MatrixXd & x){
    assert(mu.rows() == 2);
    assert(S.rows() == 2);
    assert(S.cols() == 2);

    int nsamples  = 100;

    // TODO: 
    // assert(0);
    x.resize(2,nsamples);
    Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(nsamples, 0, 2*M_PI);
    Eigen::MatrixXd Z;
    Z.resize(2,nsamples);
    double r = sqrt(11.8292);
    Eigen::MatrixXd zx; 
    zx.resize(nsamples, 1);
    zx = r*t.array().cos();
    Eigen::MatrixXd zy;
    zy.resize(nsamples, 1);
    zy = r*t.array().sin();
    Z << zx.transpose(), zy.transpose();

    x = (S.transpose()*Z).colwise() + mu;


    assert(x.cols() == nsamples);
    assert(x.rows() == 2);
}


void gaussianConfidenceQuadric3Sigma(const Eigen::VectorXd &mu, const Eigen::MatrixXd & S, Eigen::MatrixXd & Q){
    const int nx  = 3;
    assert(mu.rows() == nx);
    assert(S.rows() == nx);
    assert(S.cols() == nx);

    std::cout << "mu " << mu << std::endl;
    std::cout << "S " << S << std::endl;
    std::cout << "Q " << Q << std::endl;

    Eigen::MatrixXd z =   S.triangularView<Eigen::Upper>().transpose().solve(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Identity(S.rows(),S.rows()))*mu;
    Eigen::MatrixXd topLeft = S.triangularView<Eigen::Upper>().solve( S.triangularView<Eigen::Upper>().transpose().solve(
                Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Identity(S.rows(),S.rows()))
        );
    Eigen::MatrixXd topRight =  -S.triangularView<Eigen::Upper>().solve(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Identity(S.rows(),S.rows()))*z;
    Eigen::MatrixXd bottomLeft = -z.transpose()*S.triangularView<Eigen::Upper>().transpose().solve(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Identity(S.rows(),S.rows()));
    Eigen::MatrixXd zTz = z.transpose()*z;
    double bottomRight = zTz(0) - 14.1564;
    Q.resize(4,4);
    Q << topLeft,topRight,bottomLeft,bottomRight;

}
