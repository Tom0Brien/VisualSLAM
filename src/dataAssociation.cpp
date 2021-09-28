#include <algorithm>
#include <functional>
#include <limits>
#include <vector>
#include <iostream>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>

#include "dataAssociation.h"
#include "gaussian.hpp"
#include "measurementPointLandmark.hpp"
#include "utility.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif



double snn(const Eigen::VectorXd & mux, const Eigen::MatrixXd & Sxx, const Eigen::MatrixXd & y, const CameraParameters & param, std::vector<int>& idx, bool enforceJointCompatibility){

    double nstd     = 3;
    // Probability mass enclosed by nstd standard deviations (same as normcdf(nstd) - normcdf(-nstd))
    double c        = 2*normcdf(nstd) - 1;


    int nx_all  = mux.rows();
    assert(nx_all>0);

    // Check the length of the state vector is of dimensions such that
    // nx_all - nx is a multiple of three
    int nx      = 6;
    assert(((nx_all - nx)%3)==0);

    // Check that there is a positive number of landmarks
    int n      = (nx_all - nx)/3;
    assert(n>0);

    assert(y.rows() == 2);
    assert(y.cols() > 0);
    int ny  = y.rows();
    int m   = y.cols();

    // Pre-compute terms
    Eigen::VectorXd muY;
    Eigen::MatrixXd SYY;
    MeasurementPointLandmarkBundle landmarkBundle;
    auto h  = std::bind(
                landmarkBundle,
                std::placeholders::_1,      // x
                param,
                std::placeholders::_2,      // h
                std::placeholders::_3,      // SR
                std::placeholders::_4);     // C

    affineTransform(mux, Sxx, h, muY, SYY);

    // chi2inv LUT
    std::vector<double> chi2LUT;
    chi2LUT.resize(n);
    for (int jj = 0; jj < n; ++jj)
    {
        chi2LUT[jj] = chi2inv(c, ny*(jj+1));
    }

    // Index
    idx.clear();
    idx.resize(n, -1);

    std::vector<int> midx;
    midx.resize(m);
    for (int i = 0; i < m; ++i)
    {
        midx[i]     = i;
    }


    // Surprisal per unassociated landmark
    double sU   = std::log(param.imageSize.width) +  std::log(param.imageSize.height);

    double s    = n*sU;

    double smin = std::numeric_limits<double>::infinity();
    std::vector<int> diff;
    std::vector<int>::iterator it, ls, space;
    diff.resize(m);
    for (int j = 0; j < n; ++j)
    {


        double dsmin    = std::numeric_limits<double>::infinity();
        double scur     = s;

        double snext    = 0;
        bool jcnext;

        std::vector<int> idxcur;
        idxcur          = idx;
        space           = idxcur.begin();
        std::advance(space, j);
        ls              = std::set_difference(midx.begin(), midx.end(), idxcur.begin(), space, diff.begin());

        // associate landmark j with each unassociated feature
        for (it = diff.begin(); it < ls; ++it){
            int i   = *it;


            if (!individualCompatibility(i, j, ny, y, muY, SYY, chi2LUT)){
                continue;
            }

            std::vector<int> idxnext    = idxcur;
            idxnext[j]                  = i;

            jcnext = jointCompatibility(idxnext, sU, ny, y, muY, SYY, chi2LUT, snext);
            if (enforceJointCompatibility && !jcnext){
                continue;
            }

            double ds = snext - scur;
            if (ds < dsmin){
                idx     = idxnext;
                dsmin   = ds;
                s       = snext;
            }
        }

        // landmark j unassociated
        std::vector<int> idxnext    = idxcur;
        jointCompatibility(idxnext, sU, ny, y, muY, SYY, chi2LUT, snext);

        // Change in surprisal
        double ds = snext - scur;
        if (ds < dsmin){
            idx     = idxnext;
            s       = snext;
        }
    }

    if (smin < std::numeric_limits<double>::infinity()){
        s   = smin;
    }

    return s;
}



bool individualCompatibility(const int  & i, const int &  j, const int  & ny, const Eigen::MatrixXd & y, const Eigen::VectorXd & muY, const Eigen::MatrixXd & SYY, const std::vector<double> & chi2LUT){
    assert(y.rows() == ny);
    assert(i >= 0);
    assert(j >= 0);
    assert(SYY.rows() == muY.rows());
    assert(SYY.cols() == muY.rows());

    Eigen::MatrixXd yi, muYj, SYYj, R, QR, zij, z(1,1), Syy;

    muYj = muY.segment(j*ny,2);
    double Cj = chi2LUT[j];
    Syy = SYY.block(0,j*ny,SYY.rows(),ny);
    yi = y.block(0,i,ny, 1);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Syy);
    qr.compute(Syy);
    QR = qr.matrixQR();
    R = QR.triangularView<Eigen::Upper>();
    SYYj = R.topRows(ny);
    zij = SYYj.transpose().triangularView<Eigen::Lower>().solve(yi - muYj);
    z = zij.transpose()*zij;

    bool cij = z(0,0) < Cj;
    return cij;
}

bool jointCompatibility(const std::vector<int> & idx, const double & sU, const int  & ny, const Eigen::MatrixXd & y, const Eigen::VectorXd & muY, const Eigen::MatrixXd & SYY, const std::vector<double> & chi2LUT, double & surprisal){
    int n           = idx.size();

    assert(y.rows() == ny);
    assert(n*ny == muY.rows());
    assert(SYY.rows() == muY.rows());
    assert(SYY.cols() == muY.rows());
    assert(chi2LUT.size() == n);

    int nA          = 0;
    std::vector<bool>   flag(n);
    std::vector<int>    idxi;
    std::vector<int>    idxj;
    std::vector<int>    idxyj;
    for (int j = 0; j < n; ++j)
    {
        flag[j]   = idx[j] != -1;
        if(flag[j]){
            nA++;
            idxi.push_back(idx[j]);
            idxj.push_back(j);
            idxyj.push_back(ny * j + 0);
            idxyj.push_back(ny * j + 1);
        }
    }


    // If Eigen 3.4.0
    // Eigen::VectorXd muA     = muY(idxyj);
    // Eigen::MatrixXd SS      = SYY(Eigen::all, idxyj).householderQr().matrixQR().triangularView<Eigen::Upper>();
    Eigen::VectorXd muA(idxyj.size());
    Eigen::MatrixXd SS(SYY.rows(), idxyj.size());
    for (int i = 0; i < idxyj.size(); ++i)
    {
        muA(i)      =  muY(idxyj[i]);
        SS.col(i)   =  SYY.col(idxyj[i]);
    }

    Eigen::MatrixXd SSQR    = SS.householderQr().matrixQR().triangularView<Eigen::Upper>();
    Eigen::MatrixXd SA      = SSQR.topRows(ny*nA);

    // If using Eigen 3.4.0
    // Eigen::MatrixXd yABlock = y(Eigen::all, idxi);
    // Eigen::Map<Eigen::VectorXd> yA(yABlock.data(), yABlock.size());
    Eigen::VectorXd yA(ny*nA);
    for (int i = 0; i < idxi.size(); ++i)
    {
        yA.segment(ny*i, ny) = y.col(idxi[i]);
    }


    Eigen::VectorXd zA      = SA.triangularView<Eigen::Upper>().transpose().solve(yA - muA);
    double dM2A             = zA.squaredNorm();

    const double halflog2pi = std::log(M_PI*2)/2;
    double s1               = ny*nA*halflog2pi;
    double s2               = SA.diagonal().array().abs().log().sum();
    double s3               = 0.5*dM2A;
    double sA               = (s1 + s2 + s3);

    // Number of unassociated landmarks
    int nU                  = n - nA;

    //
    surprisal               = sA + nU * sU;

    bool debug = false;
    if (debug){
        std::cout   << "surprisal = "
                    << std::setw(8) << surprisal
                    << ", s1 = " << std::setw(8) <<  s1
                    << ", s2 = " << std::setw(8) <<  s2
                    << ", s3 = " << std::setw(8) <<  s3
                    << ", nU*sU = " << std::setw(8) <<  nU * sU;

        std::cout   << " | Landmark ID: ";
        for (int i = 0; i < idxj.size(); ++i)
        {
            std::cout   << idxj[i];
            if ((i+1) != idxj.size()) {
                std::cout   << ", ";
            }
        }

        std::cout   << " | Feature ID: ";
        for (int i = 0; i < idxi.size(); ++i)
        {
            std::cout   << idxi[i];
            if ((i+1) != idxi.size()) {
                std::cout   << ", ";
            }
        }
        std::cout   << std::endl;

    }

    // Check joint compatibility
    bool jc;
    if (nA > 0){
        jc  = dM2A <= chi2LUT[nA - 1];
    }else{
        // Vacuous truth
        jc  = true;
    }

    return jc;
}


