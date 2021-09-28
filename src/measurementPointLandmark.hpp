#ifndef MEASUREMENT_POINT_LANDMARK_SINGLE_H
#define MEASUREMENT_POINT_LANDMARK_SINGLE_H

#include <Eigen/Core>
#include "cameraModel.hpp"
#include "rotation.hpp"


struct MeasurementPointLandmarkSingle{
    int operator()(const int & j, const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd  & rQOi){
        assert(x.cols() == 1);
        const int nCameraStates = 12;
        const int featureDim    = 3;
        int nLandmarkStates      = x.rows() - nCameraStates;
        int nLandmarks           = nLandmarkStates / featureDim;

        // Check that there are feature states
        assert(nLandmarkStates > 0);
        assert(j >= 0);
        // Check that the number of states for features is a multiple of featureDim
        assert((nLandmarkStates%featureDim) == 0);

        // TODO:
        // Some call to worldToPixel
        rQOi.resize(2,1);
        Eigen::VectorXd eta = x.segment(6,6);
        Eigen::VectorXd rPNn = x.segment(nCameraStates+j*featureDim,3);
        int res = worldToPixel(rPNn,eta,param,rQOi); // what ever the return value of worldToPixel would be

        return res;
    }

    int operator()(const int & j, const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd  & rQOi, Eigen::MatrixXd  & SR){
        int res = operator()(j, x, param, rQOi);
        SR.resize(2,2);
        // TODO
        // SR
        SR(0,0) = 0.01;
        SR(0,1) = 0.0;
        SR(1,1) = 0.01;
        SR(1,0) = 0.0;

        return res;
    }

    int operator()(const int & j, const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR, Eigen::MatrixXd & J){
        int res = operator()(j, x, param, rQOi, SR);

        assert(x.cols() == 1);

        const int nCameraStates = 12;
        const int featureDim    = 3;
        int nLandmarkStates     = x.rows() - nCameraStates;
        int nLandmarks          = nLandmarkStates / featureDim;

        J.resize(2, nCameraStates + 3*nLandmarks);
        J.setZero();

        //J
        // Use either the analytical expression or autodiff
        Eigen::VectorXd eta = x.segment(6,6);
        Eigen::VectorXd rPNn = x.segment(nCameraStates+j*featureDim,3);

        typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
        typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        Matrix Rcn, Rnc;
        Vector rCNn, Thetanc, rPCc, rQCc, uQCc;

        rCNn        = eta.head(3);
        Thetanc     = eta.tail(3);

        rpy2rot<double>(Thetanc, Rnc);

        rPCc        = Rnc.transpose() * (rPNn - rCNn);

        int supportedFlag    = 0;
        supportedFlag        |= cv::CALIB_RATIONAL_MODEL;
        // supportedFlag        |= cv::CALIB_TILTED_MODEL;
        supportedFlag        |= cv::CALIB_THIN_PRISM_MODEL;

        bool isRationalModel    = (param.flag & cv::CALIB_RATIONAL_MODEL) == cv::CALIB_RATIONAL_MODEL;
        bool isThinPrisimModel  = (param.flag & cv::CALIB_THIN_PRISM_MODEL) == cv::CALIB_THIN_PRISM_MODEL;
        bool isSupported        = (param.flag & ~supportedFlag) == 0;


        // Constants
        double
                cx,
                cy,
                fx,
                fy,
                k1,
                k2,
                k3,
                k4,
                k5,
                k6,
                p1,
                p2,
                s1,
                s2,
                s3,
                s4;

        // Camera matrix
        fx  = param.Kc.at<double>( 0,  0);
        fy  = param.Kc.at<double>( 1,  1);
        cx  = param.Kc.at<double>( 0,  2);
        cy  = param.Kc.at<double>( 1,  2);


        k1  = param.distCoeffs.at<double>( 0,  0);
        k2  = param.distCoeffs.at<double>( 1,  0);
        p1  = param.distCoeffs.at<double>( 2,  0);
        p2  = param.distCoeffs.at<double>( 3,  0);
        k3  = param.distCoeffs.at<double>( 4,  0);

        // Distortion coefficients
        if (isRationalModel){
            if (isThinPrisimModel){
                s1  = param.distCoeffs.at<double>( 8,  0);
                s2  = param.distCoeffs.at<double>( 9,  0);
                s3  = param.distCoeffs.at<double>(10,  0);
                s4  = param.distCoeffs.at<double>(11,  0);
            }else{
                s1  = param.distCoeffs.at<double>( 5,  0);
                s2  = param.distCoeffs.at<double>( 6,  0);
                s3  = param.distCoeffs.at<double>( 7,  0);
                s4  = param.distCoeffs.at<double>( 8,  0);
            }
        }else{
            s1  = 0.0;
            s2  = 0.0;
            s3  = 0.0;
            s4  = 0.0;
        }

        if (isThinPrisimModel){
            k4  = param.distCoeffs.at<double>( 5,  0);
            k5  = param.distCoeffs.at<double>( 6,  0);
            k6  = param.distCoeffs.at<double>( 7,  0);
        }else{
            k4  = 0.0;
            k5  = 0.0;
            k6  = 0.0;
        }


        // Scalar Variables
        double
                alpha,
                beta,
                c,
                r,
                r2,
                r3,
                r4,
                r5,
                r6,
                u,
                u2,
                up,
                v,
                v2,
                vp,
                x_,
                y,
                z;

        x_      = rPCc(0);
        y       = rPCc(1);
        z       = rPCc(2);

        // Check that z is positive
        if(z <= 0) {
            u = 0;
            v = 0;
        } else {
            u       = x_/z;
            v       = y/z;
        }


        using std::sqrt;
        using std::pow;

        u2      = u*u;
        v2      = v*v;
        r2      = u2 + v2;
        r       = sqrt(r2);
        r3      = r2*r;

        r4      = r2*r2;
        r5      = r4*r;
        r6      = r4*r2;

        alpha   = k1*r2 + k2*r4 + k3*r6;
        beta    = k4*r2 + k5*r4 + k6*r6;
        c       = (1.0 + alpha)/(1.0 + beta);

        up      = c*u + p1*2*u*v + p2*(r2 + 2*u2) + s1*r2 + s2*r4;
        vp      = c*v + p2*2*u*v + p1*(r2 + 2*v2) + s3*r2 + s4*r4;

        Eigen::MatrixXd dudr;
        dudr.resize(1,3);
        Eigen::MatrixXd dvdr;
        dvdr.resize(1,3);
        if(z <= 0) {
            dudr << 0,0,0;
            dvdr << 0,0,0;
        } else {
            dudr << 1/z, 0, -x_/(z*z);
            dvdr << 0, 1/z, -y/(z*z);
        }

        double drdu, drdv;
        if(u!=0){
            drdu = pow(u2 + v2,-0.5)*u;
        } else {
            drdu = 0;
        }

        if(v!=0){
            drdv = pow(u2 + v2,-0.5)*v;
        } else {
            drdv = 0;
        }

        double dalphadr = 2*k1*r + 4*k2*r3 + 6*k3*r5;
        double dbetadr = 2*k4*r + 4*k5*r3 + 6*k6*r5;

        double dcdr = (dalphadr*(1+beta)-(1+alpha)*dbetadr)/((1+beta)*(1+beta));

        double duddu = dcdr*drdu*u + c + 2*p1*v + p2*(2*r*drdu + 4*u) + 2*s1*r*drdu + 4*s2*r3*drdu;
        double duddv = dcdr*drdv*u + 2*p1*u + p2*(2*r*drdv) + 2*s1*r*drdv + 4*s2*r3*drdv;
        double dvddu = dcdr*drdu*v + 2*p2*v + p1*(2*r*drdu) + 2*s3*r*drdu + 4*s4*r3*drdu;
        double dvddv = dcdr*drdv*v + c + 2*p2*u + p1*(2*r*drdv + 4*v) + 2*s3*r*drdv + 4*s4*r3*drdv;


        Eigen::MatrixXd JrPCc;
        Eigen::MatrixXd JrCNn;
        Eigen::MatrixXd JRnc;
        Eigen::MatrixXd JrPNn;
        JrPCc.resize(2,3);
        JrCNn.resize(2,3);
        JrPNn.resize(2,3);
        Eigen::MatrixXd A;
        A.resize(1,3);
        A = fx*(duddu*dudr + duddv*dvdr);

        Eigen::MatrixXd B;
        B.resize(1,3);
        B = fy*(dvddu*dudr + dvddv*dvdr);
        JrPCc << A, B;

        JrPNn = JrPCc*Rnc.transpose();
        JrCNn = JrPNn*-1;
        J.setZero();

        //Rotation
        JRnc.resize(2,3);
        Eigen::MatrixXd S1(3,3);
        Eigen::MatrixXd S2(3,3);
        Eigen::MatrixXd S3(3,3);
        S1 << 0,0,0,0,0,-1,0,1,0;
        S2 << 0,0,1,0,0,0,-1,0,0;
        S3 << 0,-1,0,1,0,0,0,0,0;

        Eigen::MatrixXd dPhi(3,3);
        Eigen::MatrixXd dTheta(3,3);
        Eigen::MatrixXd dPsi(3,3);
        dPhi.setZero();
        dTheta.setZero();
        dPsi.setZero();

        Eigen::MatrixXd RX(3,3);
        Eigen::MatrixXd RY(3,3);
        Eigen::MatrixXd RZ(3,3);
        RX.setZero();
        RY.setZero();
        RZ.setZero();
        rotx(x(3),RX);
        roty(x(4),RY);
        rotz(x(5),RZ);

        dPsi   =   RZ*S3*RY*RX;
        dTheta =   RZ*RY*S2*RX;
        dPhi   =   RZ*RY*RX*S1;

        JRnc.block(0,0,2,1) = JrPCc*(dPhi.transpose()*(rPNn - rCNn));
        JRnc.block(0,1,2,1) = JrPCc*(dTheta.transpose()*(rPNn - rCNn));
        JRnc.block(0,2,2,1) = JrPCc*(dPsi.transpose()*(rPNn - rCNn));

        J.block(0,0,2,3) = JrCNn;
        J.block(0,3,2,3) = JRnc;
        J.block(0,6+3*j,2,3) = JrPNn;

        assert(nLandmarkStates > 0);
        assert(j >= 0);
        // Check that the number of states for features is a multiple of featureDim
        assert((nLandmarkStates%featureDim) == 0);

        return res;
    }
};

struct MeasurementPointLandmarkBundle{
    void operator()(const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd & rQOi);
    void operator()(const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR);
    void operator()(const Eigen::VectorXd & x, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR, Eigen::MatrixXd & J);
};


#endif