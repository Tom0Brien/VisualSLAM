#ifndef CALIBRATION_H
#define CALIBRATION_H


#include <Eigen/Core>
#include <filesystem>
#include <opencv2/calib3d.hpp>

#include "rotation.hpp"
#include "settings.h"

struct CameraParameters{
    cv::Mat Kc;                 // Camera Matrix
    cv::Mat distCoeffs;         // Distortion coefficients
    int flag            = 0;    // Calibration flag
    double fieldOfView  = 150;  // Describe the arc of the view cone
    cv::Size imageSize;         // Image size
    // Read and write methods required for class serialisation
    void read(const cv::FileNode & node);
    void write(cv::FileStorage& fs) const;
    // Convenience function
    void print() const;
};

// ---------------------------------------------------------------------
// 
// Camera calibration
// 
// ---------------------------------------------------------------------

// bool detectChessBoard(const Settings & s, const cv::Mat & view, std::vector<cv::Point2f> & rQOi);
// void calibrateCameraFromImageSet(const Settings & s, CameraParameters & param);
// void exportCalibrationData(const std::filesystem::path & calibrationFilePath, const CameraParameters & param);
// void importCalibrationData(const std::filesystem::path & calibrationFilePath, CameraParameters & param);
// bool getPoseFromCheckerBoardImage(const cv::Mat & view const CameraParameters & param, Eigen::VectorXd & eta);
// void runCalibration(const Settings & s, const std::vector<std::vector<cv::Point2f>> & rQOi_set,  const cv::Size & imageSize, CameraParameters & param);
// void showCalibrationDataVTK(const Settings & s, const CameraParameters & param);

void calibrateCameraFromVideo(const std::filesystem::path &videoPath, const std::filesystem::path &dataPath, Settings& s);
bool detectChessBoard(const cv::Mat & view, std::vector<cv::Point2f> & rQOi );
void runCalibration(const Settings & s, const std::vector<std::vector<cv::Point2f>> & rQOi_set,  const cv::Size & imageSize, CameraParameters & param);
void generateCalibrationGrid(const Settings & s, std::vector<cv::Point3f> & rPNn_grid);
void importCalibrationData(const std::filesystem::path & calibrationFilePath, CameraParameters & param);

// ---------------------------------------------------------------------
// 
// worldToPixel
// 
// ---------------------------------------------------------------------

template<typename Scalar>
int worldToPixel(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> & rPNn, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> & eta, const CameraParameters & param, Eigen::Matrix<Scalar, Eigen::Dynamic, 1> & rQOi){

    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;



    assert(rPNn.rows() == 3);
    assert(rPNn.cols() == 1);
    assert(param.Kc.rows == 3);
    assert(param.Kc.cols == 3);
    assert(param.distCoeffs.cols == 1);
    assert(eta.rows() == 6);
    assert(eta.cols() == 1);

    Matrix Rcn, Rnc;
    Vector rCNn, Thetanc, rPCc, rQCc, uQCc;

    rCNn        = eta.head(3);
    Thetanc     = eta.tail(3);

    rpy2rot<Scalar>(Thetanc, Rnc);

    rPCc        = Rnc.transpose() * (rPNn - rCNn);

    uQCc        = rPCc / rPCc.norm();

    double maxAngle = param.fieldOfView*((CV_PI/180.0)/2.);
    double cAngle   = std::cos(maxAngle);

    if (uQCc(2)<cAngle){
        // Pixel is not within the cone of the camera
        return 1;
    }

    int supportedFlag    = 0;
    supportedFlag        |= cv::CALIB_RATIONAL_MODEL;
    // supportedFlag        |= cv::CALIB_TILTED_MODEL;
    supportedFlag        |= cv::CALIB_THIN_PRISM_MODEL;

    bool isRationalModel    = (param.flag & cv::CALIB_RATIONAL_MODEL) == cv::CALIB_RATIONAL_MODEL;
    bool isThinPrisimModel  = (param.flag & cv::CALIB_THIN_PRISM_MODEL) == cv::CALIB_THIN_PRISM_MODEL;
    bool isSupported        = (param.flag & ~supportedFlag) == 0;

    assert(isSupported);

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
    Scalar  
            alpha,
            beta,
            c,
            r,
            r2,
            r4,
            r6,
            u,
            u2,
            up,
            v,
            v2,
            vp,
            x,
            y,
            z;

    x       = rPCc(0);
    y       = rPCc(1);
    z       = rPCc(2);

    // Check that z is positive
    // assert(z>0);

    u       = x/z;
    v       = y/z;

    using std::sqrt;

    u2      = u*u;
    v2      = v*v;
    r2      = u2 + v2;
    r       = sqrt(r2);
    r4      = r2*r2;
    r6      = r4*r2;

    alpha   = k1*r2 + k2*r4 + k3*r6;
    beta    = k4*r2 + k5*r4 + k6*r6;
    c       = (1.0 + alpha)/(1.0 + beta);

    up      = c*u + p1*2*u*v + p2*(r2 + 2*u2) + s1*r2 + s2*r4;
    vp      = c*v + p2*2*u*v + p1*(r2 + 2*v2) + s3*r2 + s4*r4;

    rQOi.resize(2,1);
    rQOi    << fx*up + cx, fy*vp + cy;

    bool isInWidth  = 0 <= rQOi(0) && rQOi(0) <= param.imageSize.width-1;
    bool isInHeight = 0 <= rQOi(1) && rQOi(1) <= param.imageSize.height-1;
    if (!(isInWidth && isInHeight)){
        // Pixel is not within the image 
        return 2;
    }

    return 0;
}

// ---------------------------------------------------------------------
// 
// WorldToPixelAdaptor
// 
// ---------------------------------------------------------------------
struct WorldToPixelAdaptor{
    int operator()(const Eigen::VectorXd & rPNn, const Eigen::VectorXd & eta, const CameraParameters & param, Eigen::VectorXd & rQOi);
    int operator()(const Eigen::VectorXd & rPNn, const Eigen::VectorXd & eta, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR);
    int operator()(const Eigen::VectorXd & rPNn, const Eigen::VectorXd & eta, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR, Eigen::MatrixXd & J);
};






#endif
