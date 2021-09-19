#define _USE_MATH_DEFINES

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>

#include <Eigen/Core>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "cameraModel.hpp"
#include "gaussian.hpp"
#include "plot.h"
#include "rotation.hpp"
// #include "settings.h"
#include "utility.h"

#define __DEBUG__(X) {std::cout << "In " << __FUNCTION__ << " at Line " << __LINE__ << ": " <<X << std::endl;};
#define DEBUG(X) __DEBUG__(X) 

static void write(cv::FileStorage& fs, const std::string&, const CameraParameters& x)
{
    x.write(fs);
}

void read(const cv::FileNode& node, CameraParameters& x, const CameraParameters& default_value){

    if(node.empty()){
        x = default_value;
        std::cout << "Expected to be able to read a param structure" << std::endl;
        assert(0);
    }
    else
        x.read(node);
}

void CameraParameters::print() const
{

    std::bitset<8*sizeof(flag)> bitflag(flag);
    std::cout   << "Calibration data:" << std::endl;
    std::cout   << std::setw(30) << "cameraMatrix : " << std::endl << Kc         << std::endl;
    std::cout   << std::setw(30) << "distCoeffs : "   << std::endl << distCoeffs << std::endl;
    std::cout   << std::setw(30) << "flag : "                      << bitflag    << std::endl;
    std::cout   << std::setw(30) << "imageSize : "    << std::endl << imageSize  << std::endl;
    std::cout   << std::endl;
}

// Write serialization for this struct
void CameraParameters::write(cv::FileStorage& fs) const
{
    fs  << "{"
        << "camera_matrix"           << Kc
        << "distortion_coefficients" << distCoeffs
        << "flag"                    << flag
        << "imageSize"               << imageSize
        << "}";
}

// Read serialization for this struct
void CameraParameters::read(const cv::FileNode& node)
{
    node["camera_matrix"]           >> Kc;
    node["distortion_coefficients"] >> distCoeffs;
    node["flag"]                    >> flag;
    node["imageSize"]               >> imageSize;
}

// ------------------------------------------------------------
// Export calibration data from a file
// ------------------------------------------------------------
void exportCalibrationData(const std::filesystem::path & calibrationFilePath, const CameraParameters & param){

    // TODO
    cv::FileStorage fs(calibrationFilePath.string(), cv::FileStorage::WRITE);
    fs << "CalibrationData" << param;
    fs.release();
}

// ------------------------------------------------------------
// Import calibration data from a file
// ------------------------------------------------------------
void importCalibrationData(const std::filesystem::path & calibrationFilePath, CameraParameters & param){

    cv::FileStorage fs;

    fs.open(calibrationFilePath.string(), cv::FileStorage::READ);
    assert(fs.isOpened());
    fs["CalibrationData"] >> param;
}

void calibrateCameraFromVideo(const std::filesystem::path &videoPath, const std::filesystem::path &dataPath, Settings& s)
{
    // Calibration chessboard parameters
    const int gridWidth = 10;
    const int gridHeight = 7;
    const double squareSize = 0.022;

    // TODO
    // - Open input video at videoPath
    cv::VideoCapture cap(videoPath.string());

    // Check if camera opened successfully
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return;
    }

    // - Extract relevant frames containing the chessboard
    cv::Mat view;
    std::vector<std::vector<cv::Point2f> > rQOi_set;
    cv::Size imageSize;
    int num_frames = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int count = 0;
    while(cap.isOpened()){
        cap >> view;
        if(view.empty() || count == 120)          // If there are no more images stop the loop
        {
            break;
        }
        imageSize = view.size();  // Format input image.
        std::vector<cv::Point2f> rQOi;

        if (detectChessBoard(view, rQOi)){
            rQOi_set.push_back(rQOi);
            std::cout << "Found chess board corners in frame: " << count <<std::endl;
            count += 1;
        }
        else{
            std::cout << "Found no chess board corners in frame: " << count <<std::endl;
        }

        std::cout   << std::setw(30) << "No. of frames used : "            << rQOi_set.size() << " out of " << num_frames << " frames." <<std::endl;
        std::cout   << std::setw(30) << "inputImageSize : "             << imageSize << std::endl;

        //skip 10 frames
        for(int j = 0; j < 10; j++) {
            cap >> view;
        }
    }
    // - Perform camera calibration
    CameraParameters param;
    runCalibration(s,rQOi_set,imageSize,param);
    // - Write the camera matrix and lens distortion parameters to XML file at dataPath
    exportCalibrationData(dataPath,param);
    // - Visualise the camera calibration results

}

// ------------------------------------------------------------
// detectChessBoard in a single frame
// ------------------------------------------------------------
bool detectChessBoard(const cv::Mat & view, std::vector<cv::Point2f> & rQOi ){

    // TODO: Copy from Lab3
    cv::Size patternsize(10,7);
    cv::Mat img_gray;
    cv::cvtColor(view, img_gray, cv::COLOR_BGR2GRAY);
    bool patternfound = findChessboardCorners(img_gray, patternsize, rQOi);
    return patternfound;
}

// ------------------------------------------------------------
// run calibration on a set of detected points
// ------------------------------------------------------------
void runCalibration(const Settings & s, const std::vector<std::vector<cv::Point2f>> & rQOi_set,  const cv::Size & imageSize, CameraParameters & param){

    std::vector<cv::Point3f> rPNn_base;
    std::vector<std::vector<cv::Point3f>> rPNn;
    generateCalibrationGrid(s, rPNn_base);

    for(int i = 0; i < rQOi_set.size(); i++) {
        rPNn.push_back(rPNn_base);
    }

    
    double rms=-1;
    if( !rQOi_set.empty() ){
        std::cout << "here" << std::endl;
        // ------------------------------------------------------------
        // Run camera calibration
        // ------------------------------------------------------------
        // TODO: Copy from Lab3
        cv::Mat cameraMatrix,distCoeffs;
        distCoeffs = cv::Mat::zeros(5,1,CV_64F);
        std::vector<cv::Mat> R, rNCc;
        int flag = cv::CALIB_THIN_PRISM_MODEL | cv::CALIB_RATIONAL_MODEL;
        double rms = cv::calibrateCamera(rPNn, rQOi_set, imageSize, cameraMatrix, distCoeffs, R, rNCc, flag);
        std::cout << "Calibration rms: " << rms << std::endl;
        // ------------------------------------------------------------
        // Write parameters to the data struct 
        // ------------------------------------------------------------
        param.Kc = cameraMatrix;
        param.distCoeffs = distCoeffs;
        param.flag = flag;
        param.imageSize = imageSize;


    } else{
        std::cerr << "No imagePoints found" << std::endl;
        assert(0);
    }
}

// ------------------------------------------------------------
// Define camera calibration grid
// ------------------------------------------------------------
void generateCalibrationGrid(const Settings & s, std::vector<cv::Point3f> & rPNn_grid){
    // Defining the world coordinates for 3D points
    for(int i=0; i < s.boardSize.height; i++) {
        for(int j=0; j < s.boardSize.width; j++){
            rPNn_grid.push_back(cv::Point3f(j*s.squareSize,i*s.squareSize,0));
        }
    }
}


// ---------------------------------------------------------------------
// 
// WorldToPixelAdaptor
// 
// ---------------------------------------------------------------------
int WorldToPixelAdaptor::operator()(const Eigen::VectorXd & rPNn, const Eigen::VectorXd & eta, const CameraParameters & param, Eigen::VectorXd & rQOi){

    assert(rPNn.rows() == 3);
    assert(rPNn.cols() == 1);

    assert(eta.rows() == 6);
    assert(eta.cols() == 1);

    int err = worldToPixel(rPNn, eta, param, rQOi);
    if (err){
        return err;
    }

    assert(rQOi.rows() == 2);
    assert(rQOi.cols() == 1);

    return 0;
}

int WorldToPixelAdaptor::operator()(const Eigen::VectorXd & rPNn, const Eigen::VectorXd & eta, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR){

    int res = operator()(rPNn, eta, param, rQOi);
    SR      = Eigen::MatrixXd::Zero(rQOi.rows(), rQOi.rows());
    return res;
}

int WorldToPixelAdaptor::operator()(const Eigen::VectorXd & rPNn, const Eigen::VectorXd & eta, const CameraParameters & param, Eigen::VectorXd & rQOi, Eigen::MatrixXd & SR, Eigen::MatrixXd & J){

    assert(rPNn.rows() == 3);
    assert(rPNn.cols() == 1);

    assert(eta.rows() == 6);
    assert(eta.cols() == 1);

    // TODO
    int res = operator()(rPNn, eta, param, rQOi,SR);
    // Use either the analytical expression or autodiff
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
            x,
            y,
            z;

    x       = rPCc(0);
    y       = rPCc(1);
    z       = rPCc(2);

    // Check that z is positive
    assert(z>0);

    u       = x/z;
    v       = y/z;

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
    dudr << 1/z, 0, -x/z*z;

    Eigen::MatrixXd dvdr;
    dvdr.resize(1,3);
    dvdr << 0, 1/z, -y/z*z;

    double drdu = pow(u2 + v2,-0.5)*u; 
    double drdv = pow(u2 + v2,-0.5)*v; 

    double dalphadr = 2*k1*r + 4*k2*r3 + 6*k3*r5; 
    double dbetadr = 2*k4*r + 4*k5*r3 + 6*k6*r5;

    double dcdr = (dalphadr*(1+beta)-(1+alpha)*dbetadr)/((1+beta)*(1+beta)); 

    double duddu = dcdr*drdu*u + c + 2*p1*v + p2*(2*r*drdu + 4*u) + 2*s1*r*drdu + 4*s2*r3*drdu; 

    double duddv = dcdr*drdv*u + 2*p1*u + p2*(2*r*drdv) + 2*s1*r*drdv + 4*s2*r3*drdv;  

    double dvddu = dcdr*drdu*v + 2*p2*v + p1*(2*r*drdu) + 2*s3*r*drdu + 4*s4*r3*drdu; 

    double dvddv = dcdr*drdv*v + c + 2*p2*u + p1*(2*r*drdv + 4*v) + 2*s3*r*drdv + 4*s4*r3*drdv; 



    Eigen::MatrixXd A;
    A.resize(1,3);
    A = fx*(duddu*dudr + duddv*dvdr);

    Eigen::MatrixXd B;
    B.resize(1,3);
    B = fy*(dvddu*dudr + dvddv*dvdr);

    J.resize(2,3);
    J << A, B;
    J = J*Rnc.transpose();


    return res;
}
