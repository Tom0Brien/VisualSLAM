#include <filesystem>
#include <string>
#include "SLAM.h"
#include "model.hpp"

#include <iostream>

#include <Eigen/Core>


#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/aruco.hpp>

#include "imagefeatures.h"
#include "plot.h"
#include "cameraModel.hpp"



void runSLAMFromVideo(const std::filesystem::path &videoPath, const std::filesystem::path &cameraDataPath, const CameraParameters & param, Settings& s, int scenario, int interactive, const std::filesystem::path &outputDirectory)
{
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();

    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_out"
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // TODO
    // - Open input video at videoPath
    cv::VideoCapture cap(videoPath.string());

    // Check if camera opened successfully
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return;
    }

    //Define process model and measurement model
    SlamProcessModel     pm;
    SlamLogLikelihood    ll;
    SlamParameters slamparam;
    CameraParameters camera_param;
    importCalibrationData(cameraDataPath.string(), camera_param);
    slamparam.camera_param = camera_param;
    //Initialise the states
    int nx, ny;
    nx              = 12; // Camera states
    ny              = 0;
    Eigen::VectorXd x0(nx);
    Eigen::VectorXd muEKF(nx);
    Eigen::MatrixXd SEKF(nx, nx);

    // Initialise filter
    SEKF.fill(0);
    SEKF.diagonal() << 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,0.25,0.25,0.25;

    muEKF <<     0, // x dot
                 0, // y dot
                 0, // z dot
                 0, // Psi dot
                 0, // Theta dot
                 0, // Phi dot
                 0, // x
                 0, // y
                 -1.8, // z
                 -3.14159265359/2, // Psi
                 3.14159265359, // Theta
                 0; // Phi

    //Initialize the plot states
    int max_landmarks = 1000;
    Eigen::VectorXd muPlot(nx+max_landmarks*6);
    muPlot.setZero();
    muPlot.segment(0,12) = muEKF;
    Eigen::MatrixXd SPlot(nx+max_landmarks*6, nx+max_landmarks*6);
    SPlot.setZero();
    SPlot.block(0,0,12,12) = SEKF;
    // Initialize plot states
    PlotHandles handles;
    initPlotStates(muPlot, SPlot, param, handles);



    std::cout << "Initial state estimate" << std::endl;
    std::cout << "muEKF = \n" << muEKF << std::endl;
    std::cout << "SEKF = \n" << SEKF << std::endl;


    cv::Mat view;
    std::cout << "Lets slam lads" << std::endl;
    std::vector<int> marker_ids;

    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetacm;
    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanc;
    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanm;
    Eigen::Matrix<double, Eigen::Dynamic, 1> rMNn;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnm;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnc;



    Eigen::VectorXd u;
    double fps = cap.get(cv::CAP_PROP_FPS);
    double timestep = 1/fps;
    std::cout << "fps : " << fps << std::endl;
    std::cout << "timestep : " << timestep << std::endl;
    int count = 0;

    Eigen::VectorXd muf, mup;

    while(cap.isOpened()){
        cap >> view;
        if(view.empty()){
            break;
        }
        count++;
        cv::Mat imgout;
        Eigen::VectorXd xk, yk;
        Eigen::MatrixXd Sf, Sp;

        // ****** 1. Perform time update to current frame time ******/////
        // Calculate prediction density
        // std::cout << " Time update " << std::endl;
        timeUpdateContinuous(muEKF, SEKF, u, pm, slamparam, timestep, mup, Sp);

        // ****** 2. Identify landmarks with matching features ******/////
        slamparam.landmarks_seen.clear();
        int n_measurements;
        std::vector<Marker> detected_markers;
        // std::cout << " Detect markers " << std::endl;
        detectAndDrawArUco(view, imgout, detected_markers, param);
        // Check all detected markers, if there is a new marker update the state else if max ID not met add ID to list and initialize a new landmark
        for(int i = 0; i < detected_markers.size(); i++){

            //Search list of current markers
            std::vector<int>::iterator it = std::find(marker_ids.begin(), marker_ids.end(), detected_markers[i].id);
            //If marker was found in our list, update the state
            if(it != marker_ids.end()) {
                int j = it - marker_ids.begin();
                // Add pixel location of corners to measurement yk
                n_measurements = yk.rows();
                yk.conservativeResizeLike(Eigen::MatrixXd::Zero(n_measurements+8,1));
                for(int c = 0; c < 4; c++) {
                    Eigen::Vector2d rJcNn;
                    rJcNn << detected_markers[i].corners[c].x, detected_markers[i].corners[c].y;
                    yk.block(n_measurements+c*2,0,2,1) = rJcNn;
                }
                // Add index j to landmark seen vector
                slamparam.landmarks_seen.push_back(j);

            } else {
                // Add new landmark to now seen list
                marker_ids.push_back(detected_markers[i].id);

                //Reize the state and predicted state matrix
                int n_states = muEKF.rows();
                SEKF.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,n_states+6));
                Sp.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,n_states+6));
                muEKF.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,1));
                mup.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,1));
                double kappa = 0.01;
                for(int k = 0; k < 6; k++){
                    SEKF(SEKF.rows()-6+k,SEKF.rows()-6+k) = kappa;
                    Sp(Sp.rows()-6+k,Sp.rows()-6+k) = kappa;
                }

                // Add initial good guess
                rot2rpy(detected_markers[i].Rcm,Thetacm);
                Thetanc = mup.block(9,0,3,1);
                rpy2rot(Thetanc, Rnc);
                Rnm = Rnc*detected_markers[i].Rcm;
                rot2rpy(Rnm,Thetanm);
                rMNn = mup.block(6,0,3,1) + Rnc*detected_markers[i].rMCc;
                mup.block(mup.rows()-6,0,6,1) << rMNn, Thetanm;

                // Add marker corner measurements to vector yk
                n_measurements = yk.rows();
                yk.conservativeResizeLike(Eigen::MatrixXd::Zero(n_measurements+8,1));
                for(int c = 0; c < 4; c++) {
                    Eigen::Vector2d rJcNn;
                    rJcNn << detected_markers[i].corners[c].x, detected_markers[i].corners[c].y;
                    yk.block(n_measurements+c*2,0,2,1) = rJcNn;
                }
                // Add index j to landmark seen vector
                slamparam.landmarks_seen.push_back((n_states-nx)/6);
                std::cout << " NEW LANDMARK : " << (n_states-nx)/6 << std::endl;
                // std::cout << "yk: " << yk << std::endl;
                // std::cout << "mup: " << mup << std::endl;

            }
        }

        assert(yk.size() % 8 == 0);
        // for(int k = 0; k < slamparam.landmarks_seen.size(); k++) {
        //     std::cout << "landmarks seen : " <<  slamparam.landmarks_seen[k] << std::endl;
        // }
        // std::cout << "yk: " << yk << std::endl;
        // std::cout << "yk.rows(): " << yk.rows() << std::endl;


        //*********** 6. Perform measurement update
        // Calculate filtered density
        // std::cout << " Measurement update " << std::endl;
        // if(count > 26) {
        //     slamparam.debug = 1;
        //     std::cout << "mup " << mup << std::endl;
        //     std::cout << "mup.rows() " << mup.rows() << std::endl;
        //     vtkNew<vtkInteractorStyleTrackballCamera> threeDimInteractorStyle;
        //     vtkNew<vtkRenderWindowInteractor> threeDimInteractor;
        //     threeDimInteractor->SetInteractorStyle(threeDimInteractorStyle);
        //     threeDimInteractor->SetRenderWindow(handles.renderWindow);
        //     threeDimInteractor->Initialize();
        //     threeDimInteractor->Start();
        //     initPlotStates(muPlot, SPlot, param, handles);
        // }

        measurementUpdateIEKF(mup, Sp, u, yk, ll, slamparam, muf, Sf);
        muEKF               = muf;
        SEKF                = Sf;


        //**********  Plotting **********//

        muPlot.segment(0,muEKF.rows()) = muEKF;
        SPlot.block(0,0,SEKF.rows(),SEKF.rows()) = SEKF;
        if (interactive != 2 && count % 75 != 0){
            // updatePlotStates(view, muPlot, SPlot, param, handles);

        }
        else
        {
            std::cout << "frame no. : " << count << std::endl;
            updatePlotStates(view, muEKF, SEKF, param, handles);
            std::cout << "world postion: " << muEKF.segment(6,3) << std::endl;
            // -------------------------
            // Attach interactor for playing with the 3d interface
            // -------------------------
            vtkNew<vtkInteractorStyleTrackballCamera> threeDimInteractorStyle;
            vtkNew<vtkRenderWindowInteractor> threeDimInteractor;
            threeDimInteractor->SetInteractorStyle(threeDimInteractorStyle);
            threeDimInteractor->SetRenderWindow(handles.renderWindow);
            threeDimInteractor->Initialize();
            threeDimInteractor->Start();
            initPlotStates(muPlot, SPlot, param, handles);
        }



        if (muEKF.hasNaN()){
            std::cout << "NaNs encountered in muEKF. muEKF = \n" << muEKF << std::endl;
            return;
        }
        if (SEKF.hasNaN()){
            std::cout << "NaNs encountered in SEKF. S = \n" << SEKF << std::endl;
            return;
        }
    }
}