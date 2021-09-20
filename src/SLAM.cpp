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
    SEKF.diagonal() << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;

    muEKF <<     0, // x
                 0, // y
                 0, // z
                 0, // Psi
                 0, // Theta
                 0, // Phi
                 0, // x dot
                 0, // y dot
                 0, // z dot
                 0, // Psi dot
                 0, // Theta dot
                 0; // Phi dot

    std::cout << "Initial state estimate" << std::endl;
    std::cout << "muEKF = \n" << muEKF << std::endl;
    std::cout << "SEKF = \n" << SEKF << std::endl;

    // Initialize plot states
    PlotHandles tmpHandles;
    initPlotStates(muEKF, SEKF, param, tmpHandles);


    cv::Mat view;
    std::cout << "Lets slam lads" << std::endl;
    int count = 0;
    std::vector<int> marker_ids;

    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetacm;
    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanc;
    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanm;
    Eigen::Matrix<double, Eigen::Dynamic, 1> rMNn;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnm;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnc;



    Eigen::VectorXd u;
    double timestep = 0.01;
    while(cap.isOpened()){
        cap >> view;
        if(view.empty()){
            break;
        }
        cv::Mat imgout;
        Eigen::VectorXd xk, yk, muf, mup;
        Eigen::MatrixXd Sf, Sp;

        // ****** 1. Perform time update to current frame time ******/////

        // Calculate prediction density
        timeUpdateContinuous(muEKF, SEKF, u, pm, slamparam, timestep, mup, Sp);

        // ****** 2. Identify landmarks with matching features ******/////
        slamparam.landmarks_seen.clear();
        std::vector<Marker> detected_markers;
        int n_measurements;
        detectAndDrawArUco(view, imgout, detected_markers, param);
        // Check all detected markers, if there is a new marker update the state else if max ID not met add ID to list and initialize a new landmark
        for(int i = 0; i < detected_markers.size(); i++){
            std::cout << "detected markers size" << detected_markers.size() << std::endl;
            std::cout << "detected marker we are checking: " << detected_markers[i].id << std::endl;
            //Search list of current markers
            std::vector<int>::iterator it = std::find(marker_ids.begin(), marker_ids.end(), detected_markers[i].id);
            std::cout << it - marker_ids.begin() <<std::endl;
            //If marker was found in our list, update the state
            if(it != marker_ids.end()) {
                int j = it - marker_ids.begin();
                std::cout << "Marker ID: " << detected_markers[i].id << " found, update state, landmark (j) :" << j << std::endl;

                // Add pixel location of corners to measurement yk
                n_measurements = yk.rows();
                yk.conservativeResizeLike(Eigen::MatrixXd::Zero(n_measurements+8,1));
                for(int c = 0; c < 4; c++) {
                    Eigen::Vector2d rJcNn;
                    rJcNn << detected_markers[i].corners[c].x, detected_markers[i].corners[c].y;
                    yk.block(n_measurements+c*2,0,2,1) = rJcNn;
                }
                std::cout << "yk: " << yk << std::endl;
                // Add index j to landmark seen vector
                slamparam.landmarks_seen.push_back(j);
            } else {
                std::cout << "Marker ID: " << detected_markers[i].id << " not found in list, add new landmark" << std::endl;

                // Add new landmark to now seen list
                marker_ids.push_back(detected_markers[i].id);

                //Reize the state and predicted state matrix
                int n_states = muEKF.rows();
                SEKF.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,n_states+6));
                Sp.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,n_states+6));
                double kappa = 1e6;
                for(int k = 0; k < 6; k++){
                    SEKF(SEKF.rows()-6+k,SEKF.rows()-6+k) = kappa;
                    Sp(Sp.rows()-6+k,Sp.rows()-6+k) = kappa;
                }
                muEKF.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,1));
                mup.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,1));


                // Add initial good guess
                // rot2rpy(detected_markers[i].Rcm,Thetacm);
                // Thetanc = mup.block(9,0,3,1);
                // rpy2rot(Thetanc, Rnc);
                // Rnm = Rnc*detected_markers[i].Rcm;
                // rot2rpy(Rnm,Thetanm);
                // rMNn = mup.block(6,0,3,1) + Rnc*detected_markers[i].rMCc;
                // mup.block(mup.rows()-6,0,6,1) << rMNn, Thetanm;

                // Add to measurement to vector yk
                n_measurements = yk.rows();
                yk.conservativeResizeLike(Eigen::MatrixXd::Zero(n_measurements+8,1));
                for(int c = 0; c < 4; c++) {
                    Eigen::Vector2d rJcNn;
                    rJcNn << detected_markers[i].corners[c].x, detected_markers[i].corners[c].y;
                    yk.block(n_measurements+c*2,0,2,1) = rJcNn;
                }
                // Add index j to landmark seen vector
                slamparam.landmarks_seen.push_back((n_states-nx)/6);
                std::cout << "yk: " << yk << std::endl;

                initPlotStates(muEKF, SEKF, param, tmpHandles);
            }
        }

        std::cout << "landmarks seen size : " << slamparam.landmarks_seen.size() << std::endl;
        updatePlotStates(imgout, muEKF, SEKF, param, tmpHandles);

        // -------------------------
        // Attach interactor for playing with the 3d interface
        // -------------------------
        if(interactive == 2) {

            vtkNew<vtkInteractorStyleTrackballCamera> threeDimInteractorStyle;
            vtkNew<vtkRenderWindowInteractor> threeDimInteractor;


            threeDimInteractor->SetInteractorStyle(threeDimInteractorStyle);
            threeDimInteractor->SetRenderWindow(tmpHandles.renderWindow);

            threeDimInteractor->Initialize();
            threeDimInteractor->Start();
            int wait = cv::waitKey(1);
        }
        //*********** 3. Remove failed landmarks from map (consecutive failures to match) **************//

        //*********** 4. Identify surplus features that do not correspond to landmarks in the map **************//

        //*********** 5. Initialise up to Nmax – N new landmarks from best surplus features **************//

        //*********** 6. Perform measurement update
        // Calculate filtered density

        std::cout << "mup : " << mup << std::endl;
        std::cout << "Sp : " << Sp << std::endl;
        std::cout << "Sp.rows() : " << Sp.rows() << std::endl;
        // assert(0);
        measurementUpdateIEKF(mup, Sp, u, yk, ll, slamparam, muf, Sf);
        muEKF               = muf;
        SEKF                = Sf;

        if (muEKF.hasNaN()){
            std::cout << "NaNs encountered in muEKF. muEKF = \n" << muEKF << std::endl;
            return;
        }
        if (SEKF.hasNaN()){
            std::cout << "NaNs encountered in SEKF. S = \n" << SEKF << std::endl;
            return;
        }

        std::cout << "muEKF: " << muEKF << std::endl;
        std::cout << "SEKF: " << SEKF << std::endl;
        std::cout << "muEKF.rows(): " << muEKF.rows() << std::endl;
        std::cout << "SEKF.rows(): " << SEKF.rows() << std::endl;
        // assert(0);
    }
}