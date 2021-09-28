#include <filesystem>
#include <string>
#include "SLAM.h"
#include "utility.h"
#include "dataAssociation.h"
#include "measurementPointLandmark.hpp"
#include <opencv2/calib3d.hpp>
#include <memory>


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

    // - Open input video at videoPath
    cv::VideoCapture cap(videoPath.string());
    //Output file name
    std::string output_file_name;
    if(scenario == 1) {
        output_file_name = "../out/tags_out.MOV";
    } else if(scenario == 2) {
        output_file_name = "../out/points_out.MOV";
    } else {
        output_file_name = "../out/ducks_out.MOV";
    }
    //Create output directory
    std::filesystem::create_directories("../out");
    double width = cap.get(cv::CAP_PROP_FRAME_WIDTH)*2;
    double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);

    // Check if camera opened successfully
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return;
    }


    SlamParameters slamparam;
    CameraParameters camera_param;
    importCalibrationData(cameraDataPath.string(), camera_param);
    slamparam.camera_param = camera_param;

    //Define process model and measurement model
    SlamProcessModel     pm;

    //Default
    std::cout << "scenario" << scenario << std::endl;
    if(scenario == 1) {
        std::cout << "Scenario 1" << std::endl;
        slamparam.position_tune = 0.1;
        slamparam.orientation_tune = 0.1;
        slamparam.n_landmark = 6;
    } else if (scenario == 2){
        std::cout << "Scenario 2" << std::endl;
        slamparam.position_tune = 0.05;
        slamparam.orientation_tune = 0.01;
        slamparam.n_landmark = 3;
    }
    // slamLogLikelihoodAnalytical ll;

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
                 -3.14159265359/2, // Phi
                 3.14159265359, // Theta
                 0  ; // Psi

    //Initialize the plot states
    int max_landmarks = 50;
    Eigen::VectorXd muPlot(nx+max_landmarks*6);
    muPlot.setZero();
    muPlot.segment(0,12) = muEKF;
    Eigen::MatrixXd SPlot(nx+max_landmarks*6, nx+max_landmarks*6);
    SPlot.setZero();
    SPlot.block(0,0,12,12) = SEKF;
    // Initialize plot states
    PlotHandles handles;
    initPlotStates(muPlot, SPlot, param, handles,slamparam);

    int *size = handles.renderWindow->GetSize();
    int & w = size[0];
    int & h = size[1];
    cv::VideoWriter video(output_file_name,cv::VideoWriter::fourcc('m','p','4','v'),fps, cv::Size(w,h));



    cv::Mat frame;

    // Scenario 1
    std::vector<int> marker_ids;

    //Scenario 2
    cv::Mat landmark_descriptors;
    int max_features = 100;
    cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING);

    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetacm;
    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanc;
    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanm;
    Eigen::Matrix<double, Eigen::Dynamic, 1> rJNn;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnm;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnc;
    Eigen::VectorXd u;

    double no_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "no_frames: " << no_frames << std::endl;
    double timestep = 1/fps;
    std::cout << "fps : " << fps << std::endl;
    std::cout << "timestep : " << timestep << std::endl;
    int count = 0;
    double total_time = 0;

    Eigen::VectorXd muf, mup;

    while(cap.isOpened()){
        cap >> frame;
        if(frame.empty()){
            break;
        }
        count++;
        auto t_start = std::chrono::high_resolution_clock::now();
        cv::Mat imgout;
        Eigen::VectorXd xk, yk;
        Eigen::MatrixXd Sf, Sp;

        // ****** 1. Perform time update to current frame time ******/////
        // Calculate prediction density
        // ****** 2. Identify landmarks with matching features ******/////
        std::cout << "scenario" << scenario << std::endl;
        switch(scenario) {
            case 1:
            {
                // std::cout << " Time update " << std::endl;
                timeUpdateContinuous(muEKF, SEKF, u, pm, slamparam, timestep, mup, Sp);
                // ArucoMarkers
                slamparam.landmarks_seen.clear();
                int n_measurements;
                std::vector<Marker> detected_markers;
                // std::cout << " Detect markers " << std::endl;
                detectAndDrawArUco(frame, imgout, detected_markers, param);
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
                        double kappa = 0.05;
                        for(int k = 0; k < 6; k++){
                            SEKF(SEKF.rows()-6+k,SEKF.rows()-6+k) = kappa;
                            Sp(Sp.rows()-6+k,Sp.rows()-6+k) = kappa;
                        }
                        // Add initial good guess
                        Thetanc = mup.block(9,0,3,1);
                        rpy2rot(Thetanc, Rnc);
                        Rnm = Rnc*detected_markers[i].Rcm;
                        rot2rpy(Rnm,Thetanm);
                        rJNn = mup.segment(6,3) + Rnc*detected_markers[i].rMCc;
                        mup.block(mup.rows()-6,0,6,1) << rJNn, Thetanm;
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
                    }
                }
                // Measurement update
                ArucoLogLikelihood aruco_ll;
                std::cout << "frame no. : " << count << std::endl;
                measurementUpdateIEKF(mup, Sp, u, yk, aruco_ll, slamparam, muf, Sf);
                muEKF               = muf;
                SEKF                = Sf;
                break;
            }
            case 2:
            {
                // points
                slamparam.landmarks_seen.clear();
                std::cout << "Scenario 2" << std::endl;
                cv::Mat descriptors_found;
                std::vector<cv::KeyPoint> keypoints_found;
                detectAndDrawORB(frame, imgout, max_features, descriptors_found, keypoints_found);
                //Print some stuff
                std::cout << "Descriptor Width:" << descriptors_found.cols << std::endl;
                std::cout << "Descriptor Height:" << descriptors_found.rows << std::endl;
                // std::cout << "Descriptors :" << descriptors << std::endl;
                for(int i = 0; i < descriptors_found.rows;i++){
                    std::cout << "Keypoint " << i << " description:" << descriptors_found.row(i) << std::endl;
                }

                // Identify landmarks with matching features
                std::vector<cv::DMatch> matches;
                std::vector<int> failed_match_idx;

                for(int i = 0; i < descriptors_found.rows; i++){
                    std::vector<cv::DMatch> match;
                    // check we have landmarks to start with
                    if(landmark_descriptors.rows > 0) {
                        // for each the descriptors found, compare with our state descriptors
                        matcher.match(landmark_descriptors,descriptors_found.row(i),match);
                        std::cout << " CHECHED IF MATCH WAS GOOD" << std::endl;
                    }
                    if(match.size() > 0) {
                        std::cout << " WE FOUND A MATCH WHICH IS POTENTIALLY GOOD" << std::endl;
                        matches.push_back(match[0]);
                    } else {
                        std::cout << "No match found" << std::endl;
                        failed_match_idx.push_back(i);
                    }
                }


                matcher.match(landmark_descriptors,descriptors_found,matches);
                // Store match associations
                Eigen::MatrixXd potential_measurments;
                potential_measurments.resize(2,matches.size());
                for(int i =0; i < matches.size(); i++) {
                    potential_measurments(0,i) = keypoints_found[matches[i].trainIdx].pt.x;
                    potential_measurments(1,i) = keypoints_found[matches[i].trainIdx].pt.y;
                }
                // ------------------------------------------------------------------------
                // Generate chi2LUT
                // ------------------------------------------------------------------------
                double nstd     = 3;
                std::vector<double> chi2LUT;
                double c = 2*normcdf(3) - 1;
                for(int i=0; i < potential_measurments.cols(); i++) {
                        chi2LUT.push_back(chi2inv(c, (i+1)*2));
                        std::cout << chi2LUT[i] << ",";
                }


                //check we have some landmarks initialised already
                if(mup.rows() - 12 > 0) {
                    // Form landmark bundle
                    // ----------------------------------------------------------------
                    std::cout << std::endl;
                    std::cout << "Form landmark bundle" << std::endl;
                    MeasurementPointLandmarkBundle landmarkBundle;
                    // Function handle for use in affine transform
                    auto h  = std::bind(
                                landmarkBundle,
                                std::placeholders::_1,      // x
                                param,
                                std::placeholders::_2,      // h
                                std::placeholders::_3,      // SR
                                std::placeholders::_4);     // C

                    Eigen::VectorXd muY;
                    Eigen::MatrixXd SYY;
                    // ----------------------------------------------------------------
                    // Check compatibility and generated isCompatible flag vector
                    // ----------------------------------------------------------------
                    std::cout << std::endl;
                    std::cout << "Check compatibility" << std::endl;


                    std::cout << "FUCK FUCK FUCK" << std::endl;
                    affineTransform(mup,Sp,h,muY,SYY);

                    std::vector<char> isCompatible;
                    for(int i = 0; i < matches.size(); i++) {
                        bool res = individualCompatibility(i,i,2,potential_measurments,muY,SYY,chi2LUT);
                        isCompatible.push_back(res);
                        int n_measurements = yk.rows();
                        std::cout << "res of individualCompatibility : " << res << std::endl;
                        if(res && potential_measurments(1,i) > 50 && potential_measurments(1,i) < (1920 - 50) && potential_measurments(0,i) > 50 && potential_measurments(0,i) < (1920 - 50)){
                            std::cout << "Pixel at [" << potential_measurments(0,i) << "," << potential_measurments(1,i) << " ] in Image B, matches with landmark " << i << "." <<std::endl;
                            yk.conservativeResizeLike(Eigen::MatrixXd::Zero(n_measurements+2,1));
                            yk.tail(2) = potential_measurments.col(i);
                            slamparam.landmarks_seen.push_back(i);
                            std::cout << "y : " << yk << std::endl;
                        }
                    }
                }

                // Remove failed landmarks from map (consecutive failures to match)

                // Identify surplus features that do not correspond to landmarks in the map

                // Initialise up to Nmax – N new landmarks from best surplus features
                for(int i = 0; i < failed_match_idx.size(); i++) {
                    int max_new_landmarks = max_landmarks + 12 - mup.rows();
                    if(max_new_landmarks > 0) {
                        // Add new landmark description to now seen list
                        int n_states = muEKF.rows();
                        landmark_descriptors.push_back(descriptors_found.row(failed_match_idx[i]));
                        //Reize the state and predicted state matrix
                        SEKF.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+3,n_states+3));
                        Sp.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+3,n_states+3));
                        muEKF.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+3,1));
                        mup.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+3,1));
                        double kappa = 0.1;
                        for(int k = 0; k < 3; k++){
                            SEKF(SEKF.rows()-3+k,SEKF.rows()-3+k) = kappa;
                            Sp(Sp.rows()-3+k,Sp.rows()-3+k) = kappa;
                        }

                        // Add pixel measurements to vector y
                        int n_measurements = yk.rows();
                        yk.conservativeResizeLike(Eigen::MatrixXd::Zero(n_measurements+2,1));
                        Eigen::MatrixXd pixel(2,1);
                        std::cout << "keypoints_found.size() " << keypoints_found.size() << std::endl;
                        pixel << keypoints_found[failed_match_idx[i]].pt.x, keypoints_found[failed_match_idx[i]].pt.y;
                        yk.tail(2) = pixel;

                        // Add initial good guess

                        //YUCK
                        std::vector<cv::Point2f> pixels;
                        std::vector<cv::Point2f> undistorted_pixels;
                        cv::Point2f pixel_to_undistort;
                        std::vector<cv::Point2f> pixels_to_undistort;
                        pixel_to_undistort.x = pixel(0);
                        pixel_to_undistort.y = pixel(1);
                        pixels_to_undistort.push_back(pixel_to_undistort);

                        std::cout << "pixel : " << pixel(0) << "," << pixel(1) << std::endl;

                        cv::undistortPoints(pixels_to_undistort, undistorted_pixels, camera_param.Kc, camera_param.distCoeffs);
                        std::vector<cv::Point3f> homogeneous_pixels;
                        cv::convertPointsHomogeneous(undistorted_pixels,homogeneous_pixels);

                        // optical_ray = optical_ray/optical_ray.norm();
                        // std::cout << "optical_ray before back-project : " << optical_ray << std::endl;

                        Eigen::MatrixXd K(3,3);
                        cv::cv2eigen(camera_param.Kc, K);
                        std::cout << "K : " << K << std::endl;
                        Eigen::MatrixXd inverse_camera_matrix(3,3);
                        inverse_camera_matrix << 1/K(0,0), 0, -K(0,2)/K(0,0),0,1/K(1,1),-K(1,2)/K(1,1),0,0,1;
                        std::cout << "inverse_camera_matrix : " << inverse_camera_matrix << std::endl;
                        // Back-project the pixel into optical ray
                        Eigen::VectorXd optical_ray(3,1);
                        optical_ray << pixel(0),pixel(1),1;
                        optical_ray = inverse_camera_matrix*optical_ray;
                        std::cout << "optical_ray : " << optical_ray << std::endl;
                        std::cout << "optical_ray.norm() : " << optical_ray.norm() << std::endl;

                        //scale it by 2
                        optical_ray = 4*optical_ray;

                        std::cout << "mup " << mup << std::endl;
                        Thetanc = mup.segment(9,3);
                        rpy2rot(Thetanc, Rnc);
                        Eigen::VectorXd point(3,1);
                        point = mup.segment(6,3) + Rnc*optical_ray;
                        mup.tail(3) << point;

                        std::cout << " point in world : " << point << std::endl;

                        // Add index j to landmark seen vector
                        slamparam.landmarks_seen.push_back((n_states-nx)/3);
                        std::cout << " NEW LANDMARK : " << (n_states-nx)/3 << std::endl;
                    }

                }

                // Measurement update
                PointLogLikelihood point_ll;
                std::cout << "frame no. : " << count << std::endl;
                measurementUpdateIEKF(mup, Sp, u, yk, point_ll, slamparam, muf, Sf);
                muEKF               = muf;
                SEKF                = Sf;
                // Remove bad landmarks
                break;
            }
            case 3:
            {
                // code block
                break;
            }
            default:
            {
                // code block
                std::cout << "Invalid scenario" << std::endl;
                return;
            }
        }


        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout << "Time taken for 1 frame [s]: " << elapsed_time_ms/1000 << std::endl;
        total_time += elapsed_time_ms/1000;
        std::cout << "Total Time taken [s]: " << total_time << std::endl;
        std::cout << "World Position (N,E,D) : (" << muEKF(6) <<","<< muEKF(7) << "," << muEKF(8)<< ")" << std::endl;
        std::cout << "World Orientation (phi,theta,psi) : (" << muEKF(9) <<","<< muEKF(10) << "," << muEKF(11)<< ")" << std::endl;


        //**********  Plotting **********//
        muPlot.segment(0,muEKF.rows()) = muEKF;
        SPlot.block(0,0,SEKF.rows(),SEKF.rows()) = SEKF;
        if (interactive == 1) {
            updatePlotStates(imgout, muPlot, SPlot, param, handles,slamparam);
            if(count == no_frames) {
                PlotHandles tmpHandles;
                initPlotStates(muPlot, SPlot, param, tmpHandles,slamparam);
                updatePlotStates(imgout, muPlot, SPlot, param, tmpHandles,slamparam);
                // -------------------------
                // Attach interactor for playing with the 3d interface
                // -------------------------
                vtkNew<vtkInteractorStyleTrackballCamera> threeDimInteractorStyle;
                vtkNew<vtkRenderWindowInteractor> threeDimInteractor;
                threeDimInteractor->SetInteractorStyle(threeDimInteractorStyle);
                threeDimInteractor->SetRenderWindow(tmpHandles.renderWindow);
                threeDimInteractor->Initialize();
                threeDimInteractor->Start();
            }
            cv::Mat video_frame = getPlotFrame(handles);
            video.write(video_frame);
        }
        else if (interactive == 2 && count % 1 == 0)
        {
            PlotHandles tmpHandles;
            // Hack: call twice to get frame to show
            updatePlotStates(imgout, muPlot, SPlot, param, handles,slamparam);
            updatePlotStates(imgout, muPlot, SPlot, param, handles,slamparam);
            cv::Mat video_frame = getPlotFrame(handles);
            video.write(video_frame);
            // -------------------------
            // Attach interactor for playing with the 3d interface
            // -------------------------
            vtkNew<vtkInteractorStyleTrackballCamera> threeDimInteractorStyle;
            vtkNew<vtkRenderWindowInteractor> threeDimInteractor;
            threeDimInteractor->SetInteractorStyle(threeDimInteractorStyle);
            threeDimInteractor->SetRenderWindow(handles.renderWindow);
            threeDimInteractor->Initialize();
            threeDimInteractor->Start();
            initPlotStates(muPlot, SPlot, param, handles,slamparam);

        } else {
            updatePlotStates(imgout, muPlot, SPlot, param, handles, slamparam);
            cv::Mat video_frame = getPlotFrame(handles);
            video.write(video_frame);
        }

        if( count % 3000 == 0) {
            video.release();
            return;
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
    //Relase video
    video.release();
}

cv::Mat getPlotFrame(const PlotHandles &handles)
{
    int *size = handles.renderWindow->GetSize();
    int & w = size[0];
    int & h = size[1];
    std::shared_ptr<unsigned char[]> pixels(handles.renderWindow->GetPixelData(0, 0, w - 1, h - 1, 0));
    cv::Mat frameBufferRGB(h, w, CV_8UC3, pixels.get());
    cv::Mat frameBufferBGR;
    cv::cvtColor(frameBufferRGB, frameBufferBGR, cv::COLOR_RGB2BGR);
    cv::Mat frame;
    cv::flip(frameBufferBGR, frame, 0); // Flip vertically
    return frame;
}


