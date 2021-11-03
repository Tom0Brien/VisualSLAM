#include <filesystem>
#include <string>
#include "SLAM.h"
#include "utility.h"
#include "dataAssociation.h"
#include "measurementPointLandmark.hpp"
#include <opencv2/calib3d.hpp>
#include <memory>

#include <thread>
#include <condition_variable>
#include <mutex>

#include "fmin.hpp"

bool sortByScore(sortedLandmark const& lhs, sortedLandmark const& rhs) {
        return lhs.keypoint.response > rhs.keypoint.response;
}

void runSLAMFromVideo(const std::filesystem::path &videoPath, const std::filesystem::path &cameraDataPath, const CameraParameters & param, Settings& s, int scenario, int interactive, bool hasExport, const std::filesystem::path &outputDirectory)
{
    std::filesystem::path outputPath;
    std::string outputFilename = videoPath.stem().string() + "_out" + videoPath.extension().string();
    outputPath = outputDirectory / outputFilename;
    // Open input video at videoPath
    cv::VideoCapture cap(videoPath.string());
    // Output file name
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

    SlamParameters p;
    CameraParameters camera_param;
    importCalibrationData(cameraDataPath.string(), camera_param);
    p.camera_param = camera_param;

    //Define process model and measurement model
    SlamProcessModel     pm;

    //Initialise the states
    p.camera_states              = 12; // Camera states
    // Initialise filter
    Eigen::VectorXd x0(p.camera_states);
    Eigen::VectorXd muEKF(p.camera_states);
    Eigen::MatrixXd SEKF(p.camera_states, p.camera_states);
    SEKF.fill(0);

    if(scenario == 1) {
        std::cout << "Scenario 1" << std::endl;
        Eigen::MatrixXd position_tune(3,3);
        position_tune.setZero();
        position_tune.diagonal() <<  0.3, 0.3, 0.3;
        Eigen::MatrixXd orientation_tune(3,3);
        orientation_tune.setZero();
        orientation_tune.diagonal() <<  0.2, 0.2, 0.2;
        p.position_tune = position_tune;
        p.orientation_tune = orientation_tune;
        p.n_landmark = 6;
        p.measurement_noise = 20;
        p.max_landmarks = 150;
        p.max_features = 100;
        p.kappa = 0.5;

        muEKF <<        0,                  // x dot
                        0,                  // y dot
                        0,                  // z dot
                        0,                  // Psi dot
                        0,                  // Theta dot
                        0,                  // Phi dot
                        0,                  // x
                        0,                  // y
                        0,               // z
                        -3.14159265359/2,   // Psi
                        3.14159265359,      // Theta
                        0;                  // Phi
        SEKF.diagonal() <<  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01;
    } else if (scenario == 2){
        std::cout << "Scenario 2" << std::endl;
        // PROCESS MODEL
        Eigen::MatrixXd position_tune(3,3);
        position_tune.setZero();
        position_tune.diagonal() <<  0.5, 0.5, 0.175;
        Eigen::MatrixXd orientation_tune(3,3);
        orientation_tune.setZero();
        orientation_tune.diagonal() <<  0.05, 0.05, 0.05;
        p.position_tune = position_tune;
        p.orientation_tune = orientation_tune;
        p.camera_states = 12;

        // MEASUREMENT MODEL
        p.measurement_noise = 7;

        // MAP TUNING
        p.max_landmarks = 50;
        p.max_features = 50000;
        p.max_bad_frames = 10;
        p.feature_thresh = 0.0001;
        p.initial_pixel_distance_thresh = 150;
        p.update_pixel_distance_thresh = 1;
        p.initial_width_thresh = 250;
        p.initial_height_thresh = 100;
        // Initilizing landmarks
        p.optical_ray_length = 8;
        p.kappa = 0.5;

        p.n_landmark = 3;
        // Initial conditions
        SEKF.diagonal() << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001;
        muEKF <<        0,                  // x dot
                        0,                  // y dot
                        0,                  // z dot
                        0,                  // Psi dot
                        0,                  // Theta dot
                        0,                  // Phi dot
                        0,                  // x
                        0,                  // y
                        -1.8,               // z
                        -3.14159265359/2,   // Psi
                        3.14159265359,      // Theta
                        0;                  // Phi
    }

    //Initialize the plot states
    Eigen::VectorXd muPlot(p.camera_states+p.max_landmarks*p.n_landmark);
    muPlot.setZero();
    muPlot.segment(0,12) = muEKF;
    Eigen::MatrixXd SPlot(p.camera_states+p.max_landmarks*p.n_landmark, p.camera_states+p.max_landmarks*p.n_landmark);
    SPlot.setZero();
    SPlot.block(0,0,12,12) = SEKF;
    // Initialize plot states
    PlotHandles handles;
    initPlotStates(muPlot, SPlot, param, handles,p);

    int *size = handles.renderWindow->GetSize();
    int & w = size[0];
    int & h = size[1];
    cv::VideoWriter video(output_file_name,cv::VideoWriter::fourcc('m','p','4','v'),fps, cv::Size(w,h));

    cv::Mat frame;

    // Feature/Landmark storage
    std::vector<int> marker_ids;
    std::vector<Marker> markers;

    //Scenario 2
    std::vector<Landmark> landmarks;

    // ------------------------------------------------------------------------
    // Generate chi2LUT
    // ------------------------------------------------------------------------
    double nstd     = 3;
    std::vector<double> chi2LUT;
    double c = 2*normcdf(3) - 1;
    for(int i=0; i < p.max_landmarks; i++) {
            chi2LUT.push_back(chi2inv(c, (i+1)*2));
    }

    cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING);
    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanj;
    Eigen::Matrix<double, Eigen::Dynamic, 1> rJNn;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnj;
    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanc;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnc;
    Eigen::VectorXd u;
    Eigen::VectorXd muf, mup;

    double no_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    double timestep = 1/fps;
    int count = 0;
    double total_time = 0;

    while(cap.isOpened()){
        cap >> frame;
        if(frame.empty()){
            break;
        }
        count++;
        auto t_start = std::chrono::high_resolution_clock::now();
        cv::Mat imgout;
        Eigen::VectorXd yk;
        Eigen::MatrixXd Sf, Sp;
        switch(scenario) {
            case 1:
            {
                // ******************************************************** TIME UPDATE ********************************************************
                timeUpdateContinuous(muEKF, SEKF, u, pm, p, timestep, mup, Sp);

                // ******************************************************** DETECT MARKERS ********************************************************
                p.landmarks_seen.clear();
                int n_measurements;
                std::vector<Marker> detected_markers;
                detectAndDrawArUco(frame, imgout, detected_markers, param);
                // Check all detected markers, if there is a new marker update the state else if max ID not met add ID to list and initialize a new landmark
                for(int i = 0; i < detected_markers.size(); i++){
                    // Search list of current markers
                    std::vector<int>::iterator it = std::find(marker_ids.begin(), marker_ids.end(), detected_markers[i].id);
                    // If marker was found in our list, update the state
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
                        p.landmarks_seen.push_back(j);
                    } else {
                        // Add new landmark
                        // initializeNewMarker(yk,mup,Sp,muEKF,SEKF,detected_markers,marker_ids,p,i);
                        // Add new landmark to now seen list
                        marker_ids.push_back(detected_markers[i].id);
                        // Resize the state and predicted state matrix
                        int n_states = muEKF.rows();
                        SEKF.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,n_states+6));
                        Sp.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,n_states+6));
                        muEKF.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,1));
                        mup.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,1));
                        // S.conservativeResizeLike(Eigen::VectorXd::Zero(S.rows()+6,1));
                        // S.tail(6) << -1,-1,-1,-1,-1,-1;
                        for(int k = 0; k < 6; k++){
                            SEKF(SEKF.rows()-6+k,SEKF.rows()-6+k) = p.kappa;
                            Sp(Sp.rows()-6+k,Sp.rows()-6+k) = p.kappa;
                        }
                        // Add initial good guess
                        Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanj;
                        Eigen::Matrix<double, Eigen::Dynamic, 1> rJNn;
                        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnj;
                        Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanc;
                        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnc;

                        Thetanc = mup.block(9,0,3,1);
                        rpy2rot(Thetanc, Rnc);
                        Rnj = Rnc*detected_markers[i].Rcm;
                        rot2rpy(Rnj,Thetanj);
                        rJNn = mup.segment(6,3) + Rnc*detected_markers[i].rMCc;
                        mup.block(mup.rows()-6,0,6,1) << rJNn, Thetanj;
                        // Add marker corner measurements to vector yk
                        int n_measurements = yk.rows();
                        yk.conservativeResizeLike(Eigen::MatrixXd::Zero(n_measurements+8,1));
                        for(int c = 0; c < 4; c++) {
                            Eigen::Vector2d rJcNn;
                            rJcNn << detected_markers[i].corners[c].x, detected_markers[i].corners[c].y;
                            yk.block(n_measurements+c*2,0,2,1) = rJcNn;
                        }
                        // Add index j to landmark seen vector
                        p.landmarks_seen.push_back((n_states-p.camera_states)/6);
                    }
                }

                // ******************************************************** MEASUREMENT UPDATE ********************************************************
                // ArucoLogLikelihood aruco_ll;
                arucoLogLikelihoodAnalytical aruco_ll;
                std::cout << "frame no. : " << count << std::endl;
                measurementUpdateIEKFSR1(mup, Sp, u, yk, aruco_ll, p, muf, Sf);
                muEKF   = muf;
                SEKF    = Sf;

                if (SEKF.hasNaN() || muEKF.hasNaN()){
                    measurementUpdateIEKF(mup, Sp, u, yk, aruco_ll, p, muf, Sf);
                    muEKF   = muf;
                    SEKF    = Sf;
                }

                break;
            }
            case 2:
            {
                std::cout << "Scenario 2" << std::endl;
                // ********************************************************  TIME UPDATE ********************************************************
                timeUpdateContinuous(muEKF, SEKF, u, pm, p, timestep, mup, Sp);

                // Reset landmarks_seen for plotting
                p.landmarks_seen.clear();

                // Initialize variables
                cv::Mat descriptors_found;
                std::vector<cv::KeyPoint> keypoints_found;
                std::vector<int> matches_descriptor_idx;
                cv::Mat des;

                // ********************************************************  DETECT FEATURES ********************************************************
                detectAndDrawORB(frame, imgout, p.max_features, descriptors_found, keypoints_found);

                // Identify landmarks with matching features
                std::vector<cv::DMatch> matches;
                cv::Mat landmark_descriptors;
                combineDescriptors(landmark_descriptors,landmarks);
                matcher.match(landmark_descriptors,descriptors_found,matches);
                // Store match associations
                Eigen::MatrixXd potential_measurments;
                potential_measurments.resize(2,matches.size());
                for(int i = 0; i < matches.size(); i++) {
                    potential_measurments(0,i) = keypoints_found[matches[i].trainIdx].pt.x;
                    potential_measurments(1,i) = keypoints_found[matches[i].trainIdx].pt.y;
                }

                // ********************************************************  DATA ASSOCIATION ********************************************************

                if(mup.rows() - 12 > 0) {
                    // ----------------------------------------------------------------
                    // Form landmark bundle
                    // ----------------------------------------------------------------
                    std::cout << std::endl;
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
                    muY.setZero();
                    Eigen::MatrixXd SYY;
                    SYY.setZero();

                    // Check compatibility and generated isCompatible flag vector
                    affineTransform(mup,Sp,h,muY,SYY);
                    std::vector<bool> res;
                    for(int j = 0; j < matches.size(); j++) {
                        bool result = individualCompatibility(j,j,2,potential_measurments,muY,SYY,chi2LUT);
                        res.push_back(result);
                    }

                    // Run surprisal nearest neighbours
                    std::vector<int> idx;
                    snn(mup,Sp,potential_measurments,p.camera_param,idx,false);

                    // Populate matches and isCompatible vectors for drawMatches
                    for(int j = 0; j < idx.size(); j++){
                        cv::DMatch temp;
                        int i = idx[j];
                        bool isMatch = i >= 0;
                        temp.queryIdx = j;
                        temp.trainIdx = i;
                        // Check what current landmarks are visible to the camera
                        Eigen::VectorXd eta = mup.segment(6,6);
                        Eigen::VectorXd rPNn = mup.segment(12+j*3,3);
                        Eigen::VectorXd rQOi; // placeholder
                        bool inCameraCone = (worldToPixel(rPNn,eta,p.camera_param,rQOi) == 0);

                        if(isMatch && res[j]){ //&& !withinRadius
                            cv::putText(imgout,"J="+std::to_string(j),cv::Point(keypoints_found[matches[j].trainIdx].pt.x+10,keypoints_found[matches[j].trainIdx].pt.y+10),cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 0, 0),2);
                            // Store list of descriptor indexs we used for landmark update (to exclude from surplus descriptors later)
                            matches_descriptor_idx.push_back(matches[j].trainIdx);
                            // Update our current landmarks keypoints, desciptor, score, visiblity and pixel measurement for this frame
                            cv::KeyPoint landmark;
                            landmark.pt.x = keypoints_found[matches[j].trainIdx].pt.x;
                            landmark.pt.y = keypoints_found[matches[j].trainIdx].pt.y;
                            landmarks[j].keypoint = landmark;
                            landmarks[j].descriptor = descriptors_found.row(matches[j].trainIdx);
                            landmarks[j].score = 0;
                            landmarks[j].isVisible = true;
                            landmarks[j].pixel_measurement = potential_measurments.col(j);
                        } else {
                            landmarks[j].isVisible = false;
                            // if its in camera cone and failed match, increase the bad association score
                            // if(inCameraCone) {
                                landmarks[j].score += 1;
                            // }
                        }
                    }
                }

                // ******************************************************** REMOVE BAD LANDMARKS ********************************************************
                for(int j = 0; j < landmarks.size(); j++) {
                    if(landmarks[j].score > p.max_bad_frames) {
                        cv::putText(imgout,"DELETED LANDMARK  :  "+std::to_string(j),cv::Point(60,260),cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255),2);
                        removeBadLandmarks(mup,Sp,landmarks,j);
                        // Reset plot states
                        muPlot = Eigen::MatrixXd::Zero(muPlot.rows(),1);
                        SPlot = Eigen::MatrixXd::Zero(muPlot.rows(),muPlot.rows());
                        j = 0;
                    }
                }

                // ******************************************************** ADD NEW LANDMARKS ********************************************************

                // Sort features by score/response
                std::vector<sortedLandmark> sorted_landmarks;
                for(int i = 0; i < descriptors_found.rows; i++) {
                    sortedLandmark lm;
                    lm.keypoint = keypoints_found[i];
                    lm.descriptor = descriptors_found.row(i);
                    sorted_landmarks.push_back(lm);
                }
                std::sort(sorted_landmarks.begin(),sorted_landmarks.end(),&sortByScore);

                for(int i = 0; i < sorted_landmarks.size(); i++) {
                    int max_new_landmarks = p.max_landmarks - (mup.rows()-12)/3;
                    bool featureScore = sorted_landmarks[i].keypoint.response > p.feature_thresh;
                    bool isInWidth  = p.initial_width_thresh <= sorted_landmarks[i].keypoint.pt.x && sorted_landmarks[i].keypoint.pt.x <= p.camera_param.imageSize.width-p.initial_width_thresh;
                    bool isInHeight = p.initial_height_thresh <= sorted_landmarks[i].keypoint.pt.y && sorted_landmarks[i].keypoint.pt.y <= p.camera_param.imageSize.height-p.initial_height_thresh;
                    bool withinRadius = false;
                    // we have atleast one other landmark
                    if(landmarks.size() > 0) {
                        withinRadius = pixelDistance(landmarks,sorted_landmarks[i].keypoint,p.initial_pixel_distance_thresh);
                    }
                    if(max_new_landmarks > 0  && !withinRadius && isInWidth && isInHeight && featureScore) {
                        // Add new landmark
                        initializeNewLandmark(imgout,mup,Sp,muEKF,SEKF,sorted_landmarks,landmarks,p,i);
                    }
                }

                // Add landmarks seen to p.landmarks_seen for plotting
                for(int i = 0; i < landmarks.size(); i++){
                    std::cout <<" score : " << landmarks[i].score << std::endl;
                    if(landmarks[i].isVisible) {
                        p.landmarks_seen.push_back(i);
                    }
                }

                cv::putText(imgout,"frame no. : "+std::to_string(count),cv::Point(60,60),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0),2);
                cv::putText(imgout,"Number of current landmarks "+std::to_string((mup.rows()-12)/3)+"/"+std::to_string(p.max_landmarks),cv::Point(60,100),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0),2);
                cv::putText(imgout,"Number of seen landmarks "+std::to_string(p.landmarks_seen.size()),cv::Point(60,140),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0),2);
                cv::putText(imgout,"World Position (N,E,D) :  ("+std::to_string(muEKF(6))+","+std::to_string(muEKF(7))+","+std::to_string(muEKF(8))+")",cv::Point(60,170),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0),2);
                cv::putText(imgout,"World Orientation (phi,theta,psi)  :  ("+std::to_string(muEKF(9))+","+std::to_string(muEKF(10))+","+std::to_string(muEKF(11))+")",cv::Point(60,200),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0),2);

                // ******************************************************** MEASUREMENT UPDATE ********************************************************
                p.landmarks = landmarks;
                PointLogLikelihood point_ll;
                measurementUpdateIEKF(mup, Sp, u, yk, point_ll, p, muf, Sf);
                // measurementUpdateIEKFSR1(mup, Sp, u, yk, point_ll, p, muf, Sf);
                muEKF   = muf;
                SEKF    = Sf;

                if (SEKF.hasNaN() || muEKF.hasNaN()){
                    measurementUpdateIEKF(mup, Sp, u, yk, point_ll, p, muf, Sf);
                    muEKF   = muf;
                    SEKF    = Sf;
                }
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
        std::cout << "frame no.: " << count << std::endl;
        std::cout << "Time taken for 1 frame [s]: " << elapsed_time_ms/1000 << std::endl;
        total_time += elapsed_time_ms/1000;
        std::cout << "Total Time taken [s]: " << total_time << std::endl;
        std::cout << "World Position (N,E,D) : (" << muEKF(6) <<","<< muEKF(7) << "," << muEKF(8)<< ")" << std::endl;
        std::cout << "World Orientation (phi,theta,psi) : (" << muEKF(9) <<","<< muEKF(10) << "," << muEKF(11)<< ")" << std::endl;

        // ****************************************  Plotting ****************************************
        muPlot.segment(0,muEKF.rows()) = muEKF;
        SPlot.block(0,0,SEKF.rows(),SEKF.rows()) = SEKF;
        if (interactive == 1) {
            updatePlotStates(imgout, muPlot, SPlot, param, handles,p);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if(count == no_frames) { //
                PlotHandles tmpHandles;
                initPlotStates(muPlot, SPlot, param, tmpHandles,p);
                updatePlotStates(imgout, muPlot, SPlot, param, tmpHandles,p);
                vtkNew<vtkInteractorStyleTrackballCamera> threeDimInteractorStyle;
                vtkNew<vtkRenderWindowInteractor> threeDimInteractor;
                threeDimInteractor->SetInteractorStyle(threeDimInteractorStyle);
                threeDimInteractor->SetRenderWindow(tmpHandles.renderWindow);
                threeDimInteractor->Initialize();
                threeDimInteractor->Start();
                video.release();
                return;
            }
            if(hasExport){
                cv::Mat video_frame = getPlotFrame(handles);
                video.write(video_frame);
            }
        }
        else if ((interactive == 2 && count % 1 == 0) || count == 450)
        {
            // Hack: call twice to get frame to show
            updatePlotStates(imgout, muPlot, SPlot, param, handles,p);
            updatePlotStates(imgout, muPlot, SPlot, param, handles,p);
            cv::Mat video_frame = getPlotFrame(handles);
            video.write(video_frame);
            vtkNew<vtkInteractorStyleTrackballCamera> threeDimInteractorStyle;
            vtkNew<vtkRenderWindowInteractor> threeDimInteractor;
            threeDimInteractor->SetInteractorStyle(threeDimInteractorStyle);
            threeDimInteractor->SetRenderWindow(handles.renderWindow);
            threeDimInteractor->Initialize();
            threeDimInteractor->Start();
            initPlotStates(muPlot, SPlot, param, handles,p);

        } else {
            updatePlotStates(imgout, muPlot, SPlot, param, handles, p);
            if(hasExport){
                cv::Mat video_frame = getPlotFrame(handles);
                video.write(video_frame);
            }
        }

        if (SEKF.hasNaN() || muEKF.hasNaN()){
            updatePlotStates(imgout, muPlot, SPlot, param, handles,p);
            video.release();
            vtkNew<vtkInteractorStyleTrackballCamera> threeDimInteractorStyle;
            vtkNew<vtkRenderWindowInteractor> threeDimInteractor;
            threeDimInteractor->SetInteractorStyle(threeDimInteractorStyle);
            threeDimInteractor->SetRenderWindow(handles.renderWindow);
            threeDimInteractor->Initialize();
            threeDimInteractor->Start();
            std::cout << "(╯°□°）╯︵ ┻━┻ " << std::endl;
            std::cout << "NaNs encountered in muEKF. muEKF = \n" << muEKF << std::endl;
            std::cout << "NaNs encountered in muEKF. muEKF.rows() = \n" << muEKF.rows() << std::endl;
            std::cout << "NaNs encountered in muEKF. muEKF.cols() = \n" << muEKF.cols() << std::endl;
            std::cout << "NaNs encountered in SEKF. S = \n" << SEKF << std::endl;
            std::cout << "NaNs encountered in SEKF. S.rows() = \n" << SEKF.rows() << std::endl;
            std::cout << "NaNs encountered in SEKF. S.cols() = \n" << SEKF.cols() << std::endl;
            std::cout << "(╯°□°）╯︵ ┻━┻ " << std::endl;
            return;
        }

    }
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

bool pixelDistance(std::vector<Landmark> & landmarks, cv::KeyPoint keypoint, double pixel_distance_thresh){
    //loop through the all the current keypoints initisised and check if its far away
    bool withinRadius = false;
    cv::Point2f b(keypoint.pt.x, keypoint.pt.y);
    // loop through the landmarks just initilised and all the current landmarks
    for(int i; i < landmarks.size(); i++){
        cv::Point2f a(landmarks[i].keypoint.pt.x, landmarks[i].keypoint.pt.y);
        cv::Point2f diff = a - b;
        double dist = cv::sqrt(diff.x*diff.x + diff.y*diff.y);
        if(dist < pixel_distance_thresh) {
            withinRadius = true;
        }
    }
    return withinRadius;
}

bool pixelDistance(std::vector<Landmark> & landmarks, cv::KeyPoint keypoint, double pixel_distance_thresh, int landmark_index){
    //loop through the all the current keypoints initisised and check if its far away
    bool withinRadius = false;
    cv::Point2f b(keypoint.pt.x, keypoint.pt.y);
    // loop through the landmarks just initilised and all the current landmarks
    for(int j; j < landmarks.size(); j++){
        if(landmark_index != j) {
            cv::Point2f a(landmarks[j].keypoint.pt.x, landmarks[j].keypoint.pt.y);
            cv::Point2f diff = a - b;
            double dist = cv::sqrt(diff.x*diff.x + diff.y*diff.y);
            if(dist < pixel_distance_thresh) {
                withinRadius = true;
            }
        }
    }
    return withinRadius;
}

void removeBadLandmarks(Eigen::VectorXd & mup, Eigen::MatrixXd & Sp, std::vector<Landmark> & landmarks, int j) {
    int camera_states = 12; //camera states
    // remove landmark from vector
    landmarks.erase(landmarks.begin()+j);

    // remove landmark keypoints
    for(int i = 0; i < 3; i++){
        //remove landmark states from mu
        removeRow(mup,camera_states+j*3);
        //remove the columns of the covariance matrix for landmark
        removeColumn(Sp,camera_states+j*3);
    }

    // perform QR decomposition on S to get correct dimensions
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Sp);
    Eigen::MatrixXd R;
    R.setZero();
    R = qr.matrixQR().triangularView<Eigen::Upper>();
    // std::cout << "R" << R << std::endl;
    Eigen::MatrixXd S(Sp.cols(), Sp.cols());
    S.setZero();
    S = R.block(0,0,Sp.cols(),Sp.cols());
    //overwrite the old covariance
    Sp = S;
}

void removeRow(Eigen::VectorXd & vector, unsigned int rowToRemove)
{
    unsigned int numRows = vector.rows()-1;
    unsigned int numCols = vector.cols();

    if( rowToRemove < numRows )
        vector.block(rowToRemove,0,numRows-rowToRemove,numCols) = vector.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    vector.conservativeResize(numRows,numCols);
}

void removeColumn(Eigen::MatrixXd & matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

void  combineDescriptors(cv::Mat & landmark_descriptors, std::vector<Landmark> & landmarks) {
    for(int i = 0; i < landmarks.size(); i++) {
        landmark_descriptors.push_back(landmarks[i].descriptor);
    }
}

Eigen::MatrixXd generateOpticalRay(const CameraParameters & param, Eigen::MatrixXd & pixel, SlamParameters p) {
    cv::Mat P;
    Eigen::MatrixXd P_eig(3,4);
    P_eig.setZero();
    Eigen::MatrixXd K;
    cv::cv2eigen(param.Kc,K);
    P_eig.block(0,0,3,3) = K;
    cv::eigen2cv(P_eig,P);

    std::vector<cv::Point2f> pixels_to_undistort;
    cv::Point2f pixel_to_undistort;
    std::vector<cv::Point2f> undistorted_pixels;
    pixel_to_undistort.x = pixel(0);
    pixel_to_undistort.y = pixel(1);
    pixels_to_undistort.push_back(pixel_to_undistort);
    cv::undistortPoints(pixels_to_undistort, undistorted_pixels, param.Kc, param.distCoeffs, cv::Mat::eye(3,3, CV_64F), P);

    Eigen::MatrixXd pix(3,1);
    pix << undistorted_pixels[0].x,undistorted_pixels[0].y,1;

    Eigen::MatrixXd Kinv;
    cv::cv2eigen(param.Kc.inv(),Kinv);
    Eigen::MatrixXd uPCc(3,1);

    uPCc = Kinv*pix;
    Eigen::MatrixXd rPCc = p.optical_ray_length*uPCc;
    return rPCc;
}

void initializeNewLandmark(cv::Mat & img, Eigen::VectorXd & mup, Eigen::MatrixXd & Sp,Eigen::VectorXd & muEKF, Eigen::MatrixXd & SEKF, std::vector<sortedLandmark> sorted_landmarks, std::vector<Landmark> & landmarks, SlamParameters p, int i) {
    int n_states = mup.rows();
    //Reize the state and predicted state matrix
    SEKF.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+3,n_states+3));
    Sp.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+3,n_states+3));
    muEKF.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+3,1));
    mup.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+3,1));
    for(int k = 0; k < 3; k++){
        SEKF(SEKF.rows()-3+k,SEKF.rows()-3+k) = p.kappa;
        Sp(Sp.rows()-3+k,Sp.rows()-3+k) = p.kappa;
    }

    // Add initial good guess
    Eigen::MatrixXd pixel(2,1);
    pixel << sorted_landmarks[i].keypoint.pt.x, sorted_landmarks[i].keypoint.pt.y;
    Eigen::MatrixXd rPCc = generateOpticalRay(p.camera_param, pixel, p);
    // Scale this unit vector
    // Transform to world space
    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanc;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnc;
    Thetanc = mup.segment(9,3);
    rpy2rot(Thetanc, Rnc);
    Eigen::VectorXd point(3,1);
    point = mup.segment(6,3) + Rnc*rPCc;
    mup.tail(3) << point;

    // Add new landmark to our landmark list
    Landmark new_landmark;
    new_landmark.keypoint = sorted_landmarks[i].keypoint;
    new_landmark.descriptor = sorted_landmarks[i].descriptor;
    new_landmark.pixel_measurement = pixel;
    new_landmark.isVisible = true;
    landmarks.push_back(new_landmark);

    // Plot the initilized landmark to screen
    cv::putText(img,"J="+std::to_string(landmarks.size()),cv::Point(new_landmark.keypoint.pt.x + 30,new_landmark.keypoint.pt.y+30),cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 100, 255),2);
}


void initializeNewMarker(Eigen::VectorXd & yk, Eigen::VectorXd & mup, Eigen::MatrixXd & Sp,Eigen::VectorXd & muEKF, Eigen::MatrixXd & SEKF,std::vector<Marker> & detected_markers, std::vector<int> & marker_ids, SlamParameters p, int i) {
    // Add new landmark to now seen list
    marker_ids.push_back(detected_markers[i].id);
    // Resize the state and predicted state matrix
    int n_states = muEKF.rows();
    SEKF.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,n_states+6));
    Sp.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,n_states+6));
    muEKF.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,1));
    mup.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+6,1));
    // S.conservativeResizeLike(Eigen::VectorXd::Zero(S.rows()+6,1));
    // S.tail(6) << -1,-1,-1,-1,-1,-1;
    for(int k = 0; k < 6; k++){
        SEKF(SEKF.rows()-6+k,SEKF.rows()-6+k) = p.kappa;
        Sp(Sp.rows()-6+k,Sp.rows()-6+k) = p.kappa;
    }
    // Add initial good guess
    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanj;
    Eigen::Matrix<double, Eigen::Dynamic, 1> rJNn;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnj;
    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanc;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnc;

    Thetanc = mup.block(9,0,3,1);
    rpy2rot(Thetanc, Rnc);
    Rnj = Rnc*detected_markers[i].Rcm;
    rot2rpy(Rnj,Thetanj);
    rJNn = mup.segment(6,3) + Rnc*detected_markers[i].rMCc;
    mup.block(mup.rows()-6,0,6,1) << rJNn, Thetanj;
    // Add marker corner measurements to vector yk
    int n_measurements = yk.rows();
    yk.conservativeResizeLike(Eigen::MatrixXd::Zero(n_measurements+8,1));
    for(int c = 0; c < 4; c++) {
        Eigen::Vector2d rJcNn;
        rJcNn << detected_markers[i].corners[c].x, detected_markers[i].corners[c].y;
        yk.block(n_measurements+c*2,0,2,1) = rJcNn;
    }
    // Add index j to landmark seen vector
    p.landmarks_seen.push_back((n_states-p.camera_states)/6);
}
