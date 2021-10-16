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

std::mutex mut;
std::condition_variable cond_v;
bool ready = false;


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
    //Initialise the states
    int nx, ny;
    nx              = 12; // Camera states
    ny              = 0;
    // Initialise filter
    Eigen::VectorXd x0(nx);
    Eigen::VectorXd muEKF(nx);
    Eigen::MatrixXd SEKF(nx, nx);
    SEKF.fill(0);
    std::cout << "scenario" << scenario << std::endl;



    int max_landmarks;
    double kappa;
    double optical_ray_length;
    int max_features;
    int max_bad_frames;
    double feature_thresh;
    double initial_pixel_distance_thresh;
    double update_pixel_distance_thresh;
    if(scenario == 1) {
        std::cout << "Scenario 1" << std::endl;
        slamparam.position_tune = 0.3;
        slamparam.orientation_tune = 0.2;
        slamparam.n_landmark = 6;
        slamparam.measurement_noise = 20;
        max_landmarks = 150;
        max_features = 100;
        kappa = 0.5;
        muEKF <<        0, // x dot
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
                        0  ; // Phi
        SEKF.diagonal() <<  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01;
    } else if (scenario == 2){
        std::cout << "Scenario 2" << std::endl;
        // PROCESS MODEL
        slamparam.position_tune = 0.3;
        slamparam.orientation_tune = 0.2;
        // MEASUREMENT MODEL
        slamparam.measurement_noise = 7;
        // Map tuning
        max_landmarks = 30;
        max_features = 20000;
        max_bad_frames = 5;
        feature_thresh = 0.0002;
        initial_pixel_distance_thresh = 100;
        update_pixel_distance_thresh = 25;
        // Initilizing landmarks
        optical_ray_length = 3.5;
        kappa = 1;

        slamparam.n_landmark = 3;
        // Initial conditions
        SEKF.diagonal() << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0001, 0.0001, 0.0001, 0.0001, 0.001, 0.0001;
        muEKF <<        0, // x dot
                        0, // y dot
                        0, // z dot
                        0, // Psi dot
                        0, // Theta dot
                        0, // Phi dot
                        0, // x
                        0, // y
                        -1.8, //-1.8, // z
                        -3.14159265359/2, // Psi
                        3.14159265359, // Theta
                        0  ; // Phi
    }

    //Initialize the plot states
    Eigen::VectorXd muPlot(nx+max_landmarks*slamparam.n_landmark);
    muPlot.setZero();
    muPlot.segment(0,12) = muEKF;
    Eigen::MatrixXd SPlot(nx+max_landmarks*slamparam.n_landmark, nx+max_landmarks*slamparam.n_landmark);
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
    std::vector<Marker> markers;

    //Scenario 2
    std::vector<Landmark> landmarks;

    // ------------------------------------------------------------------------
    // Generate chi2LUT
    // ------------------------------------------------------------------------
    double nstd     = 3;
    std::vector<double> chi2LUT;
    double c = 2*normcdf(3) - 1;
    for(int i=0; i < max_landmarks; i++) {
            chi2LUT.push_back(chi2inv(c, (i+1)*2));
    }

    cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING);

    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanc;
    Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanj;
    Eigen::Matrix<double, Eigen::Dynamic, 1> rJNn;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnj;
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
                        for(int k = 0; k < 6; k++){
                            SEKF(SEKF.rows()-6+k,SEKF.rows()-6+k) = kappa;
                            Sp(Sp.rows()-6+k,Sp.rows()-6+k) = kappa;
                        }
                        // Add initial good guess
                        Thetanc = mup.block(9,0,3,1);
                        rpy2rot(Thetanc, Rnc);
                        Rnj = Rnc*detected_markers[i].Rcm;
                        rot2rpy(Rnj,Thetanj);
                        rJNn = mup.segment(6,3) + Rnc*detected_markers[i].rMCc;
                        mup.block(mup.rows()-6,0,6,1) << rJNn, Thetanj;
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

                measurementUpdateIEKFSR1(mup, Sp, u, yk, aruco_ll, slamparam, muf, Sf);
                muEKF   = muf;
                SEKF    = Sf;

                if (SEKF.hasNaN() || muEKF.hasNaN()){

                    measurementUpdateIEKF(mup, Sp, u, yk, aruco_ll, slamparam, muf, Sf);
                    muEKF   = muf;
                    SEKF    = Sf;
                }

                // ready = false;
                // std::thread t1{[&mup, &Sp, &u, &yk, &aruco_ll, &slamparam, &muf, &Sf, &muEKF, &SEKF]
                //     {
                //         measurementUpdateIEKFSR1(mup, Sp, u, yk, aruco_ll, slamparam, muf, Sf);
                //         std::unique_lock<std::mutex> lk(mut);
                //         if(!ready) {
                //         std::cout << "sheesh 1" << std::endl;
                //             muEKF   = muf;
                //             SEKF    = Sf;
                //         } else {
                //             return;
                //         }
                //         std::cout << "measurementUpdateIEKFSR1 finished" << std::endl;
                //         ready = true;
                //         cond_v.notify_one();
                //     }};

                // std::thread t2{[&mup, &Sp, &u, &yk, &aruco_ll, &slamparam, &muf, &Sf, &muEKF, &SEKF]
                //     {
                //     measurementUpdateIEKF(mup, Sp, u, yk, aruco_ll, slamparam, muf, Sf);
                //     std::unique_lock<std::mutex> lk(mut);
                //     if(!ready) {
                //         std::cout << "sheesh 2" << std::endl;
                //         muEKF   = muf;
                //         SEKF    = Sf;
                //     } else {
                //         return;
                //     }
                //     std::cout << "measurementUpdateIEKF finished" << std::endl;
                //     ready = true;
                //     cond_v.notify_one();
                //     }};

                // std::cout << "2" << std::endl;
                // {
                // std::unique_lock<std::mutex> lk(mut);
                // while (!ready)
                //     cond_v.wait(lk);
                // }

                // std::cout << "3" << std::endl;

                // // she'll be right
                // ready = false;
                // t1.join();
                // t2.join();

                break;
            }
            case 2:
            {
                //Time update!
                timeUpdateContinuous(muEKF, SEKF, u, pm, slamparam, timestep, mup, Sp);
                // points
                slamparam.landmarks_seen.clear();
                std::cout << "Scenario 2" << std::endl;
                cv::Mat descriptors_found;
                std::vector<cv::KeyPoint> keypoints_found;
                std::vector<int> matches_descriptor_idx;
                cv::Mat des;
                detectAndDrawORB(frame, imgout, max_features, descriptors_found, keypoints_found);

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

                //check we have some landmarks initialised already
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
                    Eigen::MatrixXd SYY;
                    // ----------------------------------------------------------------
                    // Check compatibility and generated isCompatible flag vector
                    // ----------------------------------------------------------------
                    affineTransform(mup,Sp,h,muY,SYY);
                    std::vector<bool> res;
                    for(int j = 0; j < matches.size(); j++) {
                        bool result = individualCompatibility(j,j,2,potential_measurments,muY,SYY,chi2LUT);
                        res.push_back(result);
                    }

                    // ------------------------------------------------------------------------
                    // Run surprisal nearest neighbours
                    // ------------------------------------------------------------------------

                    std::cout << "Run surprisal nearest neighbours." << std::endl;
                    std::cout << "mup.rows()" << mup.rows() << std::endl;
                    std::cout << "mup.cols()" << mup.cols() << std::endl;
                    // TODO
                    std::vector<int> idx;
                    snn(mup,Sp,potential_measurments,slamparam.camera_param,idx,false);

                    // ------------------------------------------------------------------------
                    // Populate matches and isCompatible vectors for drawMatches
                    // ------------------------------------------------------------------------
                    std::cout << "Populate matches and isCompatible vectors for drawMatches." << std::endl;
                    // std::vector< cv::DMatch > matches;
                    // TODO
                    for(int j = 0; j < idx.size(); j++){
                        cv::DMatch temp;
                        int i = idx[j];
                        bool isMatch = i >= 0;
                        temp.queryIdx = j;
                        temp.trainIdx = i;
                        bool withinRadius = pixelDistance(landmarks,keypoints_found[matches[j].trainIdx],update_pixel_distance_thresh,j);
                        // matches.push_back(temp);
                        if(isMatch && res[j] && !withinRadius){
                            std::cout << "Pixel " << i << " in y located at [ " << potential_measurments(0,i) << "," << potential_measurments(1,i) << "] in imageB, matches with landmark " << j << "." << std::endl;
                            // std::cout << "Pixel at [" << potential_measurments(0,j) << "," << potential_measurments(1,j) << " ] in Image B, matches with landmark " << j << "." <<std::endl;
                            cv::putText(imgout,"J="+std::to_string(j),cv::Point(keypoints_found[matches[j].trainIdx].pt.x+10,keypoints_found[matches[j].trainIdx].pt.y+10),cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 0, 0),2);

                            // store list of descriptor indexs we used for landmark update (to exclude from surplus descriptors later)
                            matches_descriptor_idx.push_back(matches[j].trainIdx);

                            //Update our landmarks keypoints, score, visiblity and pixel measurement for this frame
                            cv::KeyPoint temp;
                            temp.pt.x = keypoints_found[matches[j].trainIdx].pt.x;
                            temp.pt.y = keypoints_found[matches[j].trainIdx].pt.y;
                            landmarks[j].keypoint = temp;  // (to avoid initilizing new landmarks on top of current landmarks)
                            landmarks[j].score = 0;
                            landmarks[j].isVisible = true;
                            landmarks[j].pixel_measurement = potential_measurments.col(j);
                        } else {
                            landmarks[j].isVisible = false;
                            landmarks[j].score += 1;
                        }
                    }
                }

                // //Remove failed landmarks from map (consecutive failures to match)
                for(int j = 0; j < landmarks.size(); j++) {
                    if(landmarks[j].score > max_bad_frames) {
                        // remove all the things
                        std::cout << "REMOVE LANDMARK : " << j << std::endl;
                        cv::putText(imgout,"DELETED LANDMARK  :  "+std::to_string(j),cv::Point(60,260),cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255),2);
                        removeBadLandmarks(mup,Sp,landmarks,j);
                        //Reset plot states
                        muPlot = Eigen::MatrixXd::Zero(muPlot.rows(),1);
                        SPlot = Eigen::MatrixXd::Zero(muPlot.rows(),muPlot.rows());
                        j = 0;
                    }
                }

                // Identify surplus features that do not correspond to landmarks in the map
                // Initialise up to Nmax – N new landmarks from best surplus features
                for(int i = 0; i < descriptors_found.rows; i++) {
                    int max_new_landmarks = max_landmarks - (mup.rows()-12)/3;
                    // if index in the list of descriptor of the indexs the matches found dont initilize
                    bool found = std::find(matches_descriptor_idx.begin(), matches_descriptor_idx.end(), i) != matches_descriptor_idx.end();
                    bool featureScore = keypoints_found[i].response > feature_thresh;
                    double width_thresh = 200;
                    double height_thresh = 100;
                    bool isInWidth  = width_thresh <= keypoints_found[i].pt.x && keypoints_found[i].pt.x <= slamparam.camera_param.imageSize.width-width_thresh;
                    bool isInHeight = height_thresh <= keypoints_found[i].pt.y && keypoints_found[i].pt.y <= slamparam.camera_param.imageSize.height-height_thresh;
                    bool withinRadius = false;
                    // we have atleast one landmark
                    if(landmarks.size() > 0) {
                        withinRadius = pixelDistance(landmarks,keypoints_found[i],initial_pixel_distance_thresh);
                    }
                    // std::cout << "score : " << score << std::endl;
                    // std::cout << " found : " << found << std::endl;
                    if(max_new_landmarks > 0  && !found && featureScore && !withinRadius && isInWidth && isInHeight) { //
                        // Add pixel measurements to vector y
                        Eigen::MatrixXd pixel(2,1);
                        pixel << keypoints_found[i].pt.x, keypoints_found[i].pt.y;
                        int n_states = mup.rows();
                        //Reize the state and predicted state matrix
                        SEKF.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+3,n_states+3));
                        Sp.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+3,n_states+3));
                        muEKF.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+3,1));
                        mup.conservativeResizeLike(Eigen::MatrixXd::Zero(n_states+3,1));
                        for(int k = 0; k < 3; k++){
                            SEKF(SEKF.rows()-3+k,SEKF.rows()-3+k) = kappa;
                            Sp(Sp.rows()-3+k,Sp.rows()-3+k) = kappa;
                        }
                        // Add initial good guess
                        cv::Mat P;
                        Eigen::MatrixXd P_eig(3,4);
                        P_eig.setZero();
                        Eigen::MatrixXd K;
                        cv::cv2eigen(slamparam.camera_param.Kc,K);
                        P_eig.block(0,0,3,3) = K;
                        cv::eigen2cv(P_eig,P);

                        std::vector<cv::Point2f> pixels_to_undistort;
                        cv::Point2f pixel_to_undistort;
                        std::vector<cv::Point2f> undistorted_pixels;
                        pixel_to_undistort.x = pixel(0);
                        pixel_to_undistort.y = pixel(1);
                        pixels_to_undistort.push_back(pixel_to_undistort);
                        cv::undistortPoints(pixels_to_undistort, undistorted_pixels, slamparam.camera_param.Kc, slamparam.camera_param.distCoeffs, cv::Mat::eye(3,3, CV_64F), P);

                        Eigen::MatrixXd pix(3,1);
                        pix << undistorted_pixels[0].x,undistorted_pixels[0].y,1;

                        Eigen::MatrixXd Kinv;
                        cv::cv2eigen(slamparam.camera_param.Kc.inv(),Kinv);
                        Eigen::MatrixXd rPCc_eig(3,1);

                        rPCc_eig = Kinv*pix;

                        rPCc_eig = optical_ray_length*rPCc_eig;
                        std::cout << "optical_ray : " << rPCc_eig << std::endl;
                        //scale it by 2
                        Thetanc = mup.segment(9,3);
                        rpy2rot(Thetanc, Rnc);
                        Eigen::VectorXd point(3,1);
                        point = mup.segment(6,3) + Rnc*rPCc_eig;
                        mup.tail(3) << point;

                        // Add new landmark to our landmark list
                        Landmark new_landmark;
                        new_landmark.keypoint.pt.x = keypoints_found[i].pt.x;
                        new_landmark.keypoint.pt.y = keypoints_found[i].pt.y;
                        new_landmark.descriptor = descriptors_found.row(i);
                        new_landmark.pixel_measurement = pixel;
                        new_landmark.isVisible = true;
                        landmarks.push_back(new_landmark);
                        cv::putText(imgout,"J="+std::to_string(landmarks.size()),cv::Point(new_landmark.keypoint.pt.x + 30,new_landmark.keypoint.pt.y+30),cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 100, 255),2);
                    }
                }

                //TEMP: FIX THIS
                for(int i = 0; i < landmarks.size(); i++){
                    std::cout <<" score : " << landmarks[i].score << std::endl;
                    if(landmarks[i].isVisible) {
                        slamparam.landmarks_seen.push_back(i);
                    }
                }

                cv::putText(imgout,"frame no. : "+std::to_string(count),cv::Point(60,60),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0),2);
                cv::putText(imgout,"Number of current landmarks "+std::to_string((mup.rows()-12)/3)+"/"+std::to_string(max_landmarks),cv::Point(60,100),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0),2);
                cv::putText(imgout,"Number of seen landmarks "+std::to_string(slamparam.landmarks_seen.size()),cv::Point(60,140),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0),2);
                cv::putText(imgout,"World Position (N,E,D) :  ("+std::to_string(muEKF(6))+","+std::to_string(muEKF(7))+","+std::to_string(muEKF(8))+")",cv::Point(60,170),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0),2);
                cv::putText(imgout,"World Orientation (phi,theta,psi)  :  ("+std::to_string(muEKF(9))+","+std::to_string(muEKF(10))+","+std::to_string(muEKF(11))+")",cv::Point(60,200),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0),2);
                // Measurement update
                slamparam.landmarks = landmarks;
                PointLogLikelihood point_ll;
                measurementUpdateIEKFSR1(mup, Sp, u, yk, point_ll, slamparam, muf, Sf);
                muEKF   = muf;
                SEKF    = Sf;

                if (SEKF.hasNaN() || muEKF.hasNaN()){

                    measurementUpdateIEKF(mup, Sp, u, yk, point_ll, slamparam, muf, Sf);
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
        //**********  Plotting **********//
        muPlot.segment(0,muEKF.rows()) = muEKF;
        SPlot.block(0,0,SEKF.rows(),SEKF.rows()) = SEKF;
        if (interactive == 1) {
            updatePlotStates(imgout, muPlot, SPlot, param, handles,slamparam);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if(count == 500) { //no_frames
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
                video.release();
                return;
            }
            cv::Mat video_frame = getPlotFrame(handles);
            video.write(video_frame);
        }
        else if (interactive == 2 && count % 1 == 0)
        {
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

        if (SEKF.hasNaN() || muEKF.hasNaN()){
            updatePlotStates(imgout, muPlot, SPlot, param, handles,slamparam);
            video.release();
            // -------------------------
            // Attach interactor for playing with the 3d interface
            // -------------------------
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
    int nx = 12; //camera states
    // remove landmark from vector
    landmarks.erase(landmarks.begin()+j);

    // remove landmark keypoints
    for(int i = 0; i < 3; i++){
        //remove landmark states from mu
        removeRow(mup,nx+j*3);
        //remove the columns of the covariance matrix for landmark
        removeColumn(Sp,nx+j*3);
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
