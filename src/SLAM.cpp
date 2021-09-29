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
    double optical_ray_lengh;
    int max_features;
    if(scenario == 1) {
        std::cout << "Scenario 1" << std::endl;
        slamparam.position_tune = 0.1;
        slamparam.orientation_tune = 0.1;
        slamparam.n_landmark = 6;
        max_landmarks = 150;
        max_features = 100;
        kappa = 1;
        muEKF <<        0, // x dot
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
        SEKF.diagonal() << 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01;
    } else if (scenario == 2){
        std::cout << "Scenario 2" << std::endl;
        slamparam.position_tune = 0.01;
        slamparam.orientation_tune = 0.01;
        slamparam.n_landmark = 3;
        max_landmarks = 15;
        max_features = 2000;
        optical_ray_lengh = 4;
        kappa = 1.2;
        slamparam.measurement_noise = 2;
        SEKF.diagonal() << 0.25, 0.25, 0.25, 0.01, 0.01, 0.01, 0.025,0.025,0.025,0.025,0.025,0.025;
        muEKF <<        0, // x dot
                        0, // y dot
                        0, // z dot
                        0, // Psi dot
                        0, // Theta dot
                        0, // Phi dot
                        0, // x
                        0, // y
                        0, //-1.8, // z
                        -3.14159265359/2, // Phi
                        3.14159265359, // Theta
                        0  ; // Psi
    }

    //Initialize the plot states
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
    std::vector<cv::KeyPoint> landmark_keypoints;
    std::vector<int> bad_landmark;

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
                matcher.match(landmark_descriptors,descriptors_found,matches);
                // Store match associations
                Eigen::MatrixXd potential_measurments;
                potential_measurments.resize(2,matches.size());
                for(int i = 0; i < matches.size(); i++) {
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
                    // ----------------------------------------------------------------
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

                    affineTransform(mup,Sp,h,muY,SYY);
                    std::vector<char> isCompatible;
                    for(int j = 0; j < matches.size(); j++) {
                        bool res = individualCompatibility(j,j,2,potential_measurments,muY,SYY,chi2LUT);
                        isCompatible.push_back(res);
                        int n_measurements = yk.rows();
                        double thresh = 25;
                        bool isInWidth  = thresh <= potential_measurments(0,j) && potential_measurments(0,j) <= slamparam.camera_param.imageSize.width-thresh;
                        bool isInHeight = thresh <= potential_measurments(1,j) && potential_measurments(1,j) <= slamparam.camera_param.imageSize.height-thresh;
                        if(res){
                            // std::cout << "Pixel at [" << potential_measurments(0,j) << "," << potential_measurments(1,j) << " ] in Image B, matches with landmark " << j << "." <<std::endl;
                            cv::putText(imgout,"J="+std::to_string(j),cv::Point(keypoints_found[matches[j].trainIdx].pt.x+10,keypoints_found[matches[j].trainIdx].pt.y+10),cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 0, 0),2);
                            yk.conservativeResizeLike(Eigen::MatrixXd::Zero(n_measurements+2,1));
                            yk.tail(2) = potential_measurments.col(j);
                            slamparam.landmarks_seen.push_back(j);
                            // std::cout << "y : " << yk << std::endl;
                            matches_descriptor_idx.push_back(matches[j].trainIdx);

                            //Update our landmarks keypoints for this frame (to avoid initilizing new landmarks on top of current landmarks)
                            cv::KeyPoint temp;
                            temp.pt.x = keypoints_found[matches[j].trainIdx].pt.x;
                            temp.pt.y = keypoints_found[matches[j].trainIdx].pt.y;
                            landmark_keypoints[j] = temp;
                            bad_landmark[j] -= 1;
                        } else {
                            std::cout << "individualCompatibility failed" << std::endl;
                            bad_landmark[j] += 1;
                        }
                    }
                }

                // Remove failed landmarks from map (consecutive failures to match)
                for(int j = 0; j < bad_landmark.size(); j++) {
                            if(bad_landmark[j] > 5) {
                                // remove all nessacary things
                                std::cout << " BAD BOI " << std::endl;
                                // std::cout << " mup" << std::endl;
                                std::cout << " mup.rows()" << mup.rows() << std::endl;
                                for(int i = 0; i < slamparam.landmarks_seen.size(); i++) {
                                    std::cout << " slamparam.landmarks_seen[i] : " << slamparam.landmarks_seen[i] << std::endl;
                                }
                                removeBadLandmarks(mup,Sp,landmark_keypoints,landmark_descriptors,slamparam.landmarks_seen,bad_landmark,j);
                                std::cout << "REMOVED LANDMARK : "<< j << std::endl;
                                for(int i = 0; i < slamparam.landmarks_seen.size(); i++) {
                                    std::cout << " slamparam.landmarks_seen[i] : " << slamparam.landmarks_seen[i] << std::endl;
                                }
                                std::cout << " mup.rows()" << mup.rows() << std::endl;
                                //Reset plot states
                                muPlot.setZero();
                                SPlot.setZero();
                            }
                }

                // Identify surplus features that do not correspond to landmarks in the map
                // Initialise up to Nmax – N new landmarks from best surplus features
                for(int i = 0; i < descriptors_found.rows; i++) {
                    int max_new_landmarks = max_landmarks - (mup.rows()-12)/3;
                    // if index in the list of descriptor of the indexs the matches found dont initilize
                    bool found = std::find(matches_descriptor_idx.begin(), matches_descriptor_idx.end(), i) != matches_descriptor_idx.end();
                    double score = keypoints_found[i].response;
                    double feature_thresh = 0.0003;
                    bool distance_check = true;
                    if(landmark_keypoints.size() > 0) {
                        distance_check = pixelDistance(landmark_keypoints,keypoints_found[i]);
                    }
                    // std::cout << "score : " << score << std::endl;
                    // std::cout << " found : " << found << std::endl;
                    if(max_new_landmarks > 0  && !found && score > feature_thresh && distance_check) {
                        // Add pixel measurements to vector y
                        int n_measurements = yk.rows();
                        Eigen::MatrixXd pixel(2,1);
                        std::cout << "keypoints_found.size() " << keypoints_found.size() << std::endl;
                        pixel << keypoints_found[i].pt.x, keypoints_found[i].pt.y;
                        std::cout << "pixel : " << pixel << std::endl;
                        yk.conservativeResizeLike(Eigen::MatrixXd::Zero(n_measurements+2,1));
                        yk.tail(2) = pixel;
                        // Add new landmark description to now seen list
                        int n_states = mup.rows();
                        landmark_descriptors.push_back(descriptors_found.row(i));
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
                        std::vector<cv::Point2f> pixels_to_undistort; cv::Point2f pixel_to_undistort;
                        std::vector<cv::Point2f> ray;
                        pixel_to_undistort.x = pixel(0);
                        pixel_to_undistort.y = pixel(1);
                        pixels_to_undistort.push_back(pixel_to_undistort);
                        cv::undistortPoints(pixels_to_undistort, ray, camera_param.Kc, camera_param.distCoeffs);
                        Eigen::VectorXd optical_ray(3,1);
                        optical_ray << ray[0].x,ray[0].y,1;
                        //Scale it out
                        optical_ray = optical_ray_lengh*optical_ray;
                        //scale it by 2
                        Thetanc = mup.segment(9,3);
                        rpy2rot(Thetanc, Rnc);
                        Eigen::VectorXd point(3,1);
                        point = mup.segment(6,3) + Rnc*optical_ray;
                        mup.tail(3) << point;
                        // Add index j to landmark seen vector
                        slamparam.landmarks_seen.push_back((n_states-12)/3);
                        std::cout << "NEW LANDMARK : " << (n_states-12)/3 << std::endl;
                        // Add keypoint to keypoints list
                        cv::KeyPoint temp;
                        temp.pt.x = keypoints_found[i].pt.x;
                        temp.pt.y = keypoints_found[i].pt.y;
                        landmark_keypoints.push_back(temp);
                        // Add initial bad landmark score
                        bad_landmark.push_back(0);
                        cv::putText(imgout,"J="+std::to_string((mup.rows()-12)/3 - 1),cv::Point(keypoints_found[i].pt.x+10,keypoints_found[i].pt.y+10),cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 100, 255),2);
                    }

                }

                cv::putText(imgout,"frame no. : "+std::to_string(count),cv::Point(60,60),cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 0, 0),2);
                cv::putText(imgout,"Number of current landmarks "+std::to_string((mup.rows()-12)/3),cv::Point(60,100),cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 0, 0),2);
                cv::putText(imgout,"Number of seen landmarks "+std::to_string(slamparam.landmarks_seen.size()),cv::Point(60,140),cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 0, 0),2);
                for(int i = 0; i < slamparam.landmarks_seen.size(); i++) {
                    std::cout << " slamparam.landmarks_seen[i] : " << slamparam.landmarks_seen[i] << std::endl;
                }

                // Measurement update
                PointLogLikelihood point_ll;
                std::cout << " BEFORE MEASUREMENT UPDATE: " << std::endl;
                std::cout << "frame no.: "<< count << std::endl;
                std::cout << "mup.rows() : "<< mup.rows() << std::endl;
                std::cout << "Sp.rows() : "<< Sp.rows() << std::endl;
                std::cout << "Sp.cols() : "<< Sp.cols() << std::endl;
                std::cout << "yk : "<< yk << std::endl;
                measurementUpdateIEKF(mup, Sp, u, yk, point_ll, slamparam, muf, Sf);
                muEKF               = muf;
                SEKF                = Sf;
                std::cout << "muEKF.rows(): "<< muEKF.rows() << std::endl;
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
        std::cout << "muPlot" << muPlot << std::endl;


        if (interactive == 1) {
            updatePlotStates(imgout, muPlot, SPlot, param, handles,slamparam);
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

bool pixelDistance(std::vector<cv::KeyPoint> landmark_keypoints, cv::KeyPoint keypoint){
    //loop through the all the current keypoints initisised and check if its far away
    bool res = true;
    cv::Point2f b(keypoint.pt.x, keypoint.pt.y);
    // loop through the landmarks just initilised and all the current landmarks
    for(int i; i < landmark_keypoints.size(); i++){
        cv::Point2f a(landmark_keypoints[i].pt.x, landmark_keypoints[i].pt.y);
        cv::Point2f diff = a - b;
        float dist = cv::sqrt(diff.x*diff.x + diff.y*diff.y);
        if(dist < 50) {
            res = false;
        }
    }
    return res;
}

void removeBadLandmarks(Eigen::VectorXd & mup, Eigen::MatrixXd & Sp, std::vector<cv::KeyPoint> & landmark_keypoints, cv::Mat & landmark_descriptors, std::vector<int> & landmarks_seen, std::vector<int> & bad_landmark, int j) {
    int nx = 12; //camera states
    // remove landmark keypoints
    landmark_keypoints.erase(landmark_keypoints.begin()+nx+j);
    std::cout << "HERE" << std::endl;
    std::cout << "bad_landmark.size() " << bad_landmark.size() << std::endl;
    bad_landmark.erase(bad_landmark.begin()+j);
    // remove landmark keypoints
    for(int i = 0; i < 3; i++){
        //remove landmark states from mu
        removeRow(mup,nx+j);
        //remove the columns of the covariance matrix for landmark
        removeColumn(Sp,nx+j);
    }
    // perform qr decomposition
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
    cv::Mat good_landmark_descriptors;
    // loop through all descriptors and add to temp descriptors
    for(int i =0; i < landmark_descriptors.rows; i++){
        if(i != j) {
            good_landmark_descriptors.push_back(landmark_descriptors.row(i));
        }
    }
    // ovewrite the old descriptors
    landmark_descriptors = good_landmark_descriptors;

    //landmarks seen vector
    landmarks_seen.erase(std::remove(landmarks_seen.begin(), landmarks_seen.end(), j), landmarks_seen.end());

    // for each landmark larger then the one just deleted, reduce its index by 1
    for(int i = 0; i < landmarks_seen.size(); i++) {
        std::cout << "landmarks_seen before" << landmarks_seen[i] << std::endl;
        if( landmarks_seen[i] > j) {
             landmarks_seen[i] += -1; //reduce index
            std::cout << "landmarks_seen reduced by 1" << landmarks_seen[i] << std::endl;
        }
    }
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

