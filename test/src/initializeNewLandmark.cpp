// // #define CATCH_CONFIG_ENABLE_BENCHMARKING
// // #define CATCH_CONFIG_COLOUR_ANSI
// #include <catch2/catch.hpp>

// #include "../../src/settings.h"
// #include "../../src/cameraModel.hpp"
// #include "../../src/SLAM.h"
// #include "../../src/imagefeatures.h"


// SCENARIO("Initialize a new landmark test"){

//     WHEN("Calling initializeNewLandmark"){

//         // 1 tag measurement
//         Eigen::VectorXd y(8,1);
//         // 2 landmarks
//         Eigen::VectorXd mu(12+6);

//         mu <<       1, // x dot
//                     2, // y dot
//                     3, // z dot
//                     4, // Psi dot
//                     5, // Theta dot
//                     6, // Phi dot
//                     7, // x
//                     8, // y
//                     9, // z
//                     10, // Psi
//                     11, // Theta
//                     12,// Phi
//                     13,
//                     14,
//                     15,
//                     16,
//                     17,
//                     18;

//         Eigen::MatrixXd S(12+6,12+6);
//         S.setZero();
//         S.diagonal() <<         1, // x dot
//                                 2, // y dot
//                                 3, // z dot
//                                 4, // Psi dot
//                                 5, // Theta dot
//                                 6, // Phi dot
//                                 7, // x
//                                 8, // y
//                                 9, // z
//                                 10, // Psi
//                                 11, // Theta
//                                 12,// Phi
//                                 13,
//                                 14,
//                                 15,
//                                 16,
//                                 17,
//                                 18;
//         //Scenario 2
//         //Scenario 2
//         std::vector<Landmark> landmarks;
//         Landmark temp_landmark_1;
//         cv::KeyPoint keypoint;
//         temp_landmark_1.keypoint = keypoint;
//         cv::Mat descriptor = (cv::Mat_<double>(2,32) << 0, -1, 0, -1, 5, -1, 0, -1, 0, 0, -1, 0, -1, 5, -1, 0, -1, 0, 0, -1, 0, -1, 5, -1, 0, -1, 0, 0, 0, 0, 0, 0);
//         temp_landmark_1.descriptor = descriptor;
//         temp_landmark_1.isVisible = true;

//         landmarks.push_back(temp_landmark_1);
//         landmarks.push_back(temp_landmark_1);

//         cv::Mat img;
//         img = cv::imread("../data/1023.jpg");

//         SlamParameters p;
//                 // MAP TUNING
//         p.max_landmarks = 50;
//         p.max_features = 50000;
//         p.max_bad_frames = 10;
//         p.feature_thresh = 0.0001;
//         p.initial_pixel_distance_thresh = 150;
//         p.update_pixel_distance_thresh = 1;
//         p.initial_width_thresh = 250;
//         p.initial_height_thresh = 100;
//         // Initilizing landmarks
//         p.optical_ray_length = 8;
//         p.kappa = 0.5;

//         std::vector<sortedLandmark> sorted_landmarks;
//         for(int i = 0; i < landmarks.size(); i++) {
//             sortedLandmark lm;
//             lm.keypoint = temp_landmark_1.keypoint;
//             lm.descriptor = temp_landmark_1.descriptor;
//             sorted_landmarks.push_back(lm);
//         }

//         initializeNewLandmark(img, mu, S,mu, S, sorted_landmarks, landmarks, p, 0);

//         THEN("Dimensions of everything correct") {
//             CHECK(landmarks.size() == 1);
//             CHECK(mu.rows() == 21);
//             CHECK(mu.cols() == 1);
//             CHECK(S.rows() == 21);
//             CHECK(S.cols() == 21);
//         }
//     }
// }


