// #define CATCH_CONFIG_ENABLE_BENCHMARKING
// #define CATCH_CONFIG_COLOUR_ANSI
#include <catch2/catch.hpp>

#include "../../src/settings.h"
#include "../../src/cameraModel.hpp"
#include "../../src/SLAM.h"

SCENARIO("delete landmark test"){

    WHEN("Calling removeBadLandmark"){

        // 1 tag measurement
        Eigen::VectorXd y(8,1);
        // 2 landmarks
        Eigen::VectorXd mu(12+6);

        mu <<       1, // x dot
                    2, // y dot
                    3, // z dot
                    4, // Psi dot
                    5, // Theta dot
                    6, // Phi dot
                    7, // x
                    8, // y
                    9, // z
                    10, // Psi
                    11, // Theta
                    12,// Phi
                    13,
                    14,
                    15,
                    16,
                    17,
                    18;

        Eigen::MatrixXd S(12+6,12+6);
        S.setZero();
        S.diagonal() <<         1, // x dot
                                2, // y dot
                                3, // z dot
                                4, // Psi dot
                                5, // Theta dot
                                6, // Phi dot
                                7, // x
                                8, // y
                                9, // z
                                10, // Psi
                                11, // Theta
                                12,// Phi
                                13,
                                14,
                                15,
                                16,
                                17,
                                18;
        //Scenario 2

        std::vector<cv::KeyPoint> landmark_keypoints;
        cv::KeyPoint keypoint;
        landmark_keypoints.push_back(keypoint);
        landmark_keypoints.push_back(keypoint);

        std::vector<int> landmarks_seen;
        landmarks_seen.push_back(0);
        landmarks_seen.push_back(1);

        std::vector<int> bad_landmark;
        bad_landmark.push_back(1);
        bad_landmark.push_back(2);

        cv::Mat landmark_descriptors = (cv::Mat_<double>(2,32) << 0, -1, 0, -1, 5, -1, 0, -1, 0, 0, -1, 0, -1, 5, -1, 0, -1, 0, 0, -1, 0, -1, 5, -1, 0, -1, 0, 0, 0, 0, 0, 0,0, -1, 0, -1, 5, -1, 0, -1, 0, 0, -1, 0, -1, 5, -1, 0, -1, 0, 0, -1, 0, -1, 5, -1, 0, -1, 0, 0, 0, 0, 0, 0);

        std::cout << "mu before : " << mu << std::endl;
        std::cout << "S before : " << S << std::endl;

        removeBadLandmarks(mu, S, landmark_keypoints, landmark_descriptors,landmarks_seen,bad_landmark,0);

        std::cout << "mu after : " << mu << std::endl;
        std::cout << "S after : " << S << std::endl;

        for(int i = 0; i < landmarks_seen.size(); i++) {
            std::cout << "landmarks_seen: " << landmarks_seen[i] << std::endl;

        }

        THEN("Dimensions of everything correct") {
            CHECK(bad_landmark.size() == 1);
            CHECK(landmark_descriptors.rows == 1);
            CHECK(landmark_keypoints.size() == 1);
            CHECK(mu.rows() == 15);
            CHECK(mu.cols() == 1);
            CHECK(S.rows() == 15);
            CHECK(S.cols() == 15);
            CHECK(landmark_keypoints.size() == 1);
            CHECK(landmarks_seen.size() == 1);

        }
    }
}


