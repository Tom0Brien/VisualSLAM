// #define CATCH_CONFIG_ENABLE_BENCHMARKING
// #define CATCH_CONFIG_COLOUR_ANSI
#include <catch2/catch.hpp>

#include "../../src/settings.h"
#include "../../src/cameraModel.hpp"
#include "../../src/SLAM.h"

SCENARIO("pixel distance test"){

    double pixel_distance_thresh = 50;
    std::vector<cv::KeyPoint> landmark_keypoints;
    cv::KeyPoint keypoint1;
    cv::KeyPoint keypoint2;
    keypoint1.pt.x = 0;
    keypoint1.pt.y = 0;
    landmark_keypoints.push_back(keypoint1);
    keypoint2.pt.x = 100;
    keypoint2.pt.y = 0;
    bool withinRadius;
    WHEN("pixel distance threshold is 10"){
        withinRadius = pixelDistance(landmark_keypoints,keypoint2,pixel_distance_thresh);
        THEN("result is false") {
            REQUIRE(withinRadius == false);
        }
    }
    pixel_distance_thresh = 200;
    WHEN("pixel distance threshold is 200"){
        withinRadius = pixelDistance(landmark_keypoints,keypoint2,pixel_distance_thresh);
        THEN("result is true") {
            REQUIRE(withinRadius == true);
        }
    }
}


