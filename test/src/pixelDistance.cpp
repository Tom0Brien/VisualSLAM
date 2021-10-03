// #define CATCH_CONFIG_ENABLE_BENCHMARKING
// #define CATCH_CONFIG_COLOUR_ANSI
#include <catch2/catch.hpp>

#include "../../src/settings.h"
#include "../../src/cameraModel.hpp"
#include "../../src/SLAM.h"

SCENARIO("pixel distance test"){

    // Add some landmarks wih keypoints to landmark vector
    std::vector<Landmark> landmarks;
    Landmark temp_landmark;
    cv::KeyPoint keypoint2;
    temp_landmark.keypoint.pt.x = 0;
    temp_landmark.keypoint.pt.y = 0;

    landmarks.push_back(temp_landmark);

    temp_landmark.keypoint.pt.x = 100;
    temp_landmark.keypoint.pt.y = 0;

    landmarks.push_back(temp_landmark);

    // generate a keypoint to test
    cv::KeyPoint keypoint;
    keypoint.pt.x = 0;
    keypoint.pt.y = 0;



    bool withinRadius;

    double pixel_distance_thresh = 10;
    WHEN("pixel is close and pixel distance threshold is 10"){
        withinRadius = pixelDistance(landmarks,keypoint,pixel_distance_thresh);
        THEN("result is false") {
            REQUIRE(withinRadius == true);
        }
    }
    pixel_distance_thresh = 200;
    WHEN("pixel is close and pixel distance threshold is 200"){
        withinRadius = pixelDistance(landmarks,keypoint,pixel_distance_thresh);
        THEN("result is true") {
            REQUIRE(withinRadius == true);
        }
    }

    keypoint.pt.x = 500;
    keypoint.pt.y = 500;

    WHEN("pixel is far away and pixel distance threshold is 200"){
        withinRadius = pixelDistance(landmarks,keypoint,pixel_distance_thresh);
        THEN("result is true") {
            REQUIRE(withinRadius == false);
        }
    }
}


