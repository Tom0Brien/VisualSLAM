// #define CATCH_CONFIG_ENABLE_BENCHMARKING
// #include <catch2/catch.hpp>
// #include <Eigen/Core>
// #include <filesystem>

// #include "opencv2/core.hpp"
// #include "opencv2/highgui.hpp"
// #include "opencv2/features2d.hpp"
// #include "opencv2/xfeatures2d.hpp"



// TEST_CASE("Matcher performance: 10 features")
// {


//     std::filesystem::path imagePathA = std::filesystem::path("test") / std::filesystem::path("data") / std::filesystem::path("GOPR9473.JPG");
//     std::filesystem::path imagePathB = std::filesystem::path("test") / std::filesystem::path("data") / std::filesystem::path("GOPR9474.JPG");

//     REQUIRE(std::filesystem::exists(imagePathA));
//     REQUIRE(std::filesystem::exists(imagePathB));


//     cv::Mat viewRawA                       = cv::imread(imagePathA.string(), cv::IMREAD_COLOR);
//     REQUIRE(!viewRawA.empty());

//     cv::Mat viewRawB                       = cv::imread(imagePathB.string(), cv::IMREAD_COLOR);
//     REQUIRE(!viewRawB.empty());


//     // Initialise ORB detector
//     // https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
//     int maxNumFeatures = 10;

//     cv::Ptr<cv::ORB> orb;
//     orb = cv::ORB::create(
//         maxNumFeatures,         // nfeatures
//         1.3f,                   // scaleFactor
//         10,                     // nlevels
//         31,                     // edgeThreshold
//         0,                      // firstLevel
//         2,                      // WTA_K
//         cv::ORB::HARRIS_SCORE,  // scoreType
//         31,                     // patchSize
//         20                      // fastThreshold
//         );

//     // Detect descriptors in frame A
//     std::vector<cv::KeyPoint> keypointsA;
//     cv::Mat descriptorsA;
//     orb->detect(viewRawA, keypointsA);
//     orb->compute(viewRawA, keypointsA, descriptorsA);

//     // Detect descriptors in frame B
//     std::vector<cv::KeyPoint> keypointsB;
//     cv::Mat descriptorsB;
//     orb->detect(viewRawB, keypointsB);
//     orb->compute(viewRawB, keypointsB, descriptorsB);

//     std::vector< cv::DMatch > matches;

//     cv::BFMatcher matcherBF = cv::BFMatcher(cv::NORM_HAMMING, false);
//     BENCHMARK("Brute force matcher")
//     {
//         matcherBF.match(descriptorsA, descriptorsB,  matches);
//     };

//     cv::FlannBasedMatcher matcherFlann = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
//     std::vector< std::vector<cv::DMatch> > matchesFlann;
//     BENCHMARK("Flann based matcher knn")
//     {
//         matcherFlann.knnMatch(descriptorsA, descriptorsB,  matchesFlann, 2);
//     };
// }
// TEST_CASE("Matcher performance: 100 features")
// {


//     std::filesystem::path imagePathA = std::filesystem::path("test") / std::filesystem::path("data") / std::filesystem::path("GOPR9473.JPG");
//     std::filesystem::path imagePathB = std::filesystem::path("test") / std::filesystem::path("data") / std::filesystem::path("GOPR9474.JPG");

//     REQUIRE(std::filesystem::exists(imagePathA));
//     REQUIRE(std::filesystem::exists(imagePathB));


//     cv::Mat viewRawA                       = cv::imread(imagePathA.string(), cv::IMREAD_COLOR);
//     REQUIRE(!viewRawA.empty());

//     cv::Mat viewRawB                       = cv::imread(imagePathB.string(), cv::IMREAD_COLOR);
//     REQUIRE(!viewRawB.empty());


//     // Initialise ORB detector
//     // https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
//     int maxNumFeatures = 100;

//     cv::Ptr<cv::ORB> orb;
//     orb = cv::ORB::create(
//         maxNumFeatures,         // nfeatures
//         1.3f,                   // scaleFactor
//         10,                     // nlevels
//         31,                     // edgeThreshold
//         0,                      // firstLevel
//         2,                      // WTA_K
//         cv::ORB::HARRIS_SCORE,  // scoreType
//         31,                     // patchSize
//         20                      // fastThreshold
//         );

//     // Detect descriptors in frame A
//     std::vector<cv::KeyPoint> keypointsA;
//     cv::Mat descriptorsA;
//     orb->detect(viewRawA, keypointsA);
//     orb->compute(viewRawA, keypointsA, descriptorsA);

//     // Detect descriptors in frame B
//     std::vector<cv::KeyPoint> keypointsB;
//     cv::Mat descriptorsB;
//     orb->detect(viewRawB, keypointsB);
//     orb->compute(viewRawB, keypointsB, descriptorsB);

//     std::vector< cv::DMatch > matches;

//     cv::BFMatcher matcherBF = cv::BFMatcher(cv::NORM_HAMMING, false);
//     BENCHMARK("Brute force matcher")
//     {
//         matcherBF.match(descriptorsA, descriptorsB,  matches);
//     };

//     cv::FlannBasedMatcher matcherFlann = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
//     std::vector< std::vector<cv::DMatch> > matchesFlann;
//     BENCHMARK("Flann based matcher knn")
//     {
//         matcherFlann.knnMatch(descriptorsA, descriptorsB,  matchesFlann, 2);
//     };
// }
// TEST_CASE("Matcher performance: 1000 features")
// {


//     std::filesystem::path imagePathA = std::filesystem::path("test") / std::filesystem::path("data") / std::filesystem::path("GOPR9473.JPG");
//     std::filesystem::path imagePathB = std::filesystem::path("test") / std::filesystem::path("data") / std::filesystem::path("GOPR9474.JPG");

//     REQUIRE(std::filesystem::exists(imagePathA));
//     REQUIRE(std::filesystem::exists(imagePathB));


//     cv::Mat viewRawA                       = cv::imread(imagePathA.string(), cv::IMREAD_COLOR);
//     REQUIRE(!viewRawA.empty());

//     cv::Mat viewRawB                       = cv::imread(imagePathB.string(), cv::IMREAD_COLOR);
//     REQUIRE(!viewRawB.empty());


//     // Initialise ORB detector
//     // https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
//     int maxNumFeatures = 1000;

//     cv::Ptr<cv::ORB> orb;
//     orb = cv::ORB::create(
//         maxNumFeatures,         // nfeatures
//         1.3f,                   // scaleFactor
//         10,                     // nlevels
//         31,                     // edgeThreshold
//         0,                      // firstLevel
//         2,                      // WTA_K
//         cv::ORB::HARRIS_SCORE,  // scoreType
//         31,                     // patchSize
//         20                      // fastThreshold
//         );

//     // Detect descriptors in frame A
//     std::vector<cv::KeyPoint> keypointsA;
//     cv::Mat descriptorsA;
//     orb->detect(viewRawA, keypointsA);
//     orb->compute(viewRawA, keypointsA, descriptorsA);

//     // Detect descriptors in frame B
//     std::vector<cv::KeyPoint> keypointsB;
//     cv::Mat descriptorsB;
//     orb->detect(viewRawB, keypointsB);
//     orb->compute(viewRawB, keypointsB, descriptorsB);

//     std::vector< cv::DMatch > matches;

//     cv::BFMatcher matcherBF = cv::BFMatcher(cv::NORM_HAMMING, false);
//     BENCHMARK("Brute force matcher")
//     {
//         matcherBF.match(descriptorsA, descriptorsB,  matches);
//     };

//     cv::FlannBasedMatcher matcherFlann = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
//     std::vector< std::vector<cv::DMatch> > matchesFlann;
//     BENCHMARK("Flann based matcher knn")
//     {
//         matcherFlann.knnMatch(descriptorsA, descriptorsB,  matchesFlann, 2);
//     };
// }
// TEST_CASE("Matcher performance: 10000 features")
// {


//     std::filesystem::path imagePathA = std::filesystem::path("test") / std::filesystem::path("data") / std::filesystem::path("GOPR9473.JPG");
//     std::filesystem::path imagePathB = std::filesystem::path("test") / std::filesystem::path("data") / std::filesystem::path("GOPR9474.JPG");

//     REQUIRE(std::filesystem::exists(imagePathA));
//     REQUIRE(std::filesystem::exists(imagePathB));


//     cv::Mat viewRawA                       = cv::imread(imagePathA.string(), cv::IMREAD_COLOR);
//     REQUIRE(!viewRawA.empty());

//     cv::Mat viewRawB                       = cv::imread(imagePathB.string(), cv::IMREAD_COLOR);
//     REQUIRE(!viewRawB.empty());


//     // Initialise ORB detector
//     // https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
//     int maxNumFeatures = 10000;

//     cv::Ptr<cv::ORB> orb;
//     orb = cv::ORB::create(
//         maxNumFeatures,         // nfeatures
//         1.3f,                   // scaleFactor
//         10,                     // nlevels
//         31,                     // edgeThreshold
//         0,                      // firstLevel
//         2,                      // WTA_K
//         cv::ORB::HARRIS_SCORE,  // scoreType
//         31,                     // patchSize
//         20                      // fastThreshold
//         );

//     // Detect descriptors in frame A
//     std::vector<cv::KeyPoint> keypointsA;
//     cv::Mat descriptorsA;
//     orb->detect(viewRawA, keypointsA);
//     orb->compute(viewRawA, keypointsA, descriptorsA);

//     // Detect descriptors in frame B
//     std::vector<cv::KeyPoint> keypointsB;
//     cv::Mat descriptorsB;
//     orb->detect(viewRawB, keypointsB);
//     orb->compute(viewRawB, keypointsB, descriptorsB);

//     std::vector< cv::DMatch > matches;

//     cv::BFMatcher matcherBF = cv::BFMatcher(cv::NORM_HAMMING, false);
//     BENCHMARK("Brute force matcher")
//     {
//         matcherBF.match(descriptorsA, descriptorsB,  matches);
//     };

//     cv::FlannBasedMatcher matcherFlann = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
//     std::vector< std::vector<cv::DMatch> > matchesFlann;
//     BENCHMARK("Flann based matcher knn")
//     {
//         matcherFlann.knnMatch(descriptorsA, descriptorsB,  matchesFlann, 2);
//     };
// }
