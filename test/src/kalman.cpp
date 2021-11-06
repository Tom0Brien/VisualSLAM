// #define CATCH_CONFIG_ENABLE_BENCHMARKING
// #define CATCH_CONFIG_COLOUR_ANSI
// #define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
// #include <catch2/catch.hpp>

// #include "../../src/model.hpp"
// #include "../../src/settings.h"
// #include "../../src/cameraModel.hpp"
// #include "../../src/SLAM.h"

// SCENARIO("Case: time update"){
//     WHEN("Calling time update") {
//         // 2 landmarks
//         Eigen::VectorXd mu(12+6);

//         mu <<       1,  // x dot
//                     2,  // y dot
//                     3,  // z dot
//                     4,  // Psi dot
//                     5,  // Theta dot
//                     6,  // Phi dot
//                     7,  // x
//                     8,  // y
//                     9,  // z
//                     10, // Psi
//                     11, // Theta
//                     12, // Phi
//                     13,
//                     14,
//                     15,
//                     16,
//                     17,
//                     18;

//         Eigen::MatrixXd S(12+6,12+6);
//         S.setZero();
//         S.diagonal() <<         1,  // x dot
//                                 2,  // y dot
//                                 3,  // z dot
//                                 4,  // Psi dot
//                                 5,  // Theta dot
//                                 6,  // Phi dot
//                                 7,  // x
//                                 8,  // y
//                                 9,  // z
//                                 10, // Psi
//                                 11, // Theta
//                                 12, // Phi
//                                 13,
//                                 14,
//                                 15,
//                                 16,
//                                 17,
//                                 18;

//         SlamProcessModel     pm;
//         double timestep = 0.01;

//         Eigen::MatrixXd Sp = S;
//         Eigen::VectorXd mup = mu;
//         SlamParameters p;
//         Eigen::VectorXd u;
//         timeUpdateContinuous(mu, S, u, pm, p, timestep, mup, Sp);

//         THEN("state dimensions matches expected values")
//             {
//                 REQUIRE(mup.rows() == mu.rows());
//                 REQUIRE(mup.cols() == mu.cols());
//                 REQUIRE(Sp.cols() == S.cols());
//                 REQUIRE(Sp.rows() == S.rows());
//             }
//     }
// }