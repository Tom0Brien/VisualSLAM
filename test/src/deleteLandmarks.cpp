// // #define CATCH_CONFIG_ENABLE_BENCHMARKING
// // #define CATCH_CONFIG_COLOUR_ANSI
// #include <catch2/catch.hpp>

// #include "../../src/model.hpp"
// #include "../../src/settings.h"
// #include "../../src/cameraModel.hpp"
// #include "../../src/SLAM.h"

// SCENARIO("measurement model test"){
//     WHEN("Calling analytical and autodiff measurment model"){
//         SlamLogLikelihood ll;

//         // 1 tag measurement
//         Eigen::VectorXd y(8,1);
//         // 1 landmark
//         Eigen::VectorXd x(12+6);

//         x <<        0, // x dot
//                     0, // y dot
//                     0, // z dot
//                     0, // Psi dot
//                     0, // Theta dot
//                     0, // Phi dot
//                     0, // x
//                     0, // y
//                     -1.8, // z
//                     -3.14159265359/2, // Psi
//                     3.14159265359, // Theta
//                     0,// Phi
//                     0,
//                     1,
//                     -1.8,
//                     0,
//                     0,
//                     0;

//         std::cout << "x : " << x << std::endl;

//         y <<    1463,
//                 481,
//                 1386,
//                 480,
//                 1389,
//                 402,
//                 1467,
//                 402;

//         // std::cout << "y : " << y << std::endl;

//         Eigen::VectorXd u;
//         SlamParameters slamparam;
//         Eigen::VectorXd g;
//         Eigen::MatrixXd H;

//         // 1 landmark "seen"
//         slamparam.landmarks_seen.push_back(0);

//         CameraParameters param;
//         std::filesystem::path calibrationFilePath = "data/camera.xml";
//         importCalibrationData(calibrationFilePath, param);
//         slamparam.camera_param = param;


//         // Call log likelihood
//         double cost_auto = ll(y, x, u, slamparam, g, H);


//         REQUIRE(g.rows() == x.rows());
//         REQUIRE(g.cols() == 1);


//         //Analytical

//         Eigen::VectorXd g_analytical;
//         Eigen::MatrixXd H_analytical;

//         slamLogLikelihoodAnalytical ll_analytical;

//         double cost_ana = ll_analytical(y, x, u, slamparam, g_analytical, H_analytical);
//         std::cout << "cost_auto : " << cost_auto << std::endl;
//         std::cout << "cost_ana : " << cost_ana << std::endl;

//         std::cout << "Analytical Jacobian :   " << std::endl << g_analytical << std::endl;
//         std::cout << "Jacobian.rows()     :   " << g_analytical.rows() << std::endl;
//         std::cout << "Jacobian.cols()     :   " << g_analytical.cols() << std::endl;

//         std::cout << "autodiff g.rows()   : " << g.rows() << std::endl;
//         std::cout << "autodiff g.cols()   :   " << g.cols() << std::endl;
//         std::cout << "autodiff g          :   " << std::endl <<  g << std::endl;


//         std::cout << "Analytical H          :   " << H_analytical << std::endl;
//         std::cout << "autodiff H          :   " << H << std::endl;

//         THEN("Jacobians the same") {
//             REQUIRE(g.rows() == x.rows());
//             REQUIRE(g.cols() == 1);

//         }
//     }
// }


