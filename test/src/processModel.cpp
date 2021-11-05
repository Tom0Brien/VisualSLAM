// // #define CATCH_CONFIG_ENABLE_BENCHMARKING
// // #define CATCH_CONFIG_COLOUR_ANSI
// #include <catch2/catch.hpp>
// #include <stdio.h>      /* printf, scanf, puts, NULL */
// #include <stdlib.h>     /* srand, rand */
// #include <time.h>       /* time */

// #include "../../src/model.hpp"
// #include "../../src/settings.h"
// #include "../../src/cameraModel.hpp"
// #include "../../src/SLAM.h"

// void SlamProcessModel::operator()(const Eigen::VectorXd & x, const Eigen::VectorXd & u, const SlamParameters & param, Eigen::VectorXd & f)
// {
//     // TODO: mean function, x = [nu;eta;landmarks]
//     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> J(6,6);
//     J.setZero();

//     // Rnc
//     Eigen::Matrix<double, Eigen::Dynamic, 1> Thetanc(3,1);
//     Thetanc = x.block(9,0,3,1);
//     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Rnc(3,3);
//     rpy2rot(Thetanc,Rnc);

//     // Kinematic transform T(nu)
//     double phi   = Thetanc(0);
//     double theta = Thetanc(1);
//     double psi   = Thetanc(2);

//     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> T(3,3);
//     using std::tan;
//     using std::cos;
//     using std::sin;
//     T << 1, sin(phi)*tan(theta),cos(phi)*tan(theta),
//         0,cos(phi),-sin(phi),
//         0,sin(phi)/cos(theta),cos(phi)/cos(theta);

//     J.block(0,0,3,3) = Rnc;
//     J.block(3,3,3,3) = T;

//     f.resize(x.rows(),1);
//     f.setZero();
//     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> nu(6,1);
//     nu = x.block(0,0,6,1);
//     f.block(6,0,6,1) = J*nu;

// }

// SCENARIO("Check analytical process model jaboian is equal to autodiff"){
//     Eigen::VectorXd y;
//     srand (time(NULL));
//     int num_iterations = 100;

//     for (int i = 0; i < num_iterations; i++) {
//         Eigen::VectorXd x = Eigen::VectorXd::Random(12);
//         Eigen::VectorXd u;
//         SlamParameters slamparam;
//         Eigen::VectorXd g;
//         Eigen::MatrixXd H;
//         //Analytical
//         Eigen::VectorXd g_analytical;
//         Eigen::MatrixXd H_analytical;
//         CameraParameters param;
//         std::filesystem::path calibrationFilePath = "data/camera.xml";
//         importCalibrationData(calibrationFilePath, param);
//         slamparam.camera_param = param;
//         slamparam.measurement_noise = 2;
//         arucoLogLikelihoodAnalytical ll_analytical;
//         ArucoLogLikelihood ll;

//         // Call log likelihood
//         double cost_auto = ll(y, x, u, slamparam, g, H);
//         double cost_ana = ll_analytical(y, x, u, slamparam, g_analytical,H_analytical);
//         THEN("Jacobians the same") {


//         }
//     }
// }
