// // #define CATCH_CONFIG_ENABLE_BENCHMARKING
// // #define CATCH_CONFIG_COLOUR_ANSI
// #include <catch2/catch.hpp>

// #include "../../src/model.hpp"
// #include "../../src/settings.h"
// #include "../../src/cameraModel.hpp"
// #include "../../src/SLAM.h"

// #include <autodiff/reverse/var.hpp>
// #include <autodiff/reverse/var/eigen.hpp>


// SCENARIO("reverse mode autodiff test"){


//     // 1 landmark
//     Eigen::VectorXd x(12+6);
//     x <<        0, // x dot
//                 0, // y dot
//                 0, // z dot
//                 0, // Psi dot
//                 0, // Theta dot
//                 0, // Phi dot
//                 0, // x
//                 0, // y
//                 -1.8, // z
//                 -3.14159265359/2, // Phi
//                 3.14159265359, // Theta
//                 0,// Psi
//                 0,
//                 1,
//                 -1.8,
//                 0,
//                 0,
//                 0;

//     std::cout << "x : " << x << std::endl;

//     Eigen::Matrix<double, Eigen::Dynamic, 1> rPNn(3,1);
//     rPNn << 0,0,1;
//     Eigen::Matrix<double, Eigen::Dynamic, 1> eta(6,1);
//     eta << 0,0,0,0,0,0;
//     CameraParameters param;
//     std::filesystem::path calibrationFilePath = "data/camera.xml";
//     importCalibrationData(calibrationFilePath, param);
//     Eigen::Matrix<double, Eigen::Dynamic, 1> rQOi;

//     autodiff::VectorXvar x_var = x.cast<autodiff::var>();
//     Eigen::Matrix<autodiff::var,Eigen::Dynamic,1>  rPNn_var = rPNn.cast<autodiff::var>();
//     Eigen::Matrix<autodiff::var,Eigen::Dynamic,1>  eta_var = eta.cast<autodiff::var>();
//     Eigen::Matrix<autodiff::var,Eigen::Dynamic,1>  rQOi_var = rQOi.cast<autodiff::var>();
//     autodiff::var fvar = worldToPixel(rPNn_var,eta_var,param,rQOi_var);

//     std::cout << " here" << std::endl;
//     Eigen::Matrix<double, Eigen::Dynamic, 1> g;
//     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H = hessian(fvar, x_var, g);

//     std::cout << "fvar val " << val(fvar) << std::endl;
//     std::cout << "rQOi_var" << rQOi_var<< std::endl;
//     std::cout << "g " << g << std::endl;
//     std::cout << "H " << H << std::endl;


// }


