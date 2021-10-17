// // #define CATCH_CONFIG_ENABLE_BENCHMARKING
// // #define CATCH_CONFIG_COLOUR_ANSI
// #include <catch2/catch.hpp>

// #include "../../src/model.hpp"
// #include "../../src/settings.h"
// #include "../../src/cameraModel.hpp"
// #include "../../src/SLAM.h"

// SCENARIO("measurement model test"){
//     WHEN("Calling analytical and autodiff measurment model"){
//         SlamProcessModel pm;

//         // 1 tag measurement
//         Eigen::VectorXd y(8,1);
//         // 1 landmark
//         Eigen::VectorXd mu(12+6);
//         Eigen::MatrixXd S(12+6,12+6);

//         mu <<       0, // x dot
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

//         y <<    1463,
//                 481,
//                 1386,
//                 480,
//                 1389,
//                 402,
//                 1467,
//                 402;


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

//         //Analytical

//         Eigen::VectorXd g_analytical;
//         Eigen::MatrixXd H_analytical;

//         arucoLogLikelihoodAnalytical ll_analytical;

//         double cost_ana = ll_analytical(y, mu, u, slamparam, g_analytical, H_analytical);

//         // Call log likelihood
//         ArucoLogLikelihood ll;
//         double cost_auto = ll(y, mu, u, slamparam, g, H);


//         REQUIRE(g.rows() == mu.rows());
//         REQUIRE(g.cols() == 1);


//         std::cout << "cost_auto : " << cost_auto << std::endl;
//         std::cout << "cost_ana : " << cost_ana << std::endl;

//         std::cout << "Analytical Jacobian :   " << std::endl << g_analytical << std::endl;
//         std::cout << "Jacobian.rows()     :   " << g_analytical.rows() << std::endl;
//         std::cout << "Jacobian.cols()     :   " << g_analytical.cols() << std::endl;

//         std::cout << "autodiff g.rows()   : " << g.rows() << std::endl;
//         std::cout << "autodiff g.cols()   :   " << g.cols() << std::endl;
//         std::cout << "autodiff g          :   " << std::endl <<  g << std::endl;

//         THEN("Jacobians the same") {
//             REQUIRE(g.rows() == mu.rows());
//             REQUIRE(g.cols() == 1);
//             CHECK(g(0)  == Approx(g_analytical(0)));
//             CHECK(g(1)  == Approx(g_analytical(1)));
//             CHECK(g(2)  == Approx(g_analytical(2)));
//             CHECK(g(3)  == Approx(g_analytical(3)));
//             CHECK(g(4)  == Approx(g_analytical(4)));
//             CHECK(g(5)  == Approx(g_analytical(5)));
//             CHECK(g(6)  == Approx(g_analytical(6)));
//             CHECK(g(7)  == Approx(g_analytical(7)));
//             CHECK(g(8)  == Approx(g_analytical(8)));
//             CHECK(g(9)  == Approx(g_analytical(9)));
//             CHECK(g(10) == Approx(g_analytical(10)));
//             CHECK(g(11) == Approx(g_analytical(11)));
//             CHECK(g(12) == Approx(g_analytical(12)));
//             CHECK(g(13) == Approx(g_analytical(13)));
//             CHECK(g(14) == Approx(g_analytical(14)));
//             CHECK(g(15) == Approx(g_analytical(15)));
//             CHECK(g(16) == Approx(g_analytical(16)));
//             CHECK(g(17) == Approx(g_analytical(17)));
//         }
//     }
// }


