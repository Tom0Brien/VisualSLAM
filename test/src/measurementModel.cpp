// #define CATCH_CONFIG_ENABLE_BENCHMARKING
// #define CATCH_CONFIG_COLOUR_ANSI
#include <catch2/catch.hpp>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "../../src/model.hpp"
#include "../../src/settings.h"
#include "../../src/cameraModel.hpp"
#include "../../src/SLAM.h"

SCENARIO("Case: model with 0 measurements"){
    WHEN("Calling analytical and autodiff measurment") {
        Eigen::VectorXd y;
        srand (time(NULL));
        int num_iterations = 100;

        for (int i = 0; i < num_iterations; i++) {
            Eigen::VectorXd x = Eigen::VectorXd::Random(12);
            Eigen::VectorXd u;
            SlamParameters slamparam;
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            //Analytical
            Eigen::VectorXd g_analytical;
            Eigen::MatrixXd H_analytical;
            CameraParameters param;
            std::filesystem::path calibrationFilePath = "data/camera.xml";
            importCalibrationData(calibrationFilePath, param);
            slamparam.camera_param = param;
            slamparam.measurement_noise = 2;
            arucoLogLikelihoodAnalytical ll_analytical;
            ArucoLogLikelihood ll;

            // Call log likelihood
            double cost_auto = ll(y, x, u, slamparam, g, H);
            double cost_ana = ll_analytical(y, x, u, slamparam, g_analytical,H_analytical);
            THEN("Jacobians are equivalent") {
                REQUIRE(g.rows() == x.rows());
                REQUIRE(g.cols() == 1);
                REQUIRE(g.rows() == g_analytical.rows());
                REQUIRE(g.cols() == g_analytical.cols());
                REQUIRE(g.size() == g_analytical.size());
                CHECK(g(0)  == Approx(g_analytical(0)));
                CHECK(g(1)  == Approx(g_analytical(1)));
                CHECK(g(2)  == Approx(g_analytical(2)));
                CHECK(g(3)  == Approx(g_analytical(3)));
                CHECK(g(4)  == Approx(g_analytical(4)));
                CHECK(g(5)  == Approx(g_analytical(5)));
                CHECK(g(6)  == Approx(g_analytical(6)));
                CHECK(g(7)  == Approx(g_analytical(7)));
                CHECK(g(8)  == Approx(g_analytical(8)));
                CHECK(g(9)  == Approx(g_analytical(9)));
                CHECK(g(10) == Approx(g_analytical(10)));
                CHECK(g(11) == Approx(g_analytical(11)));
            }
        }
    }
}

SCENARIO("Case: 2 measurements and fixed state"){
    WHEN("Calling analytical and autodiff measurment") {
        Eigen::VectorXd y(16,1);
        srand (time(NULL));
        int num_iterations = 100;
        int count = 0;

        for (int i = 0; i < num_iterations; i++) {
            Eigen::VectorXd x(24,1);
            INFO("The count is " << count);
            count++;
            x <<    0, // x dot
                    0, // y dot
                    0, // z dot
                    0, // Psi dot
                    0, // Theta dot
                    0, // Phi dot
                    0, // x
                    0, // y
                    -1.8, // z
                    -3.14159265359/2, // Psi
                    3.14159265359, // Theta
                    0,// Phi
                    0.005,
                    1,
                    -1.8,
                    0,
                    0,
                    0,
                    0.005,
                    1,
                    -1.8,
                    0,
                    0,
                    0;

            y <<    1463,
                    481,
                    1386,
                    480,
                    1389,
                    402,
                    1467,
                    402,
                    1463,
                    481,
                    1386,
                    480,
                    1389,
                    402,
                    1467,
                    402;
            Eigen::VectorXd u;
            SlamParameters slamparam;
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            slamparam.landmarks_seen.push_back(0);
            slamparam.landmarks_seen.push_back(1);
            CameraParameters param;
            std::filesystem::path calibrationFilePath = "data/camera.xml";
            importCalibrationData(calibrationFilePath, param);
            slamparam.camera_param = param;
            slamparam.measurement_noise = 2;
            //Analytical
            Eigen::VectorXd g_analytical;
            Eigen::MatrixXd H_analytical;

            // Call log likelihood
            ArucoLogLikelihood ll;
            arucoLogLikelihoodAnalytical ll_analytical;
            double cost_auto = ll(y, x, u, slamparam, g, H);
            double cost_ana = ll_analytical(y, x, u, slamparam, g_analytical,H_analytical);

            THEN("Jacobians are equivalent") {
                REQUIRE(g.rows() == x.rows());
                REQUIRE(g.cols() == 1);
                REQUIRE(g.rows() == g_analytical.rows());
                REQUIRE(g.cols() == g_analytical.cols());
                REQUIRE(g.size() == g_analytical.size());
                CHECK(g(0)  == Approx(g_analytical(0)));
                CHECK(g(1)  == Approx(g_analytical(1)));
                CHECK(g(2)  == Approx(g_analytical(2)));
                CHECK(g(3)  == Approx(g_analytical(3)));
                CHECK(g(4)  == Approx(g_analytical(4)));
                CHECK(g(5)  == Approx(g_analytical(5)));
                CHECK(g(6)  == Approx(g_analytical(6)));
                CHECK(g(7)  == Approx(g_analytical(7)));
                CHECK(g(8)  == Approx(g_analytical(8)));
                CHECK(g(9)  == Approx(g_analytical(9)));
                CHECK(g(10) == Approx(g_analytical(10)));
                CHECK(g(11) == Approx(g_analytical(11)));
                CHECK(g(12) == Approx(g_analytical(12)));
                CHECK(g(13) == Approx(g_analytical(13)));
                CHECK(g(14) == Approx(g_analytical(14)));
                CHECK(g(15) == Approx(g_analytical(15)));
                CHECK(g(16) == Approx(g_analytical(16)));
                CHECK(g(17) == Approx(g_analytical(17)));
                CHECK(g(18) == Approx(g_analytical(18)));
                CHECK(g(19) == Approx(g_analytical(19)));
                CHECK(g(20) == Approx(g_analytical(20)));
                CHECK(g(21) == Approx(g_analytical(21)));
                CHECK(g(22) == Approx(g_analytical(22)));
                CHECK(g(23) == Approx(g_analytical(23)));
            }
        }
    }
}

SCENARIO("Case: measurment model with 2 measurements and random state"){
    WHEN("Calling analytical and autodiff measurment") {
        Eigen::VectorXd y(16,1);
        srand (time(NULL));
        int num_iterations = 100;

        for (int i = 0; i < num_iterations; i++) {
            Eigen::VectorXd x = Eigen::VectorXd::Random(24);

            y <<    1463,
                    481,
                    1386,
                    480,
                    1389,
                    402,
                    1467,
                    402,
                    1463,
                    481,
                    1386,
                    480,
                    1389,
                    402,
                    1467,
                    402;
            Eigen::VectorXd u;
            SlamParameters slamparam;
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            slamparam.landmarks_seen.push_back(0);
            slamparam.landmarks_seen.push_back(1);
            CameraParameters param;
            std::filesystem::path calibrationFilePath = "data/camera.xml";
            importCalibrationData(calibrationFilePath, param);
            slamparam.camera_param = param;
            slamparam.measurement_noise = 2;
            //Analytical
            Eigen::VectorXd g_analytical;
            Eigen::MatrixXd H_analytical;

            // Call log likelihood
            ArucoLogLikelihood ll;
            arucoLogLikelihoodAnalytical ll_analytical;
            double cost_auto = ll(y, x, u, slamparam, g, H);
            double cost_ana = ll_analytical(y, x, u, slamparam, g_analytical,H_analytical);

            THEN("Jacobians are equivalent") {
                REQUIRE(g.rows() == x.rows());
                REQUIRE(g.cols() == 1);
                REQUIRE(g.rows() == g_analytical.rows());
                REQUIRE(g.cols() == g_analytical.cols());
                REQUIRE(g.size() == g_analytical.size());
                CHECK(g(0)  == Approx(g_analytical(0)));
                CHECK(g(1)  == Approx(g_analytical(1)));
                CHECK(g(2)  == Approx(g_analytical(2)));
                CHECK(g(3)  == Approx(g_analytical(3)));
                CHECK(g(4)  == Approx(g_analytical(4)));
                CHECK(g(5)  == Approx(g_analytical(5)));
                CHECK(g(6)  == Approx(g_analytical(6)));
                CHECK(g(7)  == Approx(g_analytical(7)));
                CHECK(g(8)  == Approx(g_analytical(8)));
                CHECK(g(9)  == Approx(g_analytical(9)));
                CHECK(g(10) == Approx(g_analytical(10)));
                CHECK(g(11) == Approx(g_analytical(11)));
                CHECK(g(12) == Approx(g_analytical(12)));
                CHECK(g(13) == Approx(g_analytical(13)));
                CHECK(g(14) == Approx(g_analytical(14)));
                CHECK(g(15) == Approx(g_analytical(15)));
                CHECK(g(16) == Approx(g_analytical(16)));
                CHECK(g(17) == Approx(g_analytical(17)));
                CHECK(g(18) == Approx(g_analytical(18)));
                CHECK(g(19) == Approx(g_analytical(19)));
                CHECK(g(20) == Approx(g_analytical(20)));
                CHECK(g(21) == Approx(g_analytical(21)));
                CHECK(g(22) == Approx(g_analytical(22)));
                CHECK(g(23) == Approx(g_analytical(23)));
            }
        }
    }
}

SCENARIO("Case: 2 measurements and states all zero"){
    WHEN("Calling analytical and autodiff measurment") {
        // 1 tag measurement
        Eigen::VectorXd y(16,1);
        srand (time(NULL));
        int num_iterations = 10;

        for (int i = 0; i < num_iterations; i++) {
            Eigen::VectorXd x = Eigen::VectorXd::Zero(24);
            y <<    1463,
                    481,
                    1386,
                    480,
                    1389,
                    402,
                    1467,
                    402,
                    1463,
                    481,
                    1386,
                    480,
                    1389,
                    402,
                    1467,
                    402;
            Eigen::VectorXd u;
            SlamParameters slamparam;
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            // Call log likelihood
            ArucoLogLikelihood ll;
            //Analytical
            Eigen::VectorXd g_analytical;
            Eigen::MatrixXd H_analytical;
            arucoLogLikelihoodAnalytical ll_analytical;
            // 2 landmark "seen"
            slamparam.landmarks_seen.push_back(0);
            slamparam.landmarks_seen.push_back(1);
            CameraParameters param;
            std::filesystem::path calibrationFilePath = "data/camera.xml";
            importCalibrationData(calibrationFilePath, param);
            slamparam.camera_param = param;
            slamparam.measurement_noise = 2;

            double cost_auto = ll(y, x, u, slamparam, g, H);
            double cost_ana = ll_analytical(y, x, u, slamparam, g_analytical,H_analytical);

            THEN("Jacobians are equivalent") {
                REQUIRE(g.rows() == x.rows());
                REQUIRE(g.cols() == 1);
                REQUIRE(g.rows() == g_analytical.rows());
                REQUIRE(g.cols() == g_analytical.cols());
                REQUIRE(g.size() == g_analytical.size());
                CHECK(g(0)  == Approx(g_analytical(0)));
                CHECK(g(1)  == Approx(g_analytical(1)));
                CHECK(g(2)  == Approx(g_analytical(2)));
                CHECK(g(3)  == Approx(g_analytical(3)));
                CHECK(g(4)  == Approx(g_analytical(4)));
                CHECK(g(5)  == Approx(g_analytical(5)));
                CHECK(g(6)  == Approx(g_analytical(6)));
                CHECK(g(7)  == Approx(g_analytical(7)));
                CHECK(g(8)  == Approx(g_analytical(8)));
                CHECK(g(9)  == Approx(g_analytical(9)));
                CHECK(g(10) == Approx(g_analytical(10)));
                CHECK(g(11) == Approx(g_analytical(11)));
                CHECK(g(12) == Approx(g_analytical(12)));
                CHECK(g(13) == Approx(g_analytical(13)));
                CHECK(g(14) == Approx(g_analytical(14)));
                CHECK(g(15) == Approx(g_analytical(15)));
                CHECK(g(16) == Approx(g_analytical(16)));
                CHECK(g(17) == Approx(g_analytical(17)));
                CHECK(g(18) == Approx(g_analytical(18)));
                CHECK(g(19) == Approx(g_analytical(19)));
                CHECK(g(20) == Approx(g_analytical(20)));
                CHECK(g(21) == Approx(g_analytical(21)));
                CHECK(g(22) == Approx(g_analytical(22)));
                CHECK(g(23) == Approx(g_analytical(23)));
            }
        }
    }
}
SCENARIO("Case: zero states and random pixel measurements"){
    WHEN("Calling analytical and autodiff measurment") {
        srand (time(NULL));
        int num_iterations = 10;
        for (int i = 0; i < num_iterations; i++) {
            Eigen::VectorXd x = Eigen::VectorXd::Zero(24);
            Eigen::VectorXd y = Eigen::VectorXd::Random(16,1);

            Eigen::VectorXd u;
            SlamParameters slamparam;
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            CameraParameters param;
            std::filesystem::path calibrationFilePath = "data/camera.xml";
            importCalibrationData(calibrationFilePath, param);
            slamparam.camera_param = param;
            slamparam.measurement_noise = 2;
            // 2 landmark "seen"
            slamparam.landmarks_seen.push_back(0);
            slamparam.landmarks_seen.push_back(1);
            //Analytical
            Eigen::VectorXd g_analytical;
            Eigen::MatrixXd H_analytical;

            // Call log likelihood
            ArucoLogLikelihood ll;
            arucoLogLikelihoodAnalytical ll_analytical;
            double cost_auto = ll(y, x, u, slamparam, g, H);
            double cost_ana = ll_analytical(y, x, u, slamparam, g_analytical,H_analytical);

            THEN("Jacobians are equivalent") {
                REQUIRE(g.rows() == x.rows());
                REQUIRE(g.cols() == 1);
                REQUIRE(g.rows() == g_analytical.rows());
                REQUIRE(g.cols() == g_analytical.cols());
                REQUIRE(g.size() == g_analytical.size());
                CHECK(g(0)  == Approx(g_analytical(0)));
                CHECK(g(1)  == Approx(g_analytical(1)));
                CHECK(g(2)  == Approx(g_analytical(2)));
                CHECK(g(3)  == Approx(g_analytical(3)));
                CHECK(g(4)  == Approx(g_analytical(4)));
                CHECK(g(5)  == Approx(g_analytical(5)));
                CHECK(g(6)  == Approx(g_analytical(6)));
                CHECK(g(7)  == Approx(g_analytical(7)));
                CHECK(g(8)  == Approx(g_analytical(8)));
                CHECK(g(9)  == Approx(g_analytical(9)));
                CHECK(g(10) == Approx(g_analytical(10)));
                CHECK(g(11) == Approx(g_analytical(11)));
                CHECK(g(12) == Approx(g_analytical(12)));
                CHECK(g(13) == Approx(g_analytical(13)));
                CHECK(g(14) == Approx(g_analytical(14)));
                CHECK(g(15) == Approx(g_analytical(15)));
                CHECK(g(16) == Approx(g_analytical(16)));
                CHECK(g(17) == Approx(g_analytical(17)));
                CHECK(g(18) == Approx(g_analytical(18)));
                CHECK(g(19) == Approx(g_analytical(19)));
                CHECK(g(20) == Approx(g_analytical(20)));
                CHECK(g(21) == Approx(g_analytical(21)));
                CHECK(g(22) == Approx(g_analytical(22)));
                CHECK(g(23) == Approx(g_analytical(23)));
            }
        }
    }
}

SCENARIO("Case: random states and random pixel measurements"){
    WHEN("Calling analytical and autodiff measurment") {
        srand (time(NULL));
        int num_iterations = 10;
        for (int i = 0; i < num_iterations; i++) {
            Eigen::VectorXd x = Eigen::VectorXd::Random(24);
            Eigen::VectorXd y = Eigen::VectorXd::Random(16,1);

            Eigen::VectorXd u;
            SlamParameters slamparam;
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            CameraParameters param;
            std::filesystem::path calibrationFilePath = "data/camera.xml";
            importCalibrationData(calibrationFilePath, param);
            slamparam.camera_param = param;
            slamparam.measurement_noise = 2;
            // 2 landmark "seen"
            slamparam.landmarks_seen.push_back(0);
            slamparam.landmarks_seen.push_back(1);
            //Analytical
            Eigen::VectorXd g_analytical;
            Eigen::MatrixXd H_analytical;

            // Call log likelihood
            ArucoLogLikelihood ll;
            arucoLogLikelihoodAnalytical ll_analytical;
            double cost_auto = ll(y, x, u, slamparam, g, H);
            double cost_ana = ll_analytical(y, x, u, slamparam, g_analytical,H_analytical);

            THEN("Jacobians are equivalent") {
                REQUIRE(g.rows() == x.rows());
                REQUIRE(g.cols() == 1);
                REQUIRE(g.rows() == g_analytical.rows());
                REQUIRE(g.cols() == g_analytical.cols());
                REQUIRE(g.size() == g_analytical.size());
                CHECK(g(0)  == Approx(g_analytical(0)));
                CHECK(g(1)  == Approx(g_analytical(1)));
                CHECK(g(2)  == Approx(g_analytical(2)));
                CHECK(g(3)  == Approx(g_analytical(3)));
                CHECK(g(4)  == Approx(g_analytical(4)));
                CHECK(g(5)  == Approx(g_analytical(5)));
                CHECK(g(6)  == Approx(g_analytical(6)));
                CHECK(g(7)  == Approx(g_analytical(7)));
                CHECK(g(8)  == Approx(g_analytical(8)));
                CHECK(g(9)  == Approx(g_analytical(9)));
                CHECK(g(10) == Approx(g_analytical(10)));
                CHECK(g(11) == Approx(g_analytical(11)));
                CHECK(g(12) == Approx(g_analytical(12)));
                CHECK(g(13) == Approx(g_analytical(13)));
                CHECK(g(14) == Approx(g_analytical(14)));
                CHECK(g(15) == Approx(g_analytical(15)));
                CHECK(g(16) == Approx(g_analytical(16)));
                CHECK(g(17) == Approx(g_analytical(17)));
                CHECK(g(18) == Approx(g_analytical(18)));
                CHECK(g(19) == Approx(g_analytical(19)));
                CHECK(g(20) == Approx(g_analytical(20)));
                CHECK(g(21) == Approx(g_analytical(21)));
                CHECK(g(22) == Approx(g_analytical(22)));
                CHECK(g(23) == Approx(g_analytical(23)));
            }
        }
    }
}

SCENARIO("Case: points random states and random pixel measurements"){
    WHEN("Calling analytical and autodiff measurment") {
        srand (time(NULL));
        int num_iterations = 10;
        for (int i = 0; i < num_iterations; i++) {
            Eigen::VectorXd x = Eigen::VectorXd::Random(18);
            Eigen::VectorXd y = Eigen::VectorXd::Random(4,1);

            Eigen::VectorXd u;
            SlamParameters slamparam;
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            CameraParameters param;
            std::filesystem::path calibrationFilePath = "data/camera.xml";
            importCalibrationData(calibrationFilePath, param);
            slamparam.camera_param = param;
            slamparam.measurement_noise = 2;
            // 2 landmark "seen"
            slamparam.landmarks_seen.push_back(0);
            slamparam.landmarks_seen.push_back(1);
            //Analytical
            Eigen::VectorXd g_analytical;
            Eigen::MatrixXd H_analytical;

            // Call log likelihood
            PointLogLikelihood ll;
            pointLogLikelihoodAnalytical ll_analytical;
            double cost_auto = ll(y, x, u, slamparam, g, H);
            double cost_ana = ll_analytical(y, x, u, slamparam, g_analytical,H_analytical);

            THEN("Jacobians are equivalent") {
                REQUIRE(g.rows() == x.rows());
                REQUIRE(g.cols() == 1);
                REQUIRE(g.rows() == g_analytical.rows());
                REQUIRE(g.cols() == g_analytical.cols());
                REQUIRE(g.size() == g_analytical.size());
                CHECK(g(0)  == Approx(g_analytical(0)));
                CHECK(g(1)  == Approx(g_analytical(1)));
                CHECK(g(2)  == Approx(g_analytical(2)));
                CHECK(g(3)  == Approx(g_analytical(3)));
                CHECK(g(4)  == Approx(g_analytical(4)));
                CHECK(g(5)  == Approx(g_analytical(5)));
                CHECK(g(6)  == Approx(g_analytical(6)));
                CHECK(g(7)  == Approx(g_analytical(7)));
                CHECK(g(8)  == Approx(g_analytical(8)));
                CHECK(g(9)  == Approx(g_analytical(9)));
                CHECK(g(10) == Approx(g_analytical(10)));
                CHECK(g(11) == Approx(g_analytical(11)));
                CHECK(g(12) == Approx(g_analytical(12)));
                CHECK(g(13) == Approx(g_analytical(13)));
                CHECK(g(14) == Approx(g_analytical(14)));
                CHECK(g(15) == Approx(g_analytical(15)));
                CHECK(g(16) == Approx(g_analytical(16)));
                CHECK(g(17) == Approx(g_analytical(17)));
            }
        }
    }
}


