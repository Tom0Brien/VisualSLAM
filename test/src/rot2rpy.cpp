#include <catch2/catch.hpp>
#include <Eigen/Core>

#include "../../src/rotation.hpp"

SCENARIO("rot2rpy: R is identity"){

    Eigen::MatrixXd R(3,3);
    Eigen::VectorXd Theta;

    // R - [3 x 3]: 
    R <<                  1,                 0,                 0,
                            0,                 1,                 0,
                            0,                 0,                 1;
    WHEN("rot2rpy is called"){
        rot2rpy(R, Theta);
        //--------------------------------------------------------------------------------
        // Checks for Theta 
        //--------------------------------------------------------------------------------
        THEN("Theta is not empty"){
            REQUIRE(Theta.size()>0);
            
            AND_THEN("Theta has the right dimensions"){
                REQUIRE(Theta.rows()==3);
                REQUIRE(Theta.cols()==1);
                AND_THEN("Theta is correct"){

                    // Theta(:,1)
                    CHECK(Theta(0,0) == Approx(                   0));
                    CHECK(Theta(1,0) == Approx(                  -0));
                    CHECK(Theta(2,0) == Approx(                   0));

                }
            }
        }
    }
}

SCENARIO("rot2rpy: R is rotation about x axis by 0.5 rad"){

    Eigen::MatrixXd R(3,3);
    Eigen::VectorXd Theta;

    // R - [3 x 3]: 
    R <<                  1,                 0,                 0,
                            0,      0.8775825619,     -0.4794255386,
                            0,      0.4794255386,      0.8775825619;
    WHEN("rot2rpy is called"){
        rot2rpy(R, Theta);
        //--------------------------------------------------------------------------------
        // Checks for Theta 
        //--------------------------------------------------------------------------------
        THEN("Theta is not empty"){
            REQUIRE(Theta.size()>0);
            
            AND_THEN("Theta has the right dimensions"){
                REQUIRE(Theta.rows()==3);
                REQUIRE(Theta.cols()==1);
                AND_THEN("Theta is correct"){

                    // Theta(:,1)
                    CHECK(Theta(0,0) == Approx(                 0.5));
                    CHECK(Theta(1,0) == Approx(                  -0));
                    CHECK(Theta(2,0) == Approx(                   0));

                }
            }
        }
    }
}

SCENARIO("rot2rpy: R is rotation about y axis by 0.5 rad"){

    Eigen::MatrixXd R(3,3);
    Eigen::VectorXd Theta;

    // R - [3 x 3]: 
    R <<       0.8775825619,                 0,      0.4794255386,
                            0,                 1,                 0,
                -0.4794255386,                 0,      0.8775825619;
    WHEN("rot2rpy is called"){
        rot2rpy(R, Theta);
        //--------------------------------------------------------------------------------
        // Checks for Theta 
        //--------------------------------------------------------------------------------
        THEN("Theta is not empty"){
            REQUIRE(Theta.size()>0);
            
            AND_THEN("Theta has the right dimensions"){
                REQUIRE(Theta.rows()==3);
                REQUIRE(Theta.cols()==1);
                AND_THEN("Theta is correct"){

                    // Theta(:,1)
                    CHECK(Theta(0,0) == Approx(                   0));
                    CHECK(Theta(1,0) == Approx(                 0.5));
                    CHECK(Theta(2,0) == Approx(                   0));

                }
            }
        }
    }
}


SCENARIO("rot2rpy: R is rotation about z axis by 0.5 rad"){

    Eigen::MatrixXd R(3,3);
    Eigen::VectorXd Theta;

    // R - [3 x 3]: 
    R <<       0.8775825619,     -0.4794255386,                 0,
                 0.4794255386,      0.8775825619,                 0,
                            0,                 0,                 1;
    WHEN("rot2rpy is called"){
        rot2rpy(R, Theta);
        //--------------------------------------------------------------------------------
        // Checks for Theta 
        //--------------------------------------------------------------------------------
        THEN("Theta is not empty"){
            REQUIRE(Theta.size()>0);
            
            AND_THEN("Theta has the right dimensions"){
                REQUIRE(Theta.rows()==3);
                REQUIRE(Theta.cols()==1);
                AND_THEN("Theta is correct"){

                    // Theta(:,1)
                    CHECK(Theta(0,0) == Approx(                   0));
                    CHECK(Theta(1,0) == Approx(                  -0));
                    CHECK(Theta(2,0) == Approx(                 0.5));

                }
            }
        }
    }
}

SCENARIO("rot2rpy: R is rotation about all axes"){

    Eigen::MatrixXd R(3,3);
    Eigen::VectorXd Theta;

    // R - [3 x 3]: 
    R <<       0.4119822457,     -0.8337376518,     -0.3676304629,
               -0.05872664493,     -0.4269176213,      0.9023815855,
                -0.9092974268,     -0.3501754884,     -0.2248450954;
    WHEN("rot2rpy is called"){
        rot2rpy(R, Theta);
        //--------------------------------------------------------------------------------
        // Checks for Theta 
        //--------------------------------------------------------------------------------
        THEN("Theta is not empty"){
            REQUIRE(Theta.size()>0);
            Eigen::MatrixXd R_act;
            rpy2rot(Theta, R_act);

            AND_THEN("rot2rpy is an inverse mapping"){
                // R_act(:,1)
                CHECK(R_act(0,0) == Approx(R(0,0)));
                CHECK(R_act(1,0) == Approx(R(1,0)));
                CHECK(R_act(2,0) == Approx(R(2,0)));

                // R_act(:,2)
                CHECK(R_act(0,1) == Approx(R(0,1)));
                CHECK(R_act(1,1) == Approx(R(1,1)));
                CHECK(R_act(2,1) == Approx(R(2,1)));

                // R_act(:,3)
                CHECK(R_act(0,2) == Approx(R(0,2)));
                CHECK(R_act(1,2) == Approx(R(1,2)));
                CHECK(R_act(2,2) == Approx(R(2,2)));
            }
        }
    }
}
