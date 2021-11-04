#include <catch2/catch.hpp>
#include <Eigen/Core>

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#include "../../src/rotation.hpp"

SCENARIO("rotx: x = 0"){

    Eigen::MatrixXd R;
    double x = 0;

    WHEN("Calling rotx"){
        rotx(x, R);
        //--------------------------------------------------------------------------------
        // Checks for R 
        //--------------------------------------------------------------------------------
        THEN("R is not empty"){
            REQUIRE(R.size()>0);
            
            AND_THEN("R has the right dimensions"){
                REQUIRE(R.rows()==3);
                REQUIRE(R.cols()==3);
                AND_THEN("R is correct"){

                    // R(:,1)
                    CHECK(R(0,0) == Approx(                   1));
                    CHECK(R(1,0) == Approx(                   0));
                    CHECK(R(2,0) == Approx(                   0));

                    // R(:,2)
                    CHECK(R(0,1) == Approx(                   0));
                    CHECK(R(1,1) == Approx(                   1));
                    CHECK(R(2,1) == Approx(                   0));

                    // R(:,3)
                    CHECK(R(0,2) == Approx(                   0));
                    CHECK(R(1,2) == Approx(                  -0));
                    CHECK(R(2,2) == Approx(                   1));

                }
            }
        }
    }
}

SCENARIO("rotx: x = 0.1"){

    Eigen::MatrixXd R;
    double x = 0.1;

    WHEN("Calling rotx"){
        rotx(x, R);
        //--------------------------------------------------------------------------------
        // Checks for R 
        //--------------------------------------------------------------------------------
        THEN("R is not empty"){
            REQUIRE(R.size()>0);
            
            AND_THEN("R has the right dimensions"){
                REQUIRE(R.rows()==3);
                REQUIRE(R.cols()==3);
                AND_THEN("R is correct"){

                    // R(:,1)
                    CHECK(R(0,0) == Approx(                   1));
                    CHECK(R(1,0) == Approx(                   0));
                    CHECK(R(2,0) == Approx(                   0));

                    // R(:,2)
                    CHECK(R(0,1) == Approx(                   0));
                    CHECK(R(1,1) == Approx(      0.995004165278));
                    CHECK(R(2,1) == Approx(     0.0998334166468));

                    // R(:,3)
                    CHECK(R(0,2) == Approx(                   0));
                    CHECK(R(1,2) == Approx(    -0.0998334166468));
                    CHECK(R(2,2) == Approx(      0.995004165278));

                }
            }
        }
    }
}

SCENARIO("rotx: x = pi*5/3"){

    Eigen::MatrixXd R;
    double x = M_PI*5/3;

    WHEN("Calling rotx"){
        rotx(x, R);
        //--------------------------------------------------------------------------------
        // Checks for R 
        //--------------------------------------------------------------------------------
        THEN("R is not empty"){
            REQUIRE(R.size()>0);
            
            AND_THEN("R has the right dimensions"){
                REQUIRE(R.rows()==3);
                REQUIRE(R.cols()==3);
                AND_THEN("R is correct"){

                    // R(:,1)
                    CHECK(R(0,0) == Approx(                   1));
                    CHECK(R(1,0) == Approx(                   0));
                    CHECK(R(2,0) == Approx(                   0));

                    // R(:,2)
                    CHECK(R(0,1) == Approx(                   0));
                    CHECK(R(1,1) == Approx(                 0.5));
                    CHECK(R(2,1) == Approx(     -0.866025403784));

                    // R(:,3)
                    CHECK(R(0,2) == Approx(                   0));
                    CHECK(R(1,2) == Approx(      0.866025403784));
                    CHECK(R(2,2) == Approx(                 0.5));

                }
            }
        }
    }
}



SCENARIO("roty: x = 0"){

    Eigen::MatrixXd R;
    double x = 0;

    WHEN("Calling roty"){
        roty(x, R);
        //--------------------------------------------------------------------------------
        // Checks for R 
        //--------------------------------------------------------------------------------
        THEN("R is not empty"){
            REQUIRE(R.size()>0);
            
            AND_THEN("R has the right dimensions"){
                REQUIRE(R.rows()==3);
                REQUIRE(R.cols()==3);
                AND_THEN("R is correct"){

                    // R(:,1)
                    CHECK(R(0,0) == Approx(                   1));
                    CHECK(R(1,0) == Approx(                   0));
                    CHECK(R(2,0) == Approx(                  -0));

                    // R(:,2)
                    CHECK(R(0,1) == Approx(                   0));
                    CHECK(R(1,1) == Approx(                   1));
                    CHECK(R(2,1) == Approx(                   0));

                    // R(:,3)
                    CHECK(R(0,2) == Approx(                   0));
                    CHECK(R(1,2) == Approx(                   0));
                    CHECK(R(2,2) == Approx(                   1));

                }
            }
        }
    }
}

SCENARIO("roty: x = 0.1"){

    Eigen::MatrixXd R;
    double x = 0.1;

    WHEN("Calling roty"){
        roty(x, R);
        //--------------------------------------------------------------------------------
        // Checks for R 
        //--------------------------------------------------------------------------------
        THEN("R is not empty"){
            REQUIRE(R.size()>0);
            
            AND_THEN("R has the right dimensions"){
                REQUIRE(R.rows()==3);
                REQUIRE(R.cols()==3);
                AND_THEN("R is correct"){

                    // R(:,1)
                    CHECK(R(0,0) == Approx(      0.995004165278));
                    CHECK(R(1,0) == Approx(                   0));
                    CHECK(R(2,0) == Approx(    -0.0998334166468));

                    // R(:,2)
                    CHECK(R(0,1) == Approx(                   0));
                    CHECK(R(1,1) == Approx(                   1));
                    CHECK(R(2,1) == Approx(                   0));

                    // R(:,3)
                    CHECK(R(0,2) == Approx(     0.0998334166468));
                    CHECK(R(1,2) == Approx(                   0));
                    CHECK(R(2,2) == Approx(      0.995004165278));

                }
            }
        }
    }
}

SCENARIO("roty: x = pi*5/3"){

    Eigen::MatrixXd R;
    double x = M_PI*5/3;

    WHEN("Calling roty"){
        roty(x, R);
        //--------------------------------------------------------------------------------
        // Checks for R 
        //--------------------------------------------------------------------------------
        THEN("R is not empty"){
            REQUIRE(R.size()>0);
            
            AND_THEN("R has the right dimensions"){
                REQUIRE(R.rows()==3);
                REQUIRE(R.cols()==3);
                AND_THEN("R is correct"){

                    // R(:,1)
                    CHECK(R(0,0) == Approx(                 0.5));
                    CHECK(R(1,0) == Approx(                   0));
                    CHECK(R(2,0) == Approx(      0.866025403784));

                    // R(:,2)
                    CHECK(R(0,1) == Approx(                   0));
                    CHECK(R(1,1) == Approx(                   1));
                    CHECK(R(2,1) == Approx(                   0));

                    // R(:,3)
                    CHECK(R(0,2) == Approx(     -0.866025403784));
                    CHECK(R(1,2) == Approx(                   0));
                    CHECK(R(2,2) == Approx(                 0.5));

                }
            }
        }
    }
}


SCENARIO("rotz: x = 0"){

    Eigen::MatrixXd R;
    double x = 0;

    WHEN("Calling rotz"){
        rotz(x, R);
        //--------------------------------------------------------------------------------
        // Checks for R 
        //--------------------------------------------------------------------------------
        THEN("R is not empty"){
            REQUIRE(R.size()>0);
            
            AND_THEN("R has the right dimensions"){
                REQUIRE(R.rows()==3);
                REQUIRE(R.cols()==3);
                AND_THEN("R is correct"){

                    // R(:,1)
                    CHECK(R(0,0) == Approx(                   1));
                    CHECK(R(1,0) == Approx(                   0));
                    CHECK(R(2,0) == Approx(                   0));

                    // R(:,2)
                    CHECK(R(0,1) == Approx(                   0));
                    CHECK(R(1,1) == Approx(                   1));
                    CHECK(R(2,1) == Approx(                   0));

                    // R(:,3)
                    CHECK(R(0,2) == Approx(                   0));
                    CHECK(R(1,2) == Approx(                   0));
                    CHECK(R(2,2) == Approx(                   1));

                }
            }
        }
    }
}

SCENARIO("rotz: x = 0.1"){

    Eigen::MatrixXd R;
    double x = 0.1;

    WHEN("Calling rotz"){
        rotz(x, R);
        //--------------------------------------------------------------------------------
        // Checks for R 
        //--------------------------------------------------------------------------------
        THEN("R is not empty"){
            REQUIRE(R.size()>0);
            
            AND_THEN("R has the right dimensions"){
                REQUIRE(R.rows()==3);
                REQUIRE(R.cols()==3);
                AND_THEN("R is correct"){

                    // R(:,1)
                    CHECK(R(0,0) == Approx(      0.995004165278));
                    CHECK(R(1,0) == Approx(     0.0998334166468));
                    CHECK(R(2,0) == Approx(                   0));

                    // R(:,2)
                    CHECK(R(0,1) == Approx(    -0.0998334166468));
                    CHECK(R(1,1) == Approx(      0.995004165278));
                    CHECK(R(2,1) == Approx(                   0));

                    // R(:,3)
                    CHECK(R(0,2) == Approx(                   0));
                    CHECK(R(1,2) == Approx(                   0));
                    CHECK(R(2,2) == Approx(                   1));

                }
            }
        }
    }
}

SCENARIO("rotz: x = pi*5/3"){

    Eigen::MatrixXd R;
    double x = M_PI*5/3;

    WHEN("Calling rotz"){
        rotz(x, R);
        //--------------------------------------------------------------------------------
        // Checks for R 
        //--------------------------------------------------------------------------------
        THEN("R is not empty"){
            REQUIRE(R.size()>0);
            
            AND_THEN("R has the right dimensions"){
                REQUIRE(R.rows()==3);
                REQUIRE(R.cols()==3);
                AND_THEN("R is correct"){

                    // R(:,1)
                    CHECK(R(0,0) == Approx(                 0.5));
                    CHECK(R(1,0) == Approx(     -0.866025403784));
                    CHECK(R(2,0) == Approx(                   0));

                    // R(:,2)
                    CHECK(R(0,1) == Approx(      0.866025403784));
                    CHECK(R(1,1) == Approx(                 0.5));
                    CHECK(R(2,1) == Approx(                   0));

                    // R(:,3)
                    CHECK(R(0,2) == Approx(                   0));
                    CHECK(R(1,2) == Approx(                   0));
                    CHECK(R(2,2) == Approx(                   1));

                }
            }
        }
    }
}

SCENARIO("rpy2rot: Theta = [1;2;3]"){

    Eigen::MatrixXd R;
    Eigen::VectorXd Theta(3);
    // Theta - [3 x 1]: 
    Theta <<                    1,
                                2,
                                3;

    WHEN("Calling rotz"){
        rpy2rot(Theta, R);
        //--------------------------------------------------------------------------------
        // Checks for R 
        //--------------------------------------------------------------------------------
        THEN("R is not empty"){
            REQUIRE(R.size()>0);
            
            AND_THEN("R has the right dimensions"){
                REQUIRE(R.rows()==3);
                REQUIRE(R.cols()==3);
                AND_THEN("R is correct"){

                    // R(:,1)
                    CHECK(R(0,0) == Approx(      0.411982245666));
                    CHECK(R(1,0) == Approx(    -0.0587266449276));
                    CHECK(R(2,0) == Approx(     -0.909297426826));

                    // R(:,2)
                    CHECK(R(0,1) == Approx(     -0.833737651774));
                    CHECK(R(1,1) == Approx(     -0.426917621276));
                    CHECK(R(2,1) == Approx(     -0.350175488374));

                    // R(:,3)
                    CHECK(R(0,2) == Approx(     -0.367630462925));
                    CHECK(R(1,2) == Approx(      0.902381585483));
                    CHECK(R(2,2) == Approx(     -0.224845095366));

                }
            }
        }
    }
}