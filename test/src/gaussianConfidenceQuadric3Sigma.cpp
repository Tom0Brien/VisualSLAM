#include <catch2/catch.hpp>
#include <Eigen/Core>

#include "../../src/gaussian.hpp"

SCENARIO("gaussianConfidenceQuadric3Sigma: S is identity, mu is zero"){

    Eigen::MatrixXd S(3,3), Q;
    Eigen::VectorXd mu(3);

    // mu - [3 x 1]: 
    mu <<                  0,
                          -0,
                           0;

    // S - [3 x 3]: 
    S <<                  1,                 0,                 0,
                          0,                 1,                 0,
                          0,                 0,                 1;
    WHEN("gaussianConfidenceQuadric3Sigma is called"){
        gaussianConfidenceQuadric3Sigma(mu, S, Q);
        //--------------------------------------------------------------------------------
        // Checks for Q 
        //--------------------------------------------------------------------------------
        THEN("Q is not empty"){
            REQUIRE(Q.size()>0);
            
            AND_THEN("Q has the right dimensions"){
                REQUIRE(Q.rows()==4);
                REQUIRE(Q.cols()==4);
                AND_THEN("Q is correct"){

                    // Q(:,1)
                    CHECK(Q(0,0) == Approx(                   1));
                    CHECK(Q(1,0) == Approx(                   0));
                    CHECK(Q(2,0) == Approx(                   0));
                    CHECK(Q(3,0) == Approx(                   0));

                    // Q(:,2)
                    CHECK(Q(0,1) == Approx(                   0));
                    CHECK(Q(1,1) == Approx(                   1));
                    CHECK(Q(2,1) == Approx(                   0));
                    CHECK(Q(3,1) == Approx(                   0));

                    // Q(:,3)
                    CHECK(Q(0,2) == Approx(                   0));
                    CHECK(Q(1,2) == Approx(                   0));
                    CHECK(Q(2,2) == Approx(                   1));
                    CHECK(Q(3,2) == Approx(                   0));

                    // Q(:,4)
                    CHECK(Q(0,3) == Approx(                   0));
                    CHECK(Q(1,3) == Approx(                   0));
                    CHECK(Q(2,3) == Approx(                   0));
                    CHECK(Q(3,3) == Approx(      -14.1564136091));

                }
            }
        }
    }
}

SCENARIO("gaussianConfidenceQuadric3Sigma: S = diag(4,5,6), mu = [1;2;3]"){

    Eigen::MatrixXd S(3,3), Q;
    Eigen::VectorXd mu(3);

    // mu - [3 x 1]: 
    mu <<                  1,
                           2,
                           3;

    // S - [3 x 3]: 
    S <<                  4,                 0,                 0,
                          0,                 5,                 0,
                          0,                 0,                 6;

    WHEN("gaussianConfidenceQuadric3Sigma is called"){
        
        gaussianConfidenceQuadric3Sigma(mu, S, Q);

        //--------------------------------------------------------------------------------
        // Checks for Q 
        //--------------------------------------------------------------------------------
        THEN("Q is not empty"){
            REQUIRE(Q.size()>0);
            
            AND_THEN("Q has the right dimensions"){
                REQUIRE(Q.rows()==4);
                REQUIRE(Q.cols()==4);
                AND_THEN("Q is correct"){

                    // Q(:,1)
                    CHECK(Q(0,0) == Approx(              0.0625));
                    CHECK(Q(1,0) == Approx(                   0));
                    CHECK(Q(2,0) == Approx(                   0));
                    CHECK(Q(3,0) == Approx(             -0.0625));

                    // Q(:,2)
                    CHECK(Q(0,1) == Approx(                   0));
                    CHECK(Q(1,1) == Approx(                0.04));
                    CHECK(Q(2,1) == Approx(                   0));
                    CHECK(Q(3,1) == Approx(               -0.08));

                    // Q(:,3)
                    CHECK(Q(0,2) == Approx(                   0));
                    CHECK(Q(1,2) == Approx(                   0));
                    CHECK(Q(2,2) == Approx(     0.0277777777778));
                    CHECK(Q(3,2) == Approx(    -0.0833333333333));

                    // Q(:,4)
                    CHECK(Q(0,3) == Approx(             -0.0625));
                    CHECK(Q(1,3) == Approx(               -0.08));
                    CHECK(Q(2,3) == Approx(    -0.0833333333333));
                    CHECK(Q(3,3) == Approx(      -13.6839136091));

                }
            }
        }
    }
}

SCENARIO("gaussianConfidenceQuadric3Sigma: S is upper triangular, mu = [1;2;3]"){

    Eigen::MatrixXd S(3,3), Q;
    Eigen::VectorXd mu(3);

    // mu - [3 x 1]: 
    mu <<                  1,
                           2,
                           3;

    // S - [3 x 3]: 
    S <<      -0.6490137652,      -1.109613039,     -0.5586807645,
                          0,       -0.84555124,      0.1783802258,
                          0,                 0,     -0.1968614465;

    WHEN("gaussianConfidenceQuadric3Sigma is called"){
        
        gaussianConfidenceQuadric3Sigma(mu, S, Q);

        //--------------------------------------------------------------------------------
        // Checks for Q 
        //--------------------------------------------------------------------------------
        THEN("Q is not empty"){
            REQUIRE(Q.size()>0);
            
            AND_THEN("Q has the right dimensions"){
                REQUIRE(Q.rows()==4);
                REQUIRE(Q.cols()==4);
                AND_THEN("Q is correct"){

                    // Q(:,1)
                    CHECK(Q(0,0) == Approx(       44.9627201136));
                    CHECK(Q(1,0) == Approx(      -9.04064936522));
                    CHECK(Q(2,0) == Approx(      -31.5188988284));
                    CHECK(Q(3,0) == Approx(       67.6752751019));

                    // Q(:,2)
                    CHECK(Q(0,1) == Approx(      -9.04064936522));
                    CHECK(Q(1,1) == Approx(       2.54708314668));
                    CHECK(Q(2,1) == Approx(       5.44359034351));
                    CHECK(Q(3,1) == Approx(      -12.3842879587));

                    // Q(:,3)
                    CHECK(Q(0,2) == Approx(      -31.5188988284));
                    CHECK(Q(1,2) == Approx(       5.44359034351));
                    CHECK(Q(2,2) == Approx(       25.8035022835));
                    CHECK(Q(3,2) == Approx(      -56.7787887092));

                    // Q(:,4)
                    CHECK(Q(0,3) == Approx(       67.6752751019));
                    CHECK(Q(1,3) == Approx(      -12.3842879587));
                    CHECK(Q(2,3) == Approx(      -56.7787887092));
                    CHECK(Q(3,3) == Approx(       113.273253334));

                }
            }
        }
    }
}

