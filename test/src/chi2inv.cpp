#include <catch2/catch.hpp>
#include <Eigen/Core>

#include "../../src/utility.h"

SCENARIO("chi2inv: c=0.5, nu=1"){

    double c    = 0.5;
    double nu   = 1;
    WHEN("chi2inv is called"){
        double f = chi2inv(c, nu);
        THEN("Result is the same as MATLAB"){
            REQUIRE(f == Approx(0.454936423119573));    
        }
    }
}

SCENARIO("chi2inv: c=0.5, nu=2"){

    double c    = 0.5;
    double nu   = 2;
    WHEN("chi2inv is called"){
        double f = chi2inv(c, nu);
        THEN("Result is the same as MATLAB"){
            REQUIRE(f == Approx(1.386294361119890));    
        }
    }
}

SCENARIO("chi2inv: c=0.5, nu=3"){

    double c    = 0.5;
    double nu   = 3;
    WHEN("chi2inv is called"){
        double f = chi2inv(c, nu);
        THEN("Result is the same as MATLAB"){
            REQUIRE(f == Approx(2.365973884375338));    
        }
    }
}


SCENARIO("chi2inv: c=0.75, nu=3"){

    double c    = 0.75;
    double nu   = 3;
    WHEN("chi2inv is called"){
        double f = chi2inv(c, nu);
        THEN("Result is the same as MATLAB"){
            REQUIRE(f == Approx(4.108344935632316));    
        }
    }
}