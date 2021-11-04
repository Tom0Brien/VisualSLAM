#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>
#include <Eigen/Core>

#include "../../src/utility.h"

SCENARIO("normcdf: x=0"){

    double x    = 0;
    WHEN("chi2inv is called"){
        double f = normcdf(x);
        THEN("Result is the same as MATLAB"){
            REQUIRE(f == Approx(0.5));    
        }
    }
}

SCENARIO("normcdf: x=1"){

    double x    = 1;
    WHEN("chi2inv is called"){
        double f = normcdf(x);
        THEN("Result is the same as MATLAB"){
            REQUIRE(f == Approx(0.841344746068543));    
        }
    }
}

SCENARIO("normcdf: x=2"){

    double x    = 2;
    WHEN("chi2inv is called"){
        double f = normcdf(x);
        THEN("Result is the same as MATLAB"){
            REQUIRE(f == Approx(0.977249868051821));    
        }
    }
}


