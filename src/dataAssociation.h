#ifndef DATA_ASSOCIATION_H
#define DATA_ASSOCIATION_H


#include <Eigen/Core>
#include <vector>
#include "cameraModel.hpp"




bool individualCompatibility(const int  & i, const int &  j, const int  & ny, const Eigen::MatrixXd & y, const Eigen::VectorXd & muY, const Eigen::MatrixXd & SYY, const std::vector<double> & chi2LUT);
bool jointCompatibility(const std::vector<int> & idx, const double & sU, const int  & ny, const Eigen::MatrixXd & y, const Eigen::VectorXd & muY, const Eigen::MatrixXd & SYY, const std::vector<double> & chi2LUT, double & surprisal);

double snn(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S, const Eigen::MatrixXd & y, const CameraParameters & param, std::vector<int>& idx, bool enforceJointCompatibility=false);
void withinConfidenceRegion(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S, const Eigen::MatrixXd & y, const CameraParameters & param, std::vector<int>& idx);


bool isWithinConfidenceRegionJ(const int j, const Eigen::VectorXd & mu, const Eigen::MatrixXd & S, const Eigen::MatrixXd & y, const CameraParameters & param);


int isFeatureWithinALandmark(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S, const Eigen::VectorXd & y, const CameraParameters & param);







#endif