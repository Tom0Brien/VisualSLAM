#ifndef UTILITY_H
#define UTILITY_H

#include <Eigen/Core>
#include <filesystem>
#include <vector>

double chi2inv(double x, double nu);
double normcdf(double z);
std::vector<std::filesystem::path> getFilesWithExtension(std::filesystem::path const & root, std::string const & ext);



#endif