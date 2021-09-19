#ifndef UTILITY_H
#define UTILITY_H

#include <Eigen/Core>
#include <filesystem>
#include <vector>

std::vector<std::filesystem::path> getFilesWithExtension(std::filesystem::path const & root, std::string const & ext);





#endif