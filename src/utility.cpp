#include "utility.h"
#include <cassert>
#include <cmath>

bool compareChar(const char & c1, const char & c2)
{
    if (c1 == c2)
        return true;
    else if (std::toupper(c1) == std::toupper(c2))
        return true;
    return false;
}
/*
 * Case Insensitive String Comparison
 */
bool caseInSensStringCompare(const std::string & str1, const std::string &str2)
{
    return ( (str1.size() == str2.size() ) &&
             std::equal(str1.begin(), str1.end(), str2.begin(), &compareChar) );
}

std::vector<std::filesystem::path> getFilesWithExtension(std::filesystem::path const & root, std::string const & ext)
{
    std::vector<std::filesystem::path> paths;

    if (std::filesystem::exists(root) && std::filesystem::is_directory(root))
    {

        for (auto const & entry : std::filesystem::recursive_directory_iterator(root))
        {
            bool strequal = caseInSensStringCompare(entry.path().extension().string(), ext);

            if (std::filesystem::is_regular_file(entry) && strequal){

                paths.emplace_back(entry.path().filename());
            }
        }
    }
    std::sort(paths.begin(), paths.end());
   
    return paths;
}  

