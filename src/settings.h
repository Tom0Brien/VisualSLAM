#ifndef SETTINGS_H
#define SETTINGS_H

#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <string>

class Settings
{
public:
    Settings();
    bool isInputGood() const;
    void read(const cv::FileNode& node);    // Read serialization for this class
    void write(cv::FileStorage& fs) const;  // Write serialization for this class
private:
    void validate();
    
public:
    cv::Size boardSize;                     // The size of the board -> Number of items by width and height
    float squareSize;                       // The size of a square in meters
    std::string input_dir;
    std::string input_ext;
private:
    bool _goodInput;
};

// Function prototypes
void read(const cv::FileNode& node, Settings& x, const Settings& default_value = Settings());

#endif