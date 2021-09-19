#include <filesystem>
#include <iostream>
#include <opencv2/core/persistence.hpp>
#include "settings.h"

Settings::Settings()
    : _goodInput(false)
{}

// Write serialization for this class
void Settings::write(cv::FileStorage& fs) const
{
    fs << "{"
        << "BoardSize_Height"           << boardSize.height
        << "BoardSize_Width"            << boardSize.width
        << "Input_Directory"            << input_dir
        << "Input_Extension"            << input_ext
        << "Square_Size"                << squareSize
        << "}";
}

// Read serialization for this class
void Settings::read(const cv::FileNode& node)
{
    node["BoardSize_Height"]            >> boardSize.height;
    node["BoardSize_Width" ]            >> boardSize.width;
    node["Input_Directory"]             >> input_dir;
    node["Input_Extension"]             >> input_ext;
    node["Square_Size"]                 >> squareSize;

    validate();
}

bool Settings::isInputGood() const {return _goodInput;}

void Settings::validate()
{
    _goodInput = true;
    if (boardSize.width <= 0 || boardSize.height <= 0)
    {
        std::cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << std::endl;
        _goodInput = false;
    }
    if (squareSize <= 10e-6)
    {
        std::cerr << "Invalid square size " << squareSize << std::endl;
        _goodInput = false;
    }
    if (!std::filesystem::is_directory(input_dir)){
        std::cerr << "Expected input path: " << input_dir << " to be a directory" << std::endl;
        _goodInput = false;
    }
}

void read(const cv::FileNode& node, Settings& x, const Settings& default_value)
{
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}