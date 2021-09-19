#define CATCH_CONFIG_ENABLE_BENCHMARKING
#define CATCH_CONFIG_COLOUR_ANSI
#include <catch2/catch.hpp>

#include "../../src/calibrate.h"
#include "../../src/plot.h"
#include "../../src/utility.h"

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

SCENARIO("plot: Zero landmarks, S is identity, mu is zero"){

    // std::filesystem::path calibrationFilePath = "data/camera.xml";
    // CameraParameters param;
    // importCalibrationData(calibrationFilePath, param);

    // // ------------------------------------------------------------
    // // Read settings
    // // ------------------------------------------------------------
    // Settings s;
    // std::filesystem::path inputSettingsFile = "data/settings.xml";

    // if (!std::filesystem::exists(inputSettingsFile)){
    //     std::cout << "No file on path: " << inputSettingsFile << std::endl << std::endl;
    //     parser.printMessage();
    //     return -1;
    // }

    // cv::FileStorage fs(inputSettingsFile.string(), cv::FileStorage::READ); // Read the settings
    // if (!fs.isOpened())
    // {
    //     std::cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << std::endl << std::endl;
    //     parser.printMessage();
    //     return -1;
    // }
    // fs["Settings"] >> s;
    // fs.release();

    // std::vector<std::filesystem::path> imgFiles;
    // imgFiles        = getFilesWithExtension(s.input_dir, s.input_ext);
    
    // if (imgFiles.size()==0){
    //     std::cerr << "No files found in path " << s.input_dir << " with extension " << s.input_ext << std::endl;
    //     assert(0);
    // }

    CameraParameters param;
    std::filesystem::path calibrationFilePath = "data/camera.xml";
    importCalibrationData(calibrationFilePath, param);


    std::vector<std::vector<cv::Point2f> > rQOi_set;
    cv::Size imageSize;

    int nr              = 4;
    int nx              = 12;
    int nx_all          = nx + nr*6;
    Eigen::VectorXd mu(nx_all);
    mu.setZero();
    Eigen::MatrixXd S = 0.01*Eigen::MatrixXd::Identity(nx_all,nx_all);
    mu.block(0, 0, 12, 1)       << 0,0,0,0,0,0,0,0,0,0,0,0;
    mu.block(nx + 0, 0, 6, 1)   << 0.1,  -0.1, 0.5,0,0,0;
    mu.block(nx + 6, 0, 6, 1)   << -0.1, -0.1 , 0.5,0,0,0;
    mu.block(nx + 12, 0, 6, 1)  << -0.1,  0.1 , 0.5,0,0,0;
    mu.block(nx + 18, 0, 6, 1)  << 0.1,  0.1, 0.5,0,0,0;

    

    assert(nx_all==S.rows());

    PlotHandles handles;
    bool doInteractor = true;
    if (!doInteractor){
        initPlotStates(mu, S, param, handles);
    }


    std::filesystem::path  outdir("out");
    if (!std::filesystem::is_directory(outdir)){
        bool isCreated = std::filesystem::create_directory(outdir);
        assert (isCreated);
    }

    cv::Mat view;
    view = cv::imread("data/1023.jpg");

    if (!doInteractor){
        updatePlotStates(view, mu, S, param, handles);
        // std::filesystem::path  outputPath;
        // outputPath               = outdir / imgFiles.at(k);
        // WriteImage(outputPath.string(), handles.renderWindow);
    }
    else
    {
        PlotHandles tmpHandles;
        initPlotStates(mu, S, param, tmpHandles);
        updatePlotStates(view, mu, S, param, tmpHandles);

        // std::filesystem::path  outputPath;
        // outputPath               = outdir / imgFiles.at(k);
        // WriteImage(outputPath.string(), tmpHandles.renderWindow);

        // -------------------------
        // Attach interactor for playing with the 3d interface
        // -------------------------
        vtkNew<vtkInteractorStyleTrackballCamera> threeDimInteractorStyle;
        vtkNew<vtkRenderWindowInteractor> threeDimInteractor;


        threeDimInteractor->SetInteractorStyle(threeDimInteractorStyle);
        threeDimInteractor->SetRenderWindow(tmpHandles.renderWindow);

        threeDimInteractor->Initialize();
        threeDimInteractor->Start();
    }   
}


