#ifndef PLOT_H
#define PLOT_H

#include <Eigen/Core>
#include <opencv2/core.hpp>



#include <vtkActor.h>
#include <vtkActor2D.h>
#include <vtkCellArray.h>
#include <vtkColor.h>
#include <vtkContourFilter.h>
#include <vtkCubeAxesActor.h>
#include <vtkDataSetMapper.h>
#include <vtkImageData.h>
#include <vtkImageMapper.h>
#include <vtkLine.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPyramid.h>
#include <vtkQuadric.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkSampleFunction.h>
#include <vtkSmartPointer.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>

#include "cameraModel.hpp"





// -------------------------------------------------------
// Bounds
// -------------------------------------------------------
struct Bounds{
    double xmin = 0.0, xmax = 0.0;
    double ymin = 0.0, ymax = 0.0;
    double zmin = 0.0, zmax = 0.0;
};
void bounds_getVTKBounds(const Bounds & bnds, double * bounds);
void bounds_setExtremity(const Bounds & bnds, Bounds & extremity);
void bounds_calculateMaxMinSigmaPoints(Bounds & bnds, const Eigen::VectorXd & murPNn, const Eigen::MatrixXd & SrPNn, const double sigma);


// -------------------------------------------------------
// QuadricPlot
// -------------------------------------------------------
struct QuadricPlot{
    Bounds bounds;
    vtkSmartPointer<vtkActor>            contourActor;
    vtkSmartPointer<vtkContourFilter>    contours;
    vtkSmartPointer<vtkPolyDataMapper>   contourMapper;
    vtkSmartPointer<vtkQuadric>          quadric;
    vtkSmartPointer<vtkSampleFunction>   sample;
    bool isInit             = false;
    const double value      = 0.0;
};

// QuadricPlot();
// QuadricPlot(const QuadricPlot &qp);
void quadricPlot_init(QuadricPlot & qp);
void quadricPlot_update(QuadricPlot & qp, const Eigen::VectorXd & murPNn, const Eigen::MatrixXd & SrPNn);
vtkActor * quadricPlot_getActor(const QuadricPlot & qp);

// -------------------------------------------------------
// FrustumPlot
// -------------------------------------------------------
struct FrustumPlot{
    vtkSmartPointer<vtkActor> pyramidActor;
    vtkSmartPointer<vtkCellArray> cells;
    vtkSmartPointer<vtkDataSetMapper> mapper;
    vtkSmartPointer<vtkPoints> pyramidPts;
    vtkSmartPointer<vtkPyramid> pyramid;
    vtkSmartPointer<vtkUnstructuredGrid> ug;
    Eigen::MatrixXd rPCc;
    Eigen::MatrixXd rPNn;
    bool isInit = false;
};

void frustumPlot_init(FrustumPlot & fp, const CameraParameters & param);
void frustumPlot_update(FrustumPlot & fp, Eigen::VectorXd & eta);
vtkActor * frustumPlot_getActor(const FrustumPlot & fp);

// -------------------------------------------------------
// AxisPlot
// -------------------------------------------------------
struct AxisPlot{
    vtkColor3d axis1Color;
    vtkColor3d axis2Color;
    vtkColor3d axis3Color;
    vtkSmartPointer<vtkCubeAxesActor> cubeAxesActor;
    bool isInit = false;
};
void axisPlot_init(AxisPlot & ap, vtkCamera * cam);
void axisPlot_update(AxisPlot & ap, Bounds & bounds);
vtkActor * axisPlot_getActor(const AxisPlot & ap);

// -------------------------------------------------------
// BasisPlot
// -------------------------------------------------------
struct BasisPlot{
    vtkSmartPointer<vtkActor>            basisActor;
    vtkSmartPointer<vtkCellArray>        lines;
    vtkSmartPointer<vtkLine>             line0, line1, line2;
    vtkSmartPointer<vtkPoints>           basisPts;
    vtkSmartPointer<vtkPolyData>         linesPolyData;
    vtkSmartPointer<vtkPolyDataMapper>   basisMapper;
    vtkSmartPointer<vtkUnsignedCharArray> colorSet;
    Eigen::MatrixXd rPNn;
    Eigen::VectorXd rCNn;
    bool isInit = false;
};

void basisPlot_init(BasisPlot & bp);
void basisPlot_update(BasisPlot & bp, const Eigen::VectorXd & eta);
vtkActor * basisPlot_getActor(const BasisPlot & bp);


// -------------------------------------------------------
// ImagePlot
// -------------------------------------------------------
struct ImagePlot{
    vtkSmartPointer<vtkImageData> viewVTK;
    vtkSmartPointer<vtkActor2D> imageActor2d;
    vtkSmartPointer<vtkImageMapper> imageMapper;
    cv::Mat cvVTKBuffer;
    double width, height;
    bool isInit = false;
};

void imagePlot_init(ImagePlot & ip, double rendererWidth, double rendererHeight);
void imagePlot_update(ImagePlot & ip, const cv::Mat &  view);
vtkActor2D * imagePlot_getActor2D(const ImagePlot & ip);

struct PlotHandles{
    QuadricPlot qp_camera;
    std::vector<QuadricPlot> qp_features;
    FrustumPlot fp;
    AxisPlot ap;
    BasisPlot bp;
    ImagePlot ip;
    vtkSmartPointer<vtkRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderer>     threeDimRenderer;
    vtkSmartPointer<vtkRenderer>     imageRenderer;
};


// -------------------------------------------------------
// Function prototypes
// -------------------------------------------------------
void plotFeatureGaussianConfidenceEllipse(cv::Mat & img, const Eigen::VectorXd & murPNn, const Eigen::MatrixXd & SrPNn, const Eigen::VectorXd & eta, const CameraParameters & param, const Eigen::Vector3d & color);
void plotFeatureGaussianConfidenceQuadric(vtkActor* contourActor, const Eigen::VectorXd & murPNn, const Eigen::MatrixXd & SrPNn);

void initPlotStates(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S, const CameraParameters & param, PlotHandles & handles);
void updatePlotStates(const cv::Mat & view, const Eigen::VectorXd & mu, const Eigen::MatrixXd & S, const CameraParameters & param, PlotHandles & handles);

void WriteImage(std::string const& fileName, vtkRenderWindow* renWin, bool rgba=false);

#endif
