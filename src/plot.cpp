#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>


#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>

#define vtkRenderingContext2D_AUTOINIT 1(vtkRenderingContextOpenGL2)
#define vtkRenderingCore_AUTOINIT 3(vtkInteractionStyle,vtkRenderingFreeType,vtkRenderingOpenGL2)
#define vtkRenderingOpenGL2_AUTOINIT 1(vtkRenderingGL2PSOpenGL2)

#include <vtkActor.h>
#include <vtkAxesActor.h>
#include <vtkAxisFollower.h>
#include <vtkBMPWriter.h>
#include <vtkCamera.h>
#include <vtkCaptionActor2D.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkColor.h>
#include <vtkContextInteractorStyle.h>
#include <vtkContourFilter.h>
#include <vtkCubeAxesActor.h>
#include <vtkDataSetMapper.h>
#include <vtkGeometryFilter.h>
#include <vtkImageActor.h>
#include <vtkImageCast.h>
#include <vtkImageConstantPad.h>
#include <vtkImageData.h>
#include <vtkImageGradient.h>
#include <vtkImageImport.h>
#include <vtkImageLuminance.h>
#include <vtkImageMapper.h>
#include <vtkInteractorStyleImage.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkJPEGWriter.h>
#include <vtkLine.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkOutlineFilter.h>
#include <vtkPlaneSource.h>
#include <vtkPNGWriter.h>
#include <vtkPNMWriter.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPostScriptWriter.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkPyramid.h>
#include <vtkQuadric.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSampleFunction.h>
#include <vtkSmartPointer.h>
#include <vtkStripper.h>
#include <vtkTextProperty.h>
#include <vtkThreshold.h>
#include <vtkTIFFWriter.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkWindowToImageFilter.h>


// For compatibility with new VTK generic data arrays
#ifdef vtkGenericDataArray_h
#define InsertNextTupleValue InsertNextTypedTuple
#endif

#include <iostream>

#include "cameraModel.hpp"
#include "gaussian.hpp"
#include "plot.h"
#include "rotation.hpp"
#include "utility.h"


// -------------------------------------------------------
// Colouring
// -------------------------------------------------------

// Inputs
// H \in [0, 360]
// S \in [0, 1]
// V \in [0, 1]
// Outputs
// R \in [0, 1]
// R \in [0, 1]
// R \in [0, 1]
void hsv2rgb(const double & h, const double & s, const double & v, double & r, double & g, double & b){

    bool hIsValid = 0 <= h && h <=  360.0;
    bool sIsValid = 0 <= s && s <=  1.0;
    bool vIsValid = 0 <= v && v <=  1.0;

    assert(hIsValid);
    assert(sIsValid);
    assert(vIsValid);

    // https://en.wikipedia.org/wiki/HSL_and_HSV#HSV_to_RGB

    double  c, x, r1, g1, b1, m;
    int hp;
    // shift the hue to the range [0, 360] before performing calculations
    hp  = (int)(h / 60.);
    c   = v*s;
    x   = c * (1 - std::abs((hp % 2) - 1));

    switch(hp) {
        case 0: r1 = c; g1 = x; b1 = 0; break;
        case 1: r1 = x; g1 = c; b1 = 0; break;
        case 2: r1 = 0; g1 = c; b1 = x; break;
        case 3: r1 = 0; g1 = x; b1 = c; break;
        case 4: r1 = x; g1 = 0; b1 = c; break;
        case 5: r1 = c; g1 = 0; b1 = x; break;
    }
    m   = v - c;
    r   = r1 + m;
    g   = g1 + m;
    b   = b1 + m;
}


void openCV2VTK(const cv::Mat & viewCVRGB, vtkImageData* viewVTK)
{
    assert( viewCVRGB.data != NULL );

    vtkNew<vtkImageImport> importer;
    if ( viewVTK )
    {
        importer->SetOutput( viewVTK );
    }
    importer->SetDataSpacing( 1, 1, 1 );
    importer->SetDataOrigin( 0, 0, 0 );
    importer->SetWholeExtent(   0, viewCVRGB.size().width-1, 0,
                            viewCVRGB.size().height-1, 0, 0 );
    importer->SetDataExtentToWholeExtent();
    importer->SetDataScalarTypeToUnsignedChar();
    importer->SetNumberOfScalarComponents( viewCVRGB.channels() );
    importer->SetImportVoidPointer( viewCVRGB.data );
    importer->Update();

}


// -------------------------------------------------------
// Bounds
// -------------------------------------------------------

void bounds_getVTKBounds(const Bounds & bnds, double * bounds){
    bounds[0] = bnds.xmin;
    bounds[1] = bnds.xmax;
    bounds[2] = bnds.ymin;
    bounds[3] = bnds.ymax;
    bounds[4] = bnds.zmin;
    bounds[5] = bnds.zmax;
}

void bounds_setExtremity(const Bounds & bnds, Bounds & extremity){

    extremity.xmin = std::min(extremity.xmin, bnds.xmin);
    extremity.xmax = std::max(extremity.xmax, bnds.xmax);

    extremity.ymin = std::min(extremity.ymin, bnds.ymin);
    extremity.ymax = std::max(extremity.ymax, bnds.ymax);

    extremity.zmin = std::min(extremity.zmin, bnds.zmin);
    extremity.zmax = std::max(extremity.zmax, bnds.zmax);
}

void bounds_calculateMaxMinSigmaPoints(Bounds & bnds, const Eigen::VectorXd & murPNn, const Eigen::MatrixXd & SrPNn, const double sigma){
    const int nx  = 3;
    assert(murPNn.rows() == nx);
    assert(SrPNn.rows() == nx);
    assert(SrPNn.cols() == nx);

    // Marginals
    Eigen::MatrixXd Ax(1,3), Ay(1,3), Az(1,3);
    Eigen::VectorXd mux, muy, muz;
    Eigen::MatrixXd Sxx, Syy, Szz;

    Ax << 1,0,0;
    Ay << 0,1,0;
    Az << 0,0,1;

    mux     = Ax*murPNn;
    muy     = Ay*murPNn;
    muz     = Az*murPNn;

    Eigen::MatrixXd Ss;
    Eigen::HouseholderQR<Eigen::MatrixXd> qr;
    Ss                          = SrPNn*Ax.transpose();
    Ss                          = Ss.householderQr().matrixQR().triangularView<Eigen::Upper>();
    Sxx                         = Ss.topRows(1);

    Ss                          = SrPNn*Ay.transpose();
    Ss                          = Ss.householderQr().matrixQR().triangularView<Eigen::Upper>();
    Syy                         = Ss.topRows(1);

    Ss                          = SrPNn*Az.transpose();
    Ss                          = Ss.householderQr().matrixQR().triangularView<Eigen::Upper>();
    Szz                         = Ss.topRows(1);

    bnds.xmin                   = mux(0) - sigma*std::abs(Sxx(0,0));
    bnds.xmax                   = mux(0) + sigma*std::abs(Sxx(0,0));

    bnds.ymin                   = muy(0) - sigma*std::abs(Syy(0,0));
    bnds.ymax                   = muy(0) + sigma*std::abs(Syy(0,0));

    bnds.zmin                   = muz(0) - sigma*std::abs(Szz(0,0));
    bnds.zmax                   = muz(0) + sigma*std::abs(Szz(0,0));
}

// -------------------------------------------------------
// QuadricPlot
// -------------------------------------------------------

void quadricPlot_init(QuadricPlot & qp){

    qp.quadric = vtkSmartPointer<vtkQuadric>::New();

    int ns          = 25;
    qp.sample = vtkSmartPointer<vtkSampleFunction>::New();
    qp.sample->SetSampleDimensions(ns, ns, ns);
    qp.sample->SetImplicitFunction(qp.quadric);

    // create the 0 isosurface
    qp.contours = vtkSmartPointer<vtkContourFilter>::New();
    qp.contours->SetInputConnection(qp.sample->GetOutputPort());
    qp.contours->GenerateValues(1, qp.value, qp.value);

    qp.contourMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    qp.contourMapper->SetInputConnection(qp.contours->GetOutputPort());
    qp.contourMapper->ScalarVisibilityOff();

    qp.contourActor = vtkSmartPointer<vtkActor>::New();
    qp.contourActor->SetMapper(qp.contourMapper);

    qp.isInit  = true;

}
void quadricPlot_update(QuadricPlot & qp, const Eigen::VectorXd & murPNn, const Eigen::MatrixXd & SrPNn){ 
    assert(qp.isInit);

    const int nx = 3;
    assert(murPNn.rows() == nx);
    assert(SrPNn.rows() == nx);
    assert(SrPNn.cols() == nx);

    Eigen::MatrixXd Q;
    bounds_calculateMaxMinSigmaPoints(qp.bounds, murPNn, SrPNn, 6);



    // ---------------------------------------------------
    // TODO
    // ---------------------------------------------------
    double a0, a1, a2, a3, a4, a5, a6, a7, a8, a9;

    std::cout << "Here" << std::endl;

    gaussianConfidenceQuadric3Sigma(murPNn,SrPNn,Q);
    
    // Diagonals
    a0      = Q(0,0);
    a1      = Q(1,1);
    a2      = Q(2,2);
    a9      = Q(3,3);
    // Cross terms
    a3      = 2*Q(0,1);
    a4      = 2*Q(1,2);
    a5      = 2*Q(0,2);
    a6      = 2*Q(0,3);
    a7      = 2*Q(1,3);
    a8      = 2*Q(2,3);

    // ---------------------------------------------------
    // 
    
    
    qp.quadric->SetCoefficients(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);

    double boundsVTK[6];
    bounds_getVTKBounds(qp.bounds, boundsVTK);
    qp.sample->SetModelBounds(boundsVTK);
}

vtkActor * quadricPlot_getActor(const QuadricPlot & qp){ 
    assert(qp.isInit);

    return qp.contourActor;
}


// -------------------------------------------------------
// FrustumPlot
// -------------------------------------------------------
void frustumPlot_init(FrustumPlot & fp, const CameraParameters & param){

    fp.rPCc    = Eigen::MatrixXd::Zero(3,5);
    fp.rPNn    = Eigen::MatrixXd::Zero(3,5);

    int nu, nv;
    nu      = param.imageSize.width;
    nv      = param.imageSize.height;

    std::vector<cv::Point2f> p_cv;
    p_cv.push_back(cv::Point2f(   0,    0));
    p_cv.push_back(cv::Point2f(nu-1,    0));
    p_cv.push_back(cv::Point2f(nu-1, nv-1));
    p_cv.push_back(cv::Point2f(   0, nv-1));

    std::vector<cv::Point2f> rZCc2_cv;
    cv::undistortPoints(p_cv, rZCc2_cv, param.Kc, param.distCoeffs);

    Eigen::MatrixXd rZCc2(2,4);
    for (int i = 0; i < rZCc2.cols(); ++i)
    {
        rZCc2.col(i)    << rZCc2_cv[i].x,  rZCc2_cv[i].y;
    } 

    Eigen::MatrixXd rZCc(3,4), nrZCc;
    rZCc.fill(1);
    rZCc.topRows(2)     = rZCc2;
    nrZCc               = rZCc.colwise().squaredNorm().cwiseSqrt();

    for (int i = 0; i < rZCc.cols(); ++i)
    {
        rZCc.col(i)            = rZCc.col(i) / nrZCc(0,i);
    }

    
    double d            = 0.5;
    fp.rPCc.block(0,0,3,4) = d*rZCc;
    fp.rPCc.block(0,4,3,1) << 0,0,0;

    fp.pyramidPts = vtkSmartPointer<vtkPoints>::New();
    fp.pyramidPts->SetNumberOfPoints(5);

    fp.pyramid = vtkSmartPointer<vtkPyramid>::New();
    fp.pyramid->GetPointIds()->SetId(0, 0);
    fp.pyramid->GetPointIds()->SetId(1, 1);
    fp.pyramid->GetPointIds()->SetId(2, 2);
    fp.pyramid->GetPointIds()->SetId(3, 3);
    fp.pyramid->GetPointIds()->SetId(4, 4);

    fp.cells = vtkSmartPointer<vtkCellArray>::New();
    fp.cells->InsertNextCell(fp.pyramid);

    fp.ug = vtkSmartPointer<vtkUnstructuredGrid>::New();
    fp.ug->SetPoints(fp.pyramidPts);
    fp.ug->InsertNextCell(fp.pyramid->GetCellType(), fp.pyramid->GetPointIds());

    fp.mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    fp.mapper->SetInputData(fp.ug);

    vtkNew<vtkNamedColors> colors;
    fp.pyramidActor = vtkSmartPointer<vtkActor>::New();
    fp.pyramidActor->SetMapper(fp.mapper);
    fp.pyramidActor->GetProperty()->SetColor(colors->GetColor3d("Tomato").GetData());
    fp.pyramidActor->GetProperty()->SetOpacity(0.1);   

    fp.isInit  = true;

};

void frustumPlot_update(FrustumPlot & fp, Eigen::VectorXd & eta){   
    assert(fp.isInit);

    assert(eta.rows() == 6);
    assert(eta.cols() == 1);

    Eigen::VectorXd rCNn    = eta.head(3);
    Eigen::VectorXd Thetanc = eta.tail(3);
    Eigen::MatrixXd Rnc;
    rpy2rot(Thetanc, Rnc);

    fp.rPNn    =   (Rnc*fp.rPCc).colwise() + rCNn;

    fp.pyramidPts->SetPoint(0, fp.rPNn.col(0).data());
    fp.pyramidPts->SetPoint(1, fp.rPNn.col(1).data());
    fp.pyramidPts->SetPoint(2, fp.rPNn.col(2).data());
    fp.pyramidPts->SetPoint(3, fp.rPNn.col(3).data());
    fp.pyramidPts->SetPoint(4, fp.rPNn.col(4).data());
    
    fp.pyramidPts->Modified();
    fp.ug->Modified();
    fp.mapper->Modified();
};

vtkActor * frustumPlot_getActor(const FrustumPlot & fp){
    assert(fp.isInit);

    return fp.pyramidActor;
};


// -------------------------------------------------------
// AxisPlot
// -------------------------------------------------------

void axisPlot_init(AxisPlot & ap, vtkCamera * cam){
    int fontsize    = 48;

    vtkNew<vtkNamedColors> colors;
    ap.axis1Color = colors->GetColor3d("Salmon");
    ap.axis2Color = colors->GetColor3d("PaleGreen");
    ap.axis3Color = colors->GetColor3d("LightSkyBlue");

    ap.cubeAxesActor = vtkSmartPointer<vtkCubeAxesActor>::New();
    ap.cubeAxesActor->SetCamera(cam);
    ap.cubeAxesActor->GetTitleTextProperty(0)->SetColor(ap.axis1Color.GetData());
    ap.cubeAxesActor->GetTitleTextProperty(0)->SetFontSize(fontsize);
    ap.cubeAxesActor->GetLabelTextProperty(0)->SetColor(ap.axis1Color.GetData());
    ap.cubeAxesActor->GetLabelTextProperty(0)->SetFontSize(fontsize);

    ap.cubeAxesActor->GetTitleTextProperty(1)->SetColor(ap.axis2Color.GetData());
    ap.cubeAxesActor->GetTitleTextProperty(1)->SetFontSize(fontsize);
    ap.cubeAxesActor->GetLabelTextProperty(1)->SetColor(ap.axis2Color.GetData());
    ap.cubeAxesActor->GetLabelTextProperty(1)->SetFontSize(fontsize);

    ap.cubeAxesActor->GetTitleTextProperty(2)->SetColor(ap.axis3Color.GetData());
    ap.cubeAxesActor->GetTitleTextProperty(2)->SetFontSize(fontsize);
    ap.cubeAxesActor->GetLabelTextProperty(2)->SetColor(ap.axis3Color.GetData());
    ap.cubeAxesActor->GetLabelTextProperty(2)->SetFontSize(fontsize);
    
    ap.cubeAxesActor->SetXTitle("N - [m]");
    ap.cubeAxesActor->SetYTitle("E - [m]");
    ap.cubeAxesActor->SetZTitle("D - [m]");

    ap.cubeAxesActor->XAxisMinorTickVisibilityOn();
    ap.cubeAxesActor->YAxisMinorTickVisibilityOn();
    ap.cubeAxesActor->ZAxisMinorTickVisibilityOn();

    // cubeAxesActor->SetFlyModeToStaticEdges();
    ap.cubeAxesActor->SetFlyModeToFurthestTriad();
    ap.cubeAxesActor->SetUseTextActor3D(1); 

    ap.isInit  = true;

}

void axisPlot_update(AxisPlot & ap, Bounds & bounds){   
    assert(ap.isInit);

    double boundsVTK[6];
    bounds_getVTKBounds(bounds, boundsVTK);

    ap.cubeAxesActor->SetBounds(boundsVTK);
    
}

vtkActor * axisPlot_getActor(AxisPlot & ap){
    assert(ap.isInit);

    return ap.cubeAxesActor;
}

// -------------------------------------------------------
// BasisPlot
// -------------------------------------------------------

void basisPlot_init(BasisPlot & bp){

    bp.basisPts = vtkSmartPointer<vtkPoints>::New();
    bp.basisPts->SetNumberOfPoints(4);
    // Add the points to the polydata container
    bp.linesPolyData = vtkSmartPointer<vtkPolyData>::New();
    bp.linesPolyData->SetPoints(bp.basisPts);

    // Create the first line (between Origin and P0)
    bp.line0 = vtkSmartPointer<vtkLine>::New();
    bp.line0->GetPointIds()->SetId(
    0,
    0); // the second 0 is the index of the Origin in linesPolyData's points
    bp.line0->GetPointIds()->SetId(
    1, 1); // the second 1 is the index of P0 in linesPolyData's points

    // Create the second line (between Origin and P1)
    bp.line1 = vtkSmartPointer<vtkLine>::New();
    bp.line1->GetPointIds()->SetId(
    0,
    0); // the second 0 is the index of the Origin in linesPolyData's points
    bp.line1->GetPointIds()->SetId(
    1, 2); // 2 is the index of P1 in linesPolyData's points

    // Create the second line (between Origin and P2)
    bp.line2 = vtkSmartPointer<vtkLine>::New();
    bp.line2->GetPointIds()->SetId(
    0,
    0); // the second 0 is the index of the Origin in linesPolyData's points
    bp.line2->GetPointIds()->SetId(
    1, 3); // 3 is the index of P2 in linesPolyData's points

    // Create a vtkCellArray container and store the lines in it
    bp.lines = vtkSmartPointer<vtkCellArray>::New();
    bp.lines->InsertNextCell(bp.line0);
    bp.lines->InsertNextCell(bp.line1);
    bp.lines->InsertNextCell(bp.line2);

    // Add the lines to the polydata container
    bp.linesPolyData->SetLines(bp.lines);


    vtkNew<vtkNamedColors> colors;
    bp.colorSet = vtkSmartPointer<vtkUnsignedCharArray>::New();
    bp.colorSet->SetNumberOfComponents(3);
    bp.colorSet->InsertNextTupleValue(colors->GetColor3ub("red").GetData());
    bp.colorSet->InsertNextTupleValue(colors->GetColor3ub("green").GetData());
    bp.colorSet->InsertNextTupleValue(colors->GetColor3ub("blue").GetData());

    bp.linesPolyData->GetCellData()->SetScalars(bp.colorSet);        

    // https://discourse.vtk.org/t/how-to-update-adjacent-vtkpolydata-instances-after-a-transformation/1325
    bp.basisMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    bp.basisMapper->SetInputData(bp.linesPolyData);

    bp.basisActor = vtkSmartPointer<vtkActor>::New();
    bp.basisActor->SetMapper(bp.basisMapper);
    bp.basisActor->GetProperty()->SetLineWidth(4);
    bp.rPNn     = Eigen::MatrixXd::Zero(3, 3);
    bp.rCNn     = Eigen::VectorXd::Zero(3, 1);

    bp.isInit   = true;
}

void basisPlot_update(BasisPlot & bp, const Eigen::VectorXd & eta){
    assert(bp.isInit);

    assert(eta.rows() == 6);
    assert(eta.cols() == 1);

    bp.rCNn                 = eta.head(3);
    Eigen::VectorXd Thetanc = eta.tail(3);
    Eigen::MatrixXd Rnc;
    rpy2rot(Thetanc, Rnc);

    double basisScale   = 0.2; 
    bp.rPNn             = (basisScale*Rnc).colwise() + bp.rCNn;


    bp.basisPts->SetPoint(0, bp.rCNn.data());
    bp.basisPts->SetPoint(1, bp.rPNn.col(0).data());
    bp.basisPts->SetPoint(2, bp.rPNn.col(1).data());
    bp.basisPts->SetPoint(3, bp.rPNn.col(2).data());
    bp.basisPts->Modified();
    bp.linesPolyData->Modified();
};

vtkActor * basisPlot_getActor(const BasisPlot & bp){
    assert(bp.isInit);
    return bp.basisActor;
}

// -------------------------------------------------------
// ImagePlot
// -------------------------------------------------------
void imagePlot_init(ImagePlot & ip, double rendererWidth, double rendererHeight){
    ip.width   = rendererWidth;
    ip.height  = rendererHeight;

    ip.viewVTK = vtkSmartPointer<vtkImageData>::New();
    ip.imageMapper = vtkSmartPointer<vtkImageMapper>::New();
    ip.imageMapper->SetInputData(ip.viewVTK);
    ip.imageMapper->SetColorWindow(255.0);
    ip.imageMapper->SetColorLevel(127.5);
    
    ip.imageActor2d = vtkSmartPointer<vtkActor2D>::New();
    ip.imageActor2d->SetMapper(ip.imageMapper);
    ip.isInit  = true;
}

void imagePlot_update(ImagePlot & ip, const cv::Mat &  view){
    assert(ip.isInit);

    cv::Mat viewCVrgb, tmp;
    cv::resize(view, tmp, cv::Size(ip.width, ip.height), cv::INTER_LINEAR);
    cv::cvtColor(tmp, viewCVrgb, cv::COLOR_BGR2RGB);
    cv::flip(viewCVrgb, ip.cvVTKBuffer, 0);
    openCV2VTK(ip.cvVTKBuffer, ip.viewVTK);
}

vtkActor2D * imagePlot_getActor2D(ImagePlot & ip){
    assert(ip.isInit);
    return ip.imageActor2d;
}



// -------------------------------------------------------
// Plotting of states
// -------------------------------------------------------
void initPlotStates(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S, const CameraParameters & param, PlotHandles & handles){

    int nx_all  = mu.rows();
    assert(S.rows() == nx_all);
    assert(S.cols() == nx_all);

    int nx      = 12;                    // Number of camera states
    // assert( (nx_all - nx)>0);           // Check that there are features in the scene
    assert( (nx_all - nx)% 6 == 0);     // Check that the dimension for features is correct
    int nr      = (nx_all - nx)/6;      // Number of features
    std::cout << "nr " << nr << std::endl;

    double aspectRatio  = (1.0*param.imageSize.width)/param.imageSize.height;

    double windowHeight       = 512;
    double windowWidth        = 2*aspectRatio*windowHeight;

    vtkNew<vtkNamedColors> colors;
    double quadricViewport[4]       = {0.5, 0.0, 1.0, 1.0};
    handles.threeDimRenderer = vtkSmartPointer<vtkRenderer>::New();
    handles.threeDimRenderer->SetViewport(quadricViewport);
    handles.threeDimRenderer->SetBackground(colors->GetColor3d("slategray").GetData());

    double imageViewport[4]         = {0.0, 0.0, 0.5, 1.0};
    handles.imageRenderer = vtkSmartPointer<vtkRenderer>::New();
    handles.imageRenderer->SetViewport(imageViewport);
    handles.imageRenderer->SetBackground(colors->GetColor3d("white").GetData());

    handles.renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    handles.renderWindow->SetSize(windowWidth, windowHeight);
    handles.renderWindow->SetMultiSamples(0);
    handles.renderWindow->AddRenderer(handles.threeDimRenderer);
    handles.renderWindow->AddRenderer(handles.imageRenderer);


    axisPlot_init(handles.ap, handles.threeDimRenderer->GetActiveCamera());
    basisPlot_init(handles.bp);
    frustumPlot_init(handles.fp, param);
    imagePlot_init(handles.ip, windowWidth/2, windowHeight);

    // Quadric surfaces
    quadricPlot_init(handles.qp_camera);
    handles.qp_features.clear();

    for (int i = 0; i < nr; ++i)
    {
        /* code */
        QuadricPlot qp;
        quadricPlot_init(qp);
        handles.qp_features.push_back(qp);
        handles.threeDimRenderer->AddActor(quadricPlot_getActor(qp));
    }
    handles.threeDimRenderer->AddActor(axisPlot_getActor(handles.ap));
    handles.threeDimRenderer->AddActor(basisPlot_getActor(handles.bp));
    handles.threeDimRenderer->AddActor(frustumPlot_getActor(handles.fp));
    handles.threeDimRenderer->AddActor(quadricPlot_getActor(handles.qp_camera));
    handles.imageRenderer->AddActor2D(imagePlot_getActor2D(handles.ip));
}

void updatePlotStates(const cv::Mat & view, const Eigen::VectorXd & mu, const Eigen::MatrixXd & S, const CameraParameters & param, PlotHandles & handles){
    
    int nx_all  = mu.rows();
    assert(S.rows() == nx_all);
    assert(S.cols() == nx_all);

    int nx      = 12;                    // Number of camera state   s
    // assert( (nx_all - nx)>0);           // Check that there are features in the scene
    assert( (nx_all - nx)% 6 == 0);     // Check that the dimension for features is correct
    int nr      = (nx_all - nx)/6 ;      // Number of features
    Eigen::VectorXd eta    = mu.head(6);

    Eigen::VectorXd rCNn    = eta.head(3);
    Eigen::VectorXd Thetanc = eta.tail(3);
    Eigen::MatrixXd Rnc;
    rpy2rot(Thetanc, Rnc);


    Eigen::VectorXd murCNn(3);
    Eigen::MatrixXd SrCNn(3,3);
    // ---------------------------------------------------
    // TODO
    // ---------------------------------------------------

    // Calculate marginal distribution for camera position
    murCNn = mu.segment(0,3);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(S.middleCols(0,3));
    SrCNn = Eigen::MatrixXd(qr.matrixQR().triangularView<Eigen::Upper>()).block(0,0,3,3);
    

    // ---------------------------------------------------
    // 
    double r,g,b;   
    hsv2rgb(330, 1., 1., r, g, b);
    std::cout << "murCNn " << murCNn << std::endl;
    std::cout << "murCNn rows" << murCNn.rows() << std::endl;
    std::cout << "SrCNn" << SrCNn << std::endl;
    std::cout << "SrCNn rows" << SrCNn.rows() << std::endl;
    std::cout << "SrCNn cols" << SrCNn.cols() << std::endl;
    quadricPlot_update(handles.qp_camera, murCNn, SrCNn);
    quadricPlot_getActor(handles.qp_camera)->GetProperty()->SetOpacity(0.1);
    quadricPlot_getActor(handles.qp_camera)->GetProperty()->SetColor(r,g,b);

    bool r1 = std::abs(murCNn(0) - rCNn(0)) < 1e-6;
    bool r2 = std::abs(murCNn(1) - rCNn(1)) < 1e-6;
    bool r3 = std::abs(murCNn(2) - rCNn(2)) < 1e-6;

    assert (r1);
    assert (r2);
    assert (r3);


    Bounds globalBounds;
    bounds_setExtremity(handles.qp_camera.bounds, globalBounds); 

    cv::Mat outView;
    outView     = view.clone();
    for (int i = 0; i < nr; ++i)
    {

        Eigen::VectorXd murPNn(3);
        Eigen::MatrixXd SrPNn(3,3);
        // ---------------------------------------------------
        // TODO
        // ---------------------------------------------------

        // Calculate marginal distribution for feature positions
        murPNn = mu.segment(nx+i*6,3);
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(S.middleCols(nx+i*6,3));
        SrPNn = Eigen::MatrixXd(qr.matrixQR().triangularView<Eigen::Upper>()).block(0,0,3,3);
        // ---------------------------------------------------
        // 


        // Add components to render
        hsv2rgb(300*(i)/(nr), 1., 1., r, g, b);
        Eigen::Vector3d rgb;
        rgb(0) = r*255;
        rgb(1) = g*255;
        rgb(2) = b*255;

        plotFeatureGaussianConfidenceEllipse(outView, murPNn, SrPNn, eta, param, rgb);
        
        QuadricPlot & qp = handles.qp_features[i];
        quadricPlot_update(qp, murPNn, SrPNn);
        quadricPlot_getActor(qp)->GetProperty()->SetOpacity(0.5);
        quadricPlot_getActor(qp)->GetProperty()->SetColor(r,g,b);
        bounds_setExtremity(qp.bounds, globalBounds); 

    }



    axisPlot_update     (handles.ap, globalBounds);
    basisPlot_update    (handles.bp, eta);
    frustumPlot_update  (handles.fp, eta);
    imagePlot_update    (handles.ip, outView);
    
    handles.threeDimRenderer->GetActiveCamera()->Azimuth(0);
    handles.threeDimRenderer->GetActiveCamera()->Elevation(165);

    handles.threeDimRenderer->GetActiveCamera()->SetFocalPoint(0,0,0);

    double sc = 2;
    handles.threeDimRenderer->GetActiveCamera()->SetPosition(-0.75*sc,-0.75*sc,-0.5*sc);
    handles.threeDimRenderer->GetActiveCamera()->SetViewUp(0,0,-1);

    handles.renderWindow->Render();
    handles.renderWindow->SetWindowName("Confidence ellipses");

}


void plotFeatureGaussianConfidenceEllipse(cv::Mat & img, const Eigen::VectorXd & murPNn, const Eigen::MatrixXd & SrPNn, const Eigen::VectorXd & eta, const CameraParameters & param, const Eigen::Vector3d & color){
    
    const int nx  = 3;
    assert(murPNn.rows() == nx);
    assert(SrPNn.rows() == nx);
    assert(SrPNn.cols() == nx);

    // Create function handle for affine transform
    WorldToPixelAdaptor w2p;
    auto h  = std::bind(w2p, std::placeholders::_1, eta, param, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);

    // ---------------------------------------------------
    // TODO
    // ---------------------------------------------------

    // Calculate and draw the 3 sigma confidence region of 
    // (murPNn, SrPNn) onto the image as an ellipse parametrised 
    // by (murQOi, SrQOi)
    Eigen::VectorXd  muimage;
    Eigen::MatrixXd Skimage;
    Eigen::VectorXd rQOi;
    cv::Point point;
    int flag = worldToPixel(murPNn, eta, param, rQOi);
    std::cout << "flag " << flag << std::endl;
    if(flag == 0) {
        cv::drawMarker(img, cv::Point(rQOi(0),rQOi(1)),cv::Scalar(color(2),color(1),color(0)),cv::MARKER_CROSS,24,2);
        affineTransform(murPNn, SrPNn, h, muimage, Skimage);
        Eigen::MatrixXd x;
        gaussianConfidenceEllipse3Sigma(muimage,Skimage,x);
        for(int i = 0; i < x.cols() - 1; i++) {
            if(x(0,i) > 0 && x(1,i) > 0 && x(0,i+1) > 0 && x(1,i+1) > 0 && x(0,i) < 1920 && x(1,i) < 1920 && x(0,i+1) < 1920 && x(1,i+1) < 1920) {
                cv::line(img, cv::Point(x(0,i),x(1,i)) ,cv::Point(x(0,i+1),x(1,i+1)),cv::Scalar(color(2),color(1),color(0)),2);
            }
        }
    }

    // ---------------------------------------------------
    // 
    
}



void WriteImage(std::string const& fileName, vtkRenderWindow* renWin, bool rgba)
{
    if (!fileName.empty())
    {
        std::string fn = fileName;
        std::string ext;
        auto found = fn.find_last_of(".");
        if (found == std::string::npos)
        {
            ext = ".png";
            fn += ext;
        }
        else
        {
            ext = fileName.substr(found, fileName.size());
        }
        std::locale loc;
        std::transform(ext.begin(), ext.end(), ext.begin(),
           [=](char const& c) { return std::tolower(c, loc); });
        auto writer = vtkSmartPointer<vtkImageWriter>::New();
        if (ext == ".bmp")
        {
            writer = vtkSmartPointer<vtkBMPWriter>::New();
        }
        else if (ext == ".jpg")
        {
            writer = vtkSmartPointer<vtkJPEGWriter>::New();
        }
        else if (ext == ".pnm")
        {
            writer = vtkSmartPointer<vtkPNMWriter>::New();
        }
        else if (ext == ".ps")
        {
            if (rgba)
            {
                rgba = false;
            }
            writer = vtkSmartPointer<vtkPostScriptWriter>::New();
        }
        else if (ext == ".tiff")
        {
            writer = vtkSmartPointer<vtkTIFFWriter>::New();
        }
        else
        {
            writer = vtkSmartPointer<vtkPNGWriter>::New();
        }

        vtkNew<vtkWindowToImageFilter> window_to_image_filter;
        window_to_image_filter->SetInput(renWin);
        window_to_image_filter->SetScale(1); // image quality
        if (rgba)
        {
            window_to_image_filter->SetInputBufferTypeToRGBA();
        }
        else
        {
            window_to_image_filter->SetInputBufferTypeToRGB();
        }
        // Read from the front buffer.
        window_to_image_filter->ReadFrontBufferOff();
        window_to_image_filter->Update();

        writer->SetFileName(fn.c_str());
        writer->SetInputConnection(window_to_image_filter->GetOutputPort());
        writer->Write();
    }
    else
    {
        std::cerr << "No filename provided." << std::endl;
    }

}
