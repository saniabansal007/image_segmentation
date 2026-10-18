#pragma once

#include "../volume/VolumeTypes.hpp"

#include <QWidget>
#include <QLabel>
#include <QSlider>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include <QVTKOpenGLNativeWidget.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkImageSliceMapper.h>
#include <vtkImageActor.h>
#include <vtkImageProperty.h>
#include <vtkLookupTable.h>
#include <vtkExtractVOI.h>
#include <vtkMarchingSquares.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkLineSource.h>
#include <vtkCamera.h>
#include <vtkCellPicker.h>
#include <vtkInteractorStyleImage.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkTextActor.h>

namespace ctseg {

class OrthoSliceView;

// Custom interactor style for an orthogonal CT slice tile
class OrthoSliceInteractorStyle : public vtkInteractorStyleImage {
public:
    static OrthoSliceInteractorStyle* New();
    vtkTypeMacro(OrthoSliceInteractorStyle, vtkInteractorStyleImage);

    void SetView(OrthoSliceView* view) { view_ = view; }

    void OnLeftButtonDown() override;
    void OnMouseMove() override;
    void OnMouseWheelForward() override;
    void OnMouseWheelBackward() override;
    void OnChar() override {} // swallow single-key bindings

private:
    OrthoSliceView* view_ = nullptr;
};

class OrthoSliceView : public QWidget {
    Q_OBJECT
public:
    // orientation: 2 = Axial (Z), 1 = Coronal (Y), 0 = Sagittal (X)
    explicit OrthoSliceView(int orientation, const QString& title, QWidget* parent = nullptr);
    ~OrthoSliceView() override = default;

    int orientation() const { return orientation_; }
    const QString& title() const { return title_; }

    void setVolumeData(vtkImageData* grayData, VolumeType::Pointer itkVol);
    void setMaskData(vtkImageData* maskData, LabelVolumeType::Pointer itkMask);

    void setSlice(int sliceIndex);
    int currentSlice() const;
    int minSlice() const;
    int maxSlice() const;

    void setWindowLevel(double window, double level);
    void setCrosshairPosition(int x, int y, int z);
    void setCrosshairVisible(bool visible);

    // CRITICAL: ROI Contour & Fill visibility and properties
    void setBoundaryVisible(bool visible);
    void setBoundaryColor(double r, double g, double b);
    void setBoundaryLineWidth(float width);

    void setFillVisible(bool visible);
    void setFillOpacity(double opacity);
    void setFillColor(double r, double g, double b);

    void resetCamera();
    void render();
    void updateContour();

    // Internal handlers called by interactor style
    void handleVoxelClicked(const double worldPos[3]);
    void handleVoxelHovered(const double worldPos[3]);
    void stepSlice(int delta);

signals:
    void sliceChanged(int orientation, int sliceIndex);
    void voxelClicked(int x, int y, int z);
    void voxelHovered(int x, int y, int z, double intensity);
    void toggleMaximizeRequested(int orientation);

private slots:
    void onSliderValueChanged(int val);
    void onSpinBoxValueChanged(int val);
    void onPrevSliceClicked();
    void onNextSliceClicked();
    void onMaximizeButtonClicked();

private:
    void setupUI();
    void setupVTKPipeline();
    void updateCrosshairs();
    void updateBadgeText();
    bool worldToIndex(const double world[3], VolumeType::IndexType& idx) const;

    int orientation_ = 2;
    QString title_;
    bool isMaximized_ = false;

    // ITK volume references
    VolumeType::Pointer itkVolume_;
    LabelVolumeType::Pointer itkMask_;

    // VTK image references
    vtkSmartPointer<vtkImageData> grayData_;
    vtkSmartPointer<vtkImageData> maskData_;

    // Widgets
    QVTKOpenGLNativeWidget* vtkTileWidget_ = nullptr;
    QLabel* titleLabel_ = nullptr;
    QLabel* sliceBadge_ = nullptr;
    QToolButton* maxButton_ = nullptr;
    QSlider* sliceSlider_ = nullptr;
    QSpinBox* sliceSpinBox_ = nullptr;
    QToolButton* prevButton_ = nullptr;
    QToolButton* nextButton_ = nullptr;

    // VTK Pipeline
    vtkSmartPointer<vtkGenericOpenGLRenderWindow> renWin_;
    vtkSmartPointer<vtkRenderer> renderer_;
    vtkSmartPointer<OrthoSliceInteractorStyle> style_;
    vtkSmartPointer<vtkCellPicker> picker_;

    // Grayscale CT slice
    vtkSmartPointer<vtkImageSliceMapper> grayMapper_;
    vtkSmartPointer<vtkImageActor> grayActor_;

    // Mask translucent fill
    vtkSmartPointer<vtkImageSliceMapper> maskFillMapper_;
    vtkSmartPointer<vtkImageActor> maskFillActor_;
    vtkSmartPointer<vtkLookupTable> maskLut_;

    // High-contrast ROI boundary contour (derived via Marching Squares on 2D slice)
    vtkSmartPointer<vtkExtractVOI> extractVOI_;
    vtkSmartPointer<vtkMarchingSquares> marchingSquares_;
    vtkSmartPointer<vtkPolyDataMapper> contourMapper_;
    vtkSmartPointer<vtkActor> contourActor_;

    // Crosshairs
    vtkSmartPointer<vtkLineSource> crosshairHLine_;
    vtkSmartPointer<vtkLineSource> crosshairVLine_;
    vtkSmartPointer<vtkPolyDataMapper> crosshairHMapper_;
    vtkSmartPointer<vtkPolyDataMapper> crosshairVMapper_;
    vtkSmartPointer<vtkActor> crosshairHActor_;
    vtkSmartPointer<vtkActor> crosshairVActor_;

    // State
    bool boundaryVisible_ = true;
    bool fillVisible_ = true;
    double fillOpacity_ = 0.35;
    double fillColor_[3] = {1.0, 0.1, 0.1}; // default red
    double boundaryColor_[3] = {0.0, 1.0, 0.0}; // default neon green
    float boundaryWidth_ = 2.5f;
    bool crosshairVisible_ = true;
    int cursorCoords_[3] = {0, 0, 0};
};

} // namespace ctseg
