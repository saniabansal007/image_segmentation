#include "OrthoSliceView.hpp"

#include <vtkObjectFactory.h>
#include <vtkRenderWindowInteractor.h>
#include <algorithm>
#include <cmath>

namespace ctseg {

vtkStandardNewMacro(OrthoSliceInteractorStyle);

void OrthoSliceInteractorStyle::OnLeftButtonDown() {
    if (!view_) {
        Superclass::OnLeftButtonDown();
        return;
    }
    int* pos = this->Interactor->GetEventPosition();
    vtkRenderer* ren = this->CurrentRenderer;
    if (!ren) ren = this->Interactor->FindPokedRenderer(pos[0], pos[1]);
    if (!ren) {
        Superclass::OnLeftButtonDown();
        return;
    }

    auto picker = vtkSmartPointer<vtkCellPicker>::New();
    picker->SetTolerance(0.005);
    if (picker->Pick(pos[0], pos[1], 0, ren)) {
        double world[3];
        picker->GetPickPosition(world);
        view_->handleVoxelClicked(world);
    } else {
        Superclass::OnLeftButtonDown();
    }
}

void OrthoSliceInteractorStyle::OnMouseMove() {
    if (view_) {
        int* pos = this->Interactor->GetEventPosition();
        vtkRenderer* ren = this->CurrentRenderer;
        if (!ren) ren = this->Interactor->FindPokedRenderer(pos[0], pos[1]);
        if (ren) {
            auto picker = vtkSmartPointer<vtkCellPicker>::New();
            picker->SetTolerance(0.005);
            if (picker->Pick(pos[0], pos[1], 0, ren)) {
                double world[3];
                picker->GetPickPosition(world);
                view_->handleVoxelHovered(world);
            }
        }
    }
    Superclass::OnMouseMove();
}

void OrthoSliceInteractorStyle::OnMouseWheelForward() {
    if (view_) view_->stepSlice(+1);
}

void OrthoSliceInteractorStyle::OnMouseWheelBackward() {
    if (view_) view_->stepSlice(-1);
}

// -------------------------------------------------------------------------
// OrthoSliceView Implementation
// -------------------------------------------------------------------------

OrthoSliceView::OrthoSliceView(int orientation, const QString& title, QWidget* parent)
    : QWidget(parent), orientation_(orientation), title_(title) {
    setupUI();
    setupVTKPipeline();
}

void OrthoSliceView::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(4, 4, 4, 4);
    mainLayout->setSpacing(2);

    // Top Header Bar
    auto* topBar = new QWidget(this);
    auto* topLayout = new QHBoxLayout(topBar);
    topLayout->setContentsMargins(4, 2, 4, 2);
    topLayout->setSpacing(6);

    titleLabel_ = new QLabel(title_, topBar);
    titleLabel_->setStyleSheet("font-weight: bold; font-size: 13px; color: #64B5F6;");

    sliceBadge_ = new QLabel("Slice: - / -", topBar);
    sliceBadge_->setStyleSheet("background-color: #263238; color: #CFD8DC; padding: 2px 8px; border-radius: 4px; font-size: 11px;");

    maxButton_ = new QToolButton(topBar);
    maxButton_->setText("⛶");
    maxButton_->setToolTip("Maximize or restore this view");
    maxButton_->setStyleSheet("QToolButton { background-color: #37474F; color: #ECEFF1; border-radius: 3px; padding: 2px 6px; } QToolButton:hover { background-color: #455A64; }");
    connect(maxButton_, &QToolButton::clicked, this, &OrthoSliceView::onMaximizeButtonClicked);

    topLayout->addWidget(titleLabel_);
    topLayout->addStretch(1);
    topLayout->addWidget(sliceBadge_);
    topLayout->addWidget(maxButton_);
    mainLayout->addWidget(topBar);

    // VTK Rendering Widget
    vtkTileWidget_ = new QVTKOpenGLNativeWidget(this);
    vtkTileWidget_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    vtkTileWidget_->setStyleSheet("border: 1px solid #37474F; background-color: #0d1117;");
    mainLayout->addWidget(vtkTileWidget_, 1);

    // Bottom Navigation Bar
    auto* botBar = new QWidget(this);
    auto* botLayout = new QHBoxLayout(botBar);
    botLayout->setContentsMargins(2, 2, 2, 2);
    botLayout->setSpacing(6);

    prevButton_ = new QToolButton(botBar);
    prevButton_->setText("◀");
    prevButton_->setToolTip("Previous Slice");
    prevButton_->setStyleSheet("QToolButton { background-color: #263238; color: #B0BEC5; border-radius: 3px; padding: 2px 6px; } QToolButton:hover { background-color: #37474F; color: #FFFFFF; }");
    connect(prevButton_, &QToolButton::clicked, this, &OrthoSliceView::onPrevSliceClicked);

    sliceSlider_ = new QSlider(Qt::Horizontal, botBar);
    sliceSlider_->setRange(0, 0);
    sliceSlider_->setStyleSheet(
        "QSlider::groove:horizontal { height: 4px; background: #37474F; border-radius: 2px; }"
        "QSlider::sub-page:horizontal { background: #29B6F6; border-radius: 2px; }"
        "QSlider::handle:horizontal { background: #E0E0E0; width: 12px; margin-top: -4px; margin-bottom: -4px; border-radius: 6px; }"
        "QSlider::handle:horizontal:hover { background: #FFFFFF; }"
    );
    connect(sliceSlider_, &QSlider::valueChanged, this, &OrthoSliceView::onSliderValueChanged);

    nextButton_ = new QToolButton(botBar);
    nextButton_->setText("▶");
    nextButton_->setToolTip("Next Slice");
    nextButton_->setStyleSheet("QToolButton { background-color: #263238; color: #B0BEC5; border-radius: 3px; padding: 2px 6px; } QToolButton:hover { background-color: #37474F; color: #FFFFFF; }");
    connect(nextButton_, &QToolButton::clicked, this, &OrthoSliceView::onNextSliceClicked);

    sliceSpinBox_ = new QSpinBox(botBar);
    sliceSpinBox_->setRange(0, 0);
    sliceSpinBox_->setFixedWidth(65);
    sliceSpinBox_->setStyleSheet("QSpinBox { background-color: #1E293B; color: #E2E8F0; border: 1px solid #475569; border-radius: 3px; padding: 1px 4px; }");
    connect(sliceSpinBox_, QOverload<int>::of(&QSpinBox::valueChanged), this, &OrthoSliceView::onSpinBoxValueChanged);

    botLayout->addWidget(prevButton_);
    botLayout->addWidget(sliceSlider_, 1);
    botLayout->addWidget(nextButton_);
    botLayout->addWidget(sliceSpinBox_);
    mainLayout->addWidget(botBar);
}

void OrthoSliceView::setupVTKPipeline() {
    renWin_ = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    vtkTileWidget_->setRenderWindow(renWin_);

    renderer_ = vtkSmartPointer<vtkRenderer>::New();
    renderer_->SetBackground(0.06, 0.07, 0.09); // Dark medical slate
    renWin_->AddRenderer(renderer_);

    style_ = vtkSmartPointer<OrthoSliceInteractorStyle>::New();
    style_->SetView(this);
    vtkTileWidget_->interactor()->SetInteractorStyle(style_);

    picker_ = vtkSmartPointer<vtkCellPicker>::New();
    picker_->SetTolerance(0.005);

    // Grayscale CT slice actor
    grayMapper_ = vtkSmartPointer<vtkImageSliceMapper>::New();
    grayMapper_->SetOrientation(orientation_);
    grayActor_ = vtkSmartPointer<vtkImageActor>::New();
    grayActor_->SetMapper(grayMapper_);
    renderer_->AddActor(grayActor_);

    // Translucent mask fill actor
    maskFillMapper_ = vtkSmartPointer<vtkImageSliceMapper>::New();
    maskFillMapper_->SetOrientation(orientation_);
    maskFillActor_ = vtkSmartPointer<vtkImageActor>::New();
    maskFillActor_->SetMapper(maskFillMapper_);

    maskLut_ = vtkSmartPointer<vtkLookupTable>::New();
    maskLut_->SetNumberOfTableValues(2);
    maskLut_->SetTableValue(0, 0.0, 0.0, 0.0, 0.0); // label 0 = completely transparent
    maskLut_->SetTableValue(1, fillColor_[0], fillColor_[1], fillColor_[2], fillOpacity_);
    maskLut_->SetRange(0, 1);
    maskLut_->Build();

    maskFillActor_->GetProperty()->SetLookupTable(maskLut_);
    maskFillActor_->GetProperty()->UseLookupTableScalarRangeOn();
    maskFillActor_->GetProperty()->SetOpacity(fillOpacity_);
    renderer_->AddActor(maskFillActor_);

    // High-contrast ROI boundary contour actor (Marching Squares pipeline)
    extractVOI_ = vtkSmartPointer<vtkExtractVOI>::New();
    marchingSquares_ = vtkSmartPointer<vtkMarchingSquares>::New();
    marchingSquares_->SetInputConnection(extractVOI_->GetOutputPort());
    marchingSquares_->SetValue(0, 0.5); // iso-contour separating label 0 and label 1

    contourMapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
    contourMapper_->SetInputConnection(marchingSquares_->GetOutputPort());
    contourMapper_->ScalarVisibilityOff();
    // Resolve coincident topology to ensure contour lines sit crisp on top of image
    contourMapper_->SetResolveCoincidentTopologyToPolygonOffset();
    contourMapper_->SetResolveCoincidentTopologyPolygonOffsetParameters(2.0, 2.0);

    contourActor_ = vtkSmartPointer<vtkActor>::New();
    contourActor_->SetMapper(contourMapper_);
    contourActor_->GetProperty()->SetColor(boundaryColor_[0], boundaryColor_[1], boundaryColor_[2]);
    contourActor_->GetProperty()->SetLineWidth(boundaryWidth_);
    contourActor_->GetProperty()->SetAmbient(1.0); // Pure emissive color for maximum contrast
    contourActor_->GetProperty()->SetDiffuse(0.0);
    contourActor_->GetProperty()->SetSpecular(0.0);
    contourActor_->SetVisibility(false);
    renderer_->AddActor(contourActor_);

    // Crosshairs
    crosshairHLine_ = vtkSmartPointer<vtkLineSource>::New();
    crosshairHMapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
    crosshairHMapper_->SetInputConnection(crosshairHLine_->GetOutputPort());
    crosshairHActor_ = vtkSmartPointer<vtkActor>::New();
    crosshairHActor_->SetMapper(crosshairHMapper_);
    crosshairHActor_->GetProperty()->SetColor(0.2, 0.8, 1.0); // Cyan
    crosshairHActor_->GetProperty()->SetLineWidth(1.2f);
    crosshairHActor_->GetProperty()->SetOpacity(0.65);
    renderer_->AddActor(crosshairHActor_);

    crosshairVLine_ = vtkSmartPointer<vtkLineSource>::New();
    crosshairVMapper_ = vtkSmartPointer<vtkPolyDataMapper>::New();
    crosshairVMapper_->SetInputConnection(crosshairVLine_->GetOutputPort());
    crosshairVActor_ = vtkSmartPointer<vtkActor>::New();
    crosshairVActor_->SetMapper(crosshairVMapper_);
    crosshairVActor_->GetProperty()->SetColor(0.2, 0.8, 1.0);
    crosshairVActor_->GetProperty()->SetLineWidth(1.2f);
    crosshairVActor_->GetProperty()->SetOpacity(0.65);
    renderer_->AddActor(crosshairVActor_);

    // Configure camera for orthogonal view
    vtkCamera* cam = renderer_->GetActiveCamera();
    cam->ParallelProjectionOn();
    switch (orientation_) {
        case 2: // Axial (perpendicular to Z)
            cam->SetPosition(0, 0, -1);
            cam->SetFocalPoint(0, 0, 0);
            cam->SetViewUp(0, -1, 0);
            break;
        case 1: // Coronal (perpendicular to Y)
            cam->SetPosition(0, -1, 0);
            cam->SetFocalPoint(0, 0, 0);
            cam->SetViewUp(0, 0, 1);
            break;
        case 0: // Sagittal (perpendicular to X)
            cam->SetPosition(1, 0, 0);
            cam->SetFocalPoint(0, 0, 0);
            cam->SetViewUp(0, 0, 1);
            break;
    }
}

void OrthoSliceView::initGeometry(int dimX, int dimY, int dimZ, const double spacing[3], const double origin[3]) {
    isLazyMode_ = true;
    dimX_ = dimX;
    dimY_ = dimY;
    dimZ_ = dimZ;
    for (int i = 0; i < 3; ++i) {
        spacing_[i] = spacing[i];
        origin_[i] = origin[i];
    }
    fullBounds_[0] = origin_[0];
    fullBounds_[1] = origin_[0] + (dimX_ - 1) * spacing_[0];
    fullBounds_[2] = origin_[1];
    fullBounds_[3] = origin_[1] + (dimY_ - 1) * spacing_[1];
    fullBounds_[4] = origin_[2];
    fullBounds_[5] = origin_[2] + (dimZ_ - 1) * spacing_[2];

    minSliceIndex_ = 0;
    if (orientation_ == 2) maxSliceIndex_ = dimZ_ - 1;
    else if (orientation_ == 1) maxSliceIndex_ = dimY_ - 1;
    else maxSliceIndex_ = dimX_ - 1;

    sliceSlider_->blockSignals(true);
    sliceSpinBox_->blockSignals(true);
    sliceSlider_->setRange(minSliceIndex_, maxSliceIndex_);
    sliceSpinBox_->setRange(minSliceIndex_, maxSliceIndex_);
    currentSliceIndex_ = (minSliceIndex_ + maxSliceIndex_) / 2;
    sliceSlider_->setValue(currentSliceIndex_);
    sliceSpinBox_->setValue(currentSliceIndex_);
    sliceSlider_->blockSignals(false);
    sliceSpinBox_->blockSignals(false);

    // Create a 2D vtkImageData buffer for this view tile
    if (!grayData_) {
        grayData_ = vtkSmartPointer<vtkImageData>::New();
    }
    grayData_->SetSpacing(spacing_[0], spacing_[1], spacing_[2]);
    grayData_->SetOrigin(origin_[0], origin_[1], origin_[2]);

    int ext[6] = {0, 0, 0, 0, 0, 0};
    if (orientation_ == 2) {
        ext[0] = 0; ext[1] = dimX_ - 1;
        ext[2] = 0; ext[3] = dimY_ - 1;
        ext[4] = currentSliceIndex_; ext[5] = currentSliceIndex_;
    } else if (orientation_ == 1) {
        ext[0] = 0; ext[1] = dimX_ - 1;
        ext[2] = currentSliceIndex_; ext[3] = currentSliceIndex_;
        ext[4] = 0; ext[5] = dimZ_ - 1;
    } else {
        ext[0] = currentSliceIndex_; ext[1] = currentSliceIndex_;
        ext[2] = 0; ext[3] = dimY_ - 1;
        ext[4] = 0; ext[5] = dimZ_ - 1;
    }
    grayData_->SetExtent(ext);
    grayData_->AllocateScalars(VTK_UNSIGNED_SHORT, 1);
    std::memset(grayData_->GetScalarPointer(), 0, grayData_->GetNumberOfPoints() * sizeof(uint16_t));
    grayData_->Modified();

    grayMapper_->SetInputData(grayData_);
    grayMapper_->SetOrientation(orientation_);
    grayMapper_->SetSliceNumber(currentSliceIndex_);

    resetCamera();
    updateBadgeText();
    updateCrosshairs();
    render();
}

void OrthoSliceView::setSliceData2D(const uint16_t* data, int width, int height, int sliceIndex) {
    if (!grayData_ || !data) return;

    currentSliceIndex_ = std::max(minSliceIndex_, std::min(maxSliceIndex_, sliceIndex));

    int ext[6] = {0, 0, 0, 0, 0, 0};
    if (orientation_ == 2) {
        ext[0] = 0; ext[1] = dimX_ - 1;
        ext[2] = 0; ext[3] = dimY_ - 1;
        ext[4] = currentSliceIndex_; ext[5] = currentSliceIndex_;
    } else if (orientation_ == 1) {
        ext[0] = 0; ext[1] = dimX_ - 1;
        ext[2] = currentSliceIndex_; ext[3] = currentSliceIndex_;
        ext[4] = 0; ext[5] = dimZ_ - 1;
    } else {
        ext[0] = currentSliceIndex_; ext[1] = currentSliceIndex_;
        ext[2] = 0; ext[3] = dimY_ - 1;
        ext[4] = 0; ext[5] = dimZ_ - 1;
    }
    grayData_->SetExtent(ext);

    void* ptr = grayData_->GetScalarPointer();
    if (ptr) {
        size_t numPixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        std::memcpy(ptr, data, numPixels * sizeof(uint16_t));
    }
    grayData_->Modified();
    grayMapper_->SetSliceNumber(currentSliceIndex_);
    maskFillMapper_->SetSliceNumber(currentSliceIndex_);

    sliceSlider_->blockSignals(true);
    sliceSpinBox_->blockSignals(true);
    sliceSlider_->setValue(currentSliceIndex_);
    sliceSpinBox_->setValue(currentSliceIndex_);
    sliceSlider_->blockSignals(false);
    sliceSpinBox_->blockSignals(false);

    updateBadgeText();
    updateCrosshairs();
    updateContour();
    render();
}

void OrthoSliceView::setVolumeData(vtkImageData* grayData, VolumeType::Pointer itkVol) {
    isLazyMode_ = false;
    grayData_ = grayData;
    itkVolume_ = itkVol;

    grayMapper_->SetInputData(grayData_);
    int minS = grayMapper_->GetSliceNumberMinValue();
    int maxS = grayMapper_->GetSliceNumberMaxValue();

    sliceSlider_->blockSignals(true);
    sliceSpinBox_->blockSignals(true);
    sliceSlider_->setRange(minS, maxS);
    sliceSpinBox_->setRange(minS, maxS);

    int cur = (currentSliceIndex_ >= minS && currentSliceIndex_ <= maxS) ? currentSliceIndex_ : ((minS + maxS) / 2);
    sliceSlider_->setValue(cur);
    sliceSpinBox_->setValue(cur);
    grayMapper_->SetSliceNumber(cur);
    maskFillMapper_->SetSliceNumber(cur);
    currentSliceIndex_ = cur;

    sliceSlider_->blockSignals(false);
    sliceSpinBox_->blockSignals(false);

    resetCamera();
    updateBadgeText();
    updateCrosshairs();
    updateContour();
    render();
}

void OrthoSliceView::setMaskData(vtkImageData* maskData, LabelVolumeType::Pointer itkMask) {
    maskData_ = maskData;
    itkMask_ = itkMask;

    maskFillMapper_->SetInputData(maskData_);
    extractVOI_->SetInputData(maskData_);

    if (maskData_) {
        maskFillMapper_->SetSliceNumber(currentSlice());
    }

    updateContour();
    render();
}

void OrthoSliceView::setSlice(int sliceIndex) {
    int minS = minSlice();
    int maxS = maxSlice();
    int s = std::max(minS, std::min(maxS, sliceIndex));

    if (currentSlice() == s && (!isLazyMode_ || currentSliceIndex_ == s)) return;

    currentSliceIndex_ = s;

    sliceSlider_->blockSignals(true);
    sliceSpinBox_->blockSignals(true);
    sliceSlider_->setValue(s);
    sliceSpinBox_->setValue(s);
    sliceSlider_->blockSignals(false);
    sliceSpinBox_->blockSignals(false);

    if (!isLazyMode_) {
        grayMapper_->SetSliceNumber(s);
        maskFillMapper_->SetSliceNumber(s);
        updateBadgeText();
        updateCrosshairs();
        updateContour();
        render();
    } else {
        maskFillMapper_->SetSliceNumber(s);
        updateBadgeText();
        updateCrosshairs();
        updateContour();
        render();
    }
}

int OrthoSliceView::currentSlice() const {
    if (isLazyMode_) return currentSliceIndex_;
    return grayMapper_->GetSliceNumber();
}

int OrthoSliceView::minSlice() const {
    if (isLazyMode_) return minSliceIndex_;
    return grayMapper_->GetSliceNumberMinValue();
}

int OrthoSliceView::maxSlice() const {
    if (isLazyMode_) return maxSliceIndex_;
    return grayMapper_->GetSliceNumberMaxValue();
}

void OrthoSliceView::setWindowLevel(double window, double level) {
    grayActor_->GetProperty()->SetColorWindow(window);
    grayActor_->GetProperty()->SetColorLevel(level);
    render();
}

void OrthoSliceView::setCrosshairPosition(int x, int y, int z) {
    cursorCoords_[0] = x;
    cursorCoords_[1] = y;
    cursorCoords_[2] = z;

    updateCrosshairs();
    render();
}

void OrthoSliceView::setCrosshairVisible(bool visible) {
    crosshairVisible_ = visible;
    crosshairHActor_->SetVisibility(visible);
    crosshairVActor_->SetVisibility(visible);
    render();
}

void OrthoSliceView::setBoundaryVisible(bool visible) {
    boundaryVisible_ = visible;
    updateContour();
    render();
}

void OrthoSliceView::setBoundaryColor(double r, double g, double b) {
    boundaryColor_[0] = r;
    boundaryColor_[1] = g;
    boundaryColor_[2] = b;
    contourActor_->GetProperty()->SetColor(r, g, b);
    render();
}

void OrthoSliceView::setBoundaryLineWidth(float width) {
    boundaryWidth_ = width;
    contourActor_->GetProperty()->SetLineWidth(width);
    render();
}

void OrthoSliceView::setFillVisible(bool visible) {
    fillVisible_ = visible;
    maskFillActor_->SetVisibility(visible);
    render();
}

void OrthoSliceView::setFillOpacity(double opacity) {
    fillOpacity_ = opacity;
    maskFillActor_->GetProperty()->SetOpacity(opacity);
    maskLut_->SetTableValue(1, fillColor_[0], fillColor_[1], fillColor_[2], opacity);
    maskLut_->Build();
    render();
}

void OrthoSliceView::setFillColor(double r, double g, double b) {
    fillColor_[0] = r;
    fillColor_[1] = g;
    fillColor_[2] = b;
    maskLut_->SetTableValue(1, r, g, b, fillOpacity_);
    maskLut_->Build();
    render();
}

void OrthoSliceView::resetCamera() {
    if (isLazyMode_) {
        renderer_->ResetCamera(fullBounds_);
        render();
        return;
    }
    if (!grayData_) return;
    double bounds[6];
    grayData_->GetBounds(bounds);
    renderer_->ResetCamera(bounds);
    render();
}

void OrthoSliceView::render() {
    if (renWin_) renWin_->Render();
}

void OrthoSliceView::updateContour() {
    if (!maskData_ || !boundaryVisible_) {
        contourActor_->SetVisibility(false);
        return;
    }

    int s = currentSlice();
    int extent[6];
    maskData_->GetExtent(extent);

    int voi[6] = {extent[0], extent[1], extent[2], extent[3], extent[4], extent[5]};
    if (orientation_ == 2) {
        voi[4] = s; voi[5] = s;
    } else if (orientation_ == 1) {
        voi[2] = s; voi[3] = s;
    } else {
        voi[0] = s; voi[1] = s;
    }

    extractVOI_->SetVOI(voi);
    extractVOI_->Update();
    marchingSquares_->Update();

    bool hasLines = (marchingSquares_->GetOutput() && marchingSquares_->GetOutput()->GetNumberOfLines() > 0);
    contourActor_->SetVisibility(boundaryVisible_ && hasLines);
}

void OrthoSliceView::updateCrosshairs() {
    if (!grayData_ || !crosshairVisible_) {
        crosshairHActor_->SetVisibility(false);
        crosshairVActor_->SetVisibility(false);
        return;
    }

    double bounds[6];
    double spacing[3];
    double origin[3];

    if (isLazyMode_) {
        for (int i = 0; i < 6; ++i) bounds[i] = fullBounds_[i];
        for (int i = 0; i < 3; ++i) {
            spacing[i] = spacing_[i];
            origin[i] = origin_[i];
        }
    } else {
        grayData_->GetBounds(bounds);
        grayData_->GetSpacing(spacing);
        grayData_->GetOrigin(origin);
    }

    double wx = origin[0] + cursorCoords_[0] * spacing[0];
    double wy = origin[1] + cursorCoords_[1] * spacing[1];
    double wz = origin[2] + cursorCoords_[2] * spacing[2];

    int s = currentSlice();

    if (orientation_ == 2) { // Axial (plane Z = currentSlice)
        double currentZ = origin[2] + s * spacing[2];
        crosshairHLine_->SetPoint1(bounds[0], wy, currentZ);
        crosshairHLine_->SetPoint2(bounds[1], wy, currentZ);
        crosshairVLine_->SetPoint1(wx, bounds[2], currentZ);
        crosshairVLine_->SetPoint2(wx, bounds[3], currentZ);
    } else if (orientation_ == 1) { // Coronal (plane Y = currentSlice)
        double currentY = origin[1] + s * spacing[1];
        crosshairHLine_->SetPoint1(bounds[0], currentY, wz);
        crosshairHLine_->SetPoint2(bounds[1], currentY, wz);
        crosshairVLine_->SetPoint1(wx, currentY, bounds[4]);
        crosshairVLine_->SetPoint2(wx, currentY, bounds[5]);
    } else { // Sagittal (plane X = currentSlice)
        double currentX = origin[0] + s * spacing[0];
        crosshairHLine_->SetPoint1(currentX, bounds[2], wz);
        crosshairHLine_->SetPoint2(currentX, bounds[3], wz);
        crosshairVLine_->SetPoint1(currentX, wy, bounds[4]);
        crosshairVLine_->SetPoint2(currentX, wy, bounds[5]);
    }

    crosshairHLine_->Update();
    crosshairVLine_->Update();
    crosshairHActor_->SetVisibility(crosshairVisible_);
    crosshairVActor_->SetVisibility(crosshairVisible_);
}

void OrthoSliceView::updateBadgeText() {
    sliceBadge_->setText(QString("Slice %1 / %2").arg(currentSlice()).arg(maxSlice()));
}

bool OrthoSliceView::worldToIndex(const double world[3], VolumeType::IndexType& idx) const {
    if (itkVolume_) {
        VolumeType::PointType pt;
        pt[0] = world[0]; pt[1] = world[1]; pt[2] = world[2];
        return itkVolume_->TransformPhysicalPointToIndex(pt, idx);
    }
    if (dimX_ > 0 && spacing_[0] > 0 && spacing_[1] > 0 && spacing_[2] > 0) {
        long long ix = std::lround((world[0] - origin_[0]) / spacing_[0]);
        long long iy = std::lround((world[1] - origin_[1]) / spacing_[1]);
        long long iz = std::lround((world[2] - origin_[2]) / spacing_[2]);
        if (ix < 0 || ix >= dimX_ || iy < 0 || iy >= dimY_ || iz < 0 || iz >= dimZ_) return false;
        idx[0] = ix; idx[1] = iy; idx[2] = iz;
        return true;
    }
    return false;
}

void OrthoSliceView::handleVoxelClicked(const double worldPos[3]) {
    VolumeType::IndexType idx;
    if (worldToIndex(worldPos, idx)) {
        emit voxelClicked(static_cast<int>(idx[0]), static_cast<int>(idx[1]), static_cast<int>(idx[2]));
    }
}

void OrthoSliceView::handleVoxelHovered(const double worldPos[3]) {
    VolumeType::IndexType idx;
    if (worldToIndex(worldPos, idx)) {
        double val = 0.0;
        if (itkVolume_ && itkVolume_->GetLargestPossibleRegion().IsInside(idx)) {
            val = static_cast<double>(itkVolume_->GetPixel(idx));
        }
        emit voxelHovered(static_cast<int>(idx[0]), static_cast<int>(idx[1]), static_cast<int>(idx[2]), val);
    }
}

void OrthoSliceView::stepSlice(int delta) {
    setSlice(currentSlice() + delta);
    emit sliceChanged(orientation_, currentSlice());
}

void OrthoSliceView::onSliderValueChanged(int val) {
    setSlice(val);
    emit sliceChanged(orientation_, val);
}

void OrthoSliceView::onSpinBoxValueChanged(int val) {
    setSlice(val);
    emit sliceChanged(orientation_, val);
}

void OrthoSliceView::onPrevSliceClicked() {
    stepSlice(-1);
}

void OrthoSliceView::onNextSliceClicked() {
    stepSlice(+1);
}

void OrthoSliceView::onMaximizeButtonClicked() {
    isMaximized_ = !isMaximized_;
    maxButton_->setText(isMaximized_ ? "❐" : "⛶");
    emit toggleMaximizeRequested(orientation_);
}

} // namespace ctseg
