#include "CTViewerWindow.hpp"
#include "../volume/VolumeIO.hpp"
#include "../io/SliceIO.hpp"
#include "../preprocessing/Preprocessing.hpp"

#if defined(HAVE_CUDA)
#include "../gpu/GPUPreprocessing.hpp"
#include "../gpu/GPUSegmentation.hpp"
#endif

#include <QApplication>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QScrollArea>
#include <QFormLayout>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QWindow>
#include <QScreen>
#include <QPixmap>
#include <QElapsedTimer>
#include <QDir>
#include <QStyle>
#include <QMouseEvent>

// ITK Filters
#include <itkMinimumMaximumImageCalculator.h>
#include <itkImageRegionConstIterator.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkMedianImageFilter.h>
#include <itkGradientAnisotropicDiffusionImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkOtsuThresholdImageFilter.h>
#include <itkOtsuMultipleThresholdsImageFilter.h>
#include <itkConfidenceConnectedImageFilter.h>
#include <itkConnectedThresholdImageFilter.h>
#include <itkScalarImageKmeansImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkTileImageFilter.h>
#include <itkOutputWindow.h>

#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace ctseg {

CTViewerWindow::CTViewerWindow(QWidget* parent) : QMainWindow(parent) {
    // Suppress redundant ITK warning popups
    itk::OutputWindow::SetGlobalWarningDisplay(false);

    setWindowTitle("CT Medical Image Segmentation & Multi-Plane Viewer [ITK / VTK / Qt]");
    resize(1600, 950);

    applyDarkTheme();
    setupUI();
    setupMenuBar();
    setupStatusBar();

    qApp->installEventFilter(this);
}

CTViewerWindow::~CTViewerWindow() {
    if (isProcessRunning_.load()) {
        stopRunningProcess();
        if (activeWorkerThread_ && activeWorkerThread_->isRunning()) {
            activeWorkerThread_->wait(3000);
        }
    }
}

void CTViewerWindow::stopRunningProcess() {
    if (!isProcessRunning_.load()) return;

    cancelRequested_.store(true);
    statusPerfLabel_->setText("Stopping running process... please wait");
    if (stopProcessBtn_) {
        stopProcessBtn_->setText("Stopping...");
        stopProcessBtn_->setEnabled(false);
    }
    if (currentProcessObject_) {
        try {
            currentProcessObject_->SetAbortGenerateData(true);
        } catch (...) {}
    }
    QApplication::processEvents();
}

void CTViewerWindow::resetProcessUI() {
    isProcessRunning_.store(false);
    cancelRequested_.store(false);
    currentProcessObject_ = nullptr;
    activeWorkerThread_ = nullptr;

    if (applyFilterBtn_) {
        applyFilterBtn_->setText("Apply Filter");
        applyFilterBtn_->setStyleSheet("background-color: #059669; font-weight: bold;");
        applyFilterBtn_->setEnabled(true);
    }
    if (applySegBtn_) {
        applySegBtn_->setText("Apply Segmentation");
        applySegBtn_->setStyleSheet("background-color: #D97706; font-weight: bold;");
        applySegBtn_->setEnabled(true);
    }
    if (stopProcessBtn_) {
        stopProcessBtn_->setText("■ Stop");
        stopProcessBtn_->setEnabled(true);
        stopProcessBtn_->setVisible(false);
    }
    if (stopProcessAct_) {
        stopProcessAct_->setEnabled(false);
    }
    if (progressBar_) {
        progressBar_->setVisible(false);
        progressBar_->setValue(0);
    }
    QApplication::restoreOverrideCursor();
}

void CTViewerWindow::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Escape && isProcessRunning_.load()) {
        stopRunningProcess();
        event->accept();
        return;
    }
    QMainWindow::keyPressEvent(event);
}

void CTViewerWindow::closeEvent(QCloseEvent* event) {
    if (isProcessRunning_.load()) {
        stopRunningProcess();
        if (activeWorkerThread_ && activeWorkerThread_->isRunning()) {
            activeWorkerThread_->wait(2000);
        }
    }
    event->accept();
}

bool CTViewerWindow::eventFilter(QObject* obj, QEvent* event) {
    if (event->type() == QEvent::KeyPress) {
        auto* keyEvent = static_cast<QKeyEvent*>(event);
        if (keyEvent->key() == Qt::Key_Escape && isProcessRunning_.load()) {
            stopRunningProcess();
            return true;
        }
    } else if (event->type() == QEvent::MouseButtonPress) {
        auto* me = static_cast<QMouseEvent*>(event);
        if (me->button() == Qt::LeftButton) {
            auto* slider = qobject_cast<QSlider*>(obj);
            if (slider) {
                int newPos = QStyle::sliderValueFromPosition(
                    slider->minimum(), slider->maximum(),
                    me->pos().x(), slider->width());
                slider->setValue(newPos);
            }
        }
    }
    return QMainWindow::eventFilter(obj, event);
}

void CTViewerWindow::applyDarkTheme() {
    setStyleSheet(
        "QMainWindow { background-color: #0F172A; }"
        "QWidget { color: #E2E8F0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 12px; }"
        "QGroupBox { font-weight: bold; font-size: 12px; border: 1px solid #334155; border-radius: 6px; margin-top: 22px; padding: 14px 8px 8px 8px; background-color: #1E293B; }"
        "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px; top: 2px; padding: 2px 8px; color: #38BDF8; background-color: #1E293B; border: 1px solid #38BDF8; border-radius: 4px; font-weight: bold; font-size: 12px; }"
        "QPushButton { background-color: #2563EB; color: #FFFFFF; font-weight: bold; border-radius: 4px; padding: 6px 12px; border: none; }"
        "QPushButton:hover { background-color: #1D4ED8; }"
        "QPushButton:pressed { background-color: #1E40AF; }"
        "QPushButton:disabled { background-color: #334155; color: #64748B; }"
        "QComboBox, QSpinBox, QDoubleSpinBox { background-color: #0F172A; border: 1px solid #475569; border-radius: 4px; padding: 4px 6px; color: #F8FAFC; selection-background-color: #2563EB; }"
        "QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #38BDF8; }"
        "QComboBox QAbstractItemView { background-color: #1E293B; color: #F8FAFC; selection-background-color: #2563EB; border: 1px solid #475569; }"
        "QSlider::groove:horizontal { height: 5px; background: #334155; border-radius: 2px; }"
        "QSlider::sub-page:horizontal { background: #38BDF8; border-radius: 2px; }"
        "QSlider::handle:horizontal { background: #F8FAFC; width: 14px; margin-top: -5px; margin-bottom: -5px; border-radius: 7px; }"
        "QCheckBox, QRadioButton { spacing: 6px; }"
        "QCheckBox::indicator, QRadioButton::indicator { width: 15px; height: 15px; }"
        "QStatusBar { background-color: #0B1120; border-top: 1px solid #1E293B; color: #94A3B8; font-size: 11px; }"
        "QMenuBar { background-color: #0F172A; border-bottom: 1px solid #1E293B; color: #E2E8F0; }"
        "QMenuBar::item:selected { background-color: #1E293B; }"
        "QMenu { background-color: #1E293B; border: 1px solid #334155; color: #E2E8F0; }"
        "QMenu::item:selected { background-color: #2563EB; color: #FFFFFF; }"
        "QScrollArea { border: none; background-color: transparent; }"
        "QSplitter::handle { background-color: #1E293B; }"
    );
}

void CTViewerWindow::setupMenuBar() {
    auto* fileMenu = menuBar()->addMenu("&File");
    auto* openDirAct = fileMenu->addAction("Open Slice &Directory (3D Volume)...");
    connect(openDirAct, &QAction::triggered, this, &CTViewerWindow::onOpenDirectoryTriggered);

    auto* openSliceAct = fileMenu->addAction("Open Single &2D Slice...");
    connect(openSliceAct, &QAction::triggered, this, &CTViewerWindow::onOpenSingleSliceTriggered);

    fileMenu->addSeparator();
    auto* exportFilteredAct = fileMenu->addAction("Export Filtered Image as TIFF...");
    connect(exportFilteredAct, &QAction::triggered, this, &CTViewerWindow::onExportFilteredClicked);

    auto* exportMaskAct = fileMenu->addAction("Export Segmentation Mask as TIFF...");
    connect(exportMaskAct, &QAction::triggered, this, &CTViewerWindow::onExportMaskClicked);

    auto* exportMaskedAct = fileMenu->addAction("Export Segmented CT (Masked ROI) as TIFF...");
    connect(exportMaskedAct, &QAction::triggered, this, &CTViewerWindow::onExportMaskedCTClicked);

    fileMenu->addSeparator();
    auto* exitAct = fileMenu->addAction("E&xit");
    connect(exitAct, &QAction::triggered, this, &QWidget::close);

    auto* viewMenu = menuBar()->addMenu("&View");
    auto* layout3Act = viewMenu->addAction("3-View Orthogonal (Axial | Coronal | Sagittal)");
    connect(layout3Act, &QAction::triggered, [this]() { onViewLayoutChanged(0); });
    auto* layoutAxialAct = viewMenu->addAction("Axial View Only");
    connect(layoutAxialAct, &QAction::triggered, [this]() { onViewLayoutChanged(1); });
    auto* layoutCoronalAct = viewMenu->addAction("Coronal View Only");
    connect(layoutCoronalAct, &QAction::triggered, [this]() { onViewLayoutChanged(2); });
    auto* layoutSagittalAct = viewMenu->addAction("Sagittal View Only");
    connect(layoutSagittalAct, &QAction::triggered, [this]() { onViewLayoutChanged(3); });

    viewMenu->addSeparator();
    auto* resetCamAct = viewMenu->addAction("Reset Cameras / Fit to Window");
    connect(resetCamAct, &QAction::triggered, this, &CTViewerWindow::onResetCamerasClicked);

    auto* processMenu = menuBar()->addMenu("&Process");
    stopProcessAct_ = processMenu->addAction("&Stop Running Process");
    stopProcessAct_->setShortcut(QKeySequence(Qt::Key_Escape));
    stopProcessAct_->setEnabled(false);
    connect(stopProcessAct_, &QAction::triggered, this, &CTViewerWindow::stopRunningProcess);

    auto* helpMenu = menuBar()->addMenu("&Help");
    auto* aboutAct = helpMenu->addAction("&About CT Segmentation Viewer");
    connect(aboutAct, &QAction::triggered, this, &CTViewerWindow::onAboutTriggered);
}

void CTViewerWindow::setupUI() {
    mainSplitter_ = new QSplitter(Qt::Horizontal, this);
    setCentralWidget(mainSplitter_);

    // Left Control Panel with scroll area
    auto* scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    scrollArea->setMinimumWidth(380);
    scrollArea->setMaximumWidth(460);

    auto* panelContent = new QWidget(scrollArea);
    setupControlPanel(panelContent);
    scrollArea->setWidget(panelContent);
    mainSplitter_->addWidget(scrollArea);

    // Right View Area
    auto* viewContainer = new QWidget(this);
    setupViewArea(viewContainer);
    mainSplitter_->addWidget(viewContainer);

    mainSplitter_->setStretchFactor(0, 0);
    mainSplitter_->setStretchFactor(1, 1);
}

void CTViewerWindow::setupControlPanel(QWidget* panel) {
    auto* layout = new QVBoxLayout(panel);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(10);

    // 1. DATASET INFO & CONTRAST (WINDOW / LEVEL)
    auto* infoGroup = new QGroupBox("Dataset & Contrast (Window / Level)", panel);
    auto* infoLayout = new QVBoxLayout(infoGroup);
    infoLayout->setSpacing(6);

    datasetInfoLabel_ = new QLabel("No dataset loaded.\nUse File -> Open to load slices.", infoGroup);
    datasetInfoLabel_->setStyleSheet("color: #94A3B8; font-size: 11px;");
    infoLayout->addWidget(datasetInfoLabel_);

    auto* presetLayout = new QHBoxLayout();
    presetLayout->addWidget(new QLabel("Preset:", infoGroup));
    presetComboBox_ = new QComboBox(infoGroup);
    presetComboBox_->addItems({"Auto (1%-99% Percentile)", "Full Range", "Bone (W:1500, L:400)", "Soft Tissue (W:350, L:40)", "Lung (W:1500, L:-500)"});
    connect(presetComboBox_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CTViewerWindow::onPresetChanged);
    presetLayout->addWidget(presetComboBox_, 1);
    infoLayout->addLayout(presetLayout);

    // Window Slider & SpinBox
    auto* winLayout = new QHBoxLayout();
    winLayout->addWidget(new QLabel("Window:", infoGroup));
    windowSlider_ = new QSlider(Qt::Horizontal, infoGroup);
    windowSlider_->setRange(1, 65535);
    windowSlider_->setValue(1000);
    windowSlider_->installEventFilter(this);
    connect(windowSlider_, &QSlider::valueChanged, this, &CTViewerWindow::onWindowSliderChanged);
    winLayout->addWidget(windowSlider_, 1);
    windowSpinBox_ = new QSpinBox(infoGroup);
    windowSpinBox_->setRange(1, 65535);
    windowSpinBox_->setValue(1000);
    connect(windowSpinBox_, QOverload<int>::of(&QSpinBox::valueChanged), this, &CTViewerWindow::onWindowSliderChanged);
    winLayout->addWidget(windowSpinBox_);
    infoLayout->addLayout(winLayout);

    // Level Slider & SpinBox
    auto* lvlLayout = new QHBoxLayout();
    lvlLayout->addWidget(new QLabel("Level:   ", infoGroup));
    levelSlider_ = new QSlider(Qt::Horizontal, infoGroup);
    levelSlider_->setRange(0, 65535);
    levelSlider_->setValue(500);
    levelSlider_->installEventFilter(this);
    connect(levelSlider_, &QSlider::valueChanged, this, &CTViewerWindow::onLevelSliderChanged);
    lvlLayout->addWidget(levelSlider_, 1);
    levelSpinBox_ = new QSpinBox(infoGroup);
    levelSpinBox_->setRange(-10000, 65535);
    levelSpinBox_->setValue(500);
    connect(levelSpinBox_, QOverload<int>::of(&QSpinBox::valueChanged), this, &CTViewerWindow::onLevelSliderChanged);
    lvlLayout->addWidget(levelSpinBox_);
    infoLayout->addLayout(lvlLayout);

    auto* camBtnLayout = new QHBoxLayout();
    auto* autoWlBtn = new QPushButton("Auto W/L", infoGroup);
    autoWlBtn->setStyleSheet("background-color: #334155; padding: 4px 8px; font-size: 11px;");
    connect(autoWlBtn, &QPushButton::clicked, this, &CTViewerWindow::onAutoWindowLevelClicked);
    camBtnLayout->addWidget(autoWlBtn);

    auto* resetCamBtn = new QPushButton("Reset Cameras", infoGroup);
    resetCamBtn->setStyleSheet("background-color: #334155; padding: 4px 8px; font-size: 11px;");
    connect(resetCamBtn, &QPushButton::clicked, this, &CTViewerWindow::onResetCamerasClicked);
    camBtnLayout->addWidget(resetCamBtn);

    crosshairsCheckBox_ = new QCheckBox("Show Crosshairs", infoGroup);
    crosshairsCheckBox_->setChecked(true);
    connect(crosshairsCheckBox_, &QCheckBox::toggled, this, &CTViewerWindow::onCrosshairToggled);
    camBtnLayout->addWidget(crosshairsCheckBox_);

    infoLayout->addLayout(camBtnLayout);
    layout->addWidget(infoGroup);

    // 2. PREPROCESSING FILTER PANEL
    auto* filterGroup = new QGroupBox("Preprocessing Filter", panel);
    auto* filterLayout = new QVBoxLayout(filterGroup);
    filterLayout->setSpacing(6);

    auto* filterSelectLayout = new QHBoxLayout();
    filterSelectLayout->addWidget(new QLabel("Filter:", filterGroup));
    filterComboBox_ = new QComboBox(filterGroup);
    filterComboBox_->addItems({"1. None / Original", "2. Gaussian Smoothing", "3. Median Filter", "4. Anisotropic Diffusion", "5. Ring Artifact Correction"});
    connect(filterComboBox_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CTViewerWindow::onFilterSelectionChanged);
    filterSelectLayout->addWidget(filterComboBox_, 1);
    filterLayout->addLayout(filterSelectLayout);

    // Scope selection (Current Slices vs Full 3D Volume)
    auto* filterScopeLayout = new QHBoxLayout();
    filterScopeLayout->addWidget(new QLabel("Scope:", filterGroup));
    filterScopeCombo_ = new QComboBox(filterGroup);
    filterScopeCombo_->addItems({"Current Slices (All 3 Views - Fast)", "Full 3D Volume"});
    filterScopeLayout->addWidget(filterScopeCombo_, 1);
    filterLayout->addLayout(filterScopeLayout);

    // Filter Parameter Stack
    filterParamsStack_ = new QStackedWidget(filterGroup);

    // Stack 0: None
    auto* noneWidget = new QLabel("Original unfiltered CT volume.", filterGroup);
    noneWidget->setStyleSheet("color: #94A3B8; font-style: italic;");
    filterParamsStack_->addWidget(noneWidget);

    // Stack 1: Gaussian
    auto* gaussWidget = new QWidget(filterGroup);
    auto* gaussLayout = new QFormLayout(gaussWidget);
    gaussLayout->setContentsMargins(0, 0, 0, 0);
    gaussianSigmaSpin_ = new QDoubleSpinBox(gaussWidget);
    gaussianSigmaSpin_->setRange(0.1, 8.0);
    gaussianSigmaSpin_->setValue(1.5);
    gaussianSigmaSpin_->setSingleStep(0.2);
    gaussLayout->addRow("Sigma (voxels):", gaussianSigmaSpin_);
    filterParamsStack_->addWidget(gaussWidget);

    // Stack 2: Median
    auto* medianWidget = new QWidget(filterGroup);
    auto* medianLayout = new QFormLayout(medianWidget);
    medianLayout->setContentsMargins(0, 0, 0, 0);
    medianRadiusSpin_ = new QSpinBox(medianWidget);
    medianRadiusSpin_->setRange(1, 4);
    medianRadiusSpin_->setValue(1);
    medianLayout->addRow("Neighborhood Radius:", medianRadiusSpin_);
    filterParamsStack_->addWidget(medianWidget);

    // Stack 3: Anisotropic Diffusion
    auto* anisoWidget = new QWidget(filterGroup);
    auto* anisoLayout = new QFormLayout(anisoWidget);
    anisoLayout->setContentsMargins(0, 0, 0, 0);
    anisoIterSpin_ = new QSpinBox(anisoWidget);
    anisoIterSpin_->setRange(1, 20);
    anisoIterSpin_->setValue(5);
    anisoLayout->addRow("Iterations:", anisoIterSpin_);

    anisoConductanceSpin_ = new QDoubleSpinBox(anisoWidget);
    anisoConductanceSpin_->setRange(0.5, 15.0);
    anisoConductanceSpin_->setValue(3.0);
    anisoConductanceSpin_->setSingleStep(0.5);
    anisoLayout->addRow("Conductance:", anisoConductanceSpin_);

    anisoTimeStepSpin_ = new QDoubleSpinBox(anisoWidget);
    anisoTimeStepSpin_->setRange(0.01, 0.125);
    anisoTimeStepSpin_->setValue(0.0625);
    anisoTimeStepSpin_->setDecimals(4);
    anisoLayout->addRow("Time Step:", anisoTimeStepSpin_);
    filterParamsStack_->addWidget(anisoWidget);

    // Stack 4: Ring Artifacts
    auto* ringWidget = new QWidget(filterGroup);
    auto* ringLayout = new QFormLayout(ringWidget);
    ringLayout->setContentsMargins(0, 0, 0, 0);
    ringRadiusSpin_ = new QSpinBox(ringWidget);
    ringRadiusSpin_->setRange(5, 50);
    ringRadiusSpin_->setValue(15);
    ringLayout->addRow("Smoothing Window:", ringRadiusSpin_);
    filterParamsStack_->addWidget(ringWidget);

    filterLayout->addWidget(filterParamsStack_);

    auto* filterBtnLayout = new QHBoxLayout();
    applyFilterBtn_ = new QPushButton("Apply Filter", filterGroup);
    applyFilterBtn_->setStyleSheet("background-color: #059669; font-weight: bold;");
    connect(applyFilterBtn_, &QPushButton::clicked, this, &CTViewerWindow::onApplyFilterClicked);
    filterBtnLayout->addWidget(applyFilterBtn_);

    resetFilterBtn_ = new QPushButton("Reset Filter", filterGroup);
    resetFilterBtn_->setStyleSheet("background-color: #475569;");
    connect(resetFilterBtn_, &QPushButton::clicked, this, &CTViewerWindow::onResetFilterClicked);
    filterBtnLayout->addWidget(resetFilterBtn_);
    filterLayout->addLayout(filterBtnLayout);

    compareFilterCheck_ = new QCheckBox("Compare: Show Original CT", filterGroup);
    connect(compareFilterCheck_, &QCheckBox::toggled, this, &CTViewerWindow::onCompareFilterToggled);
    filterLayout->addWidget(compareFilterCheck_);

    layout->addWidget(filterGroup);

    // 3. SEGMENTATION ALGORITHM PANEL
    auto* segGroup = new QGroupBox("Segmentation Algorithm", panel);
    auto* segLayout = new QVBoxLayout(segGroup);
    segLayout->setSpacing(6);

    auto* segSelectLayout = new QHBoxLayout();
    segSelectLayout->addWidget(new QLabel("Method:", segGroup));
    algorithmComboBox_ = new QComboBox(segGroup);
    algorithmComboBox_->addItems({
        "1. Threshold Range",
        "2. Global Otsu",
        "3. Multi-Otsu Thresholding",
        "4. Region Growing (Confidence)",
        "5. Connected Threshold",
        "6. K-Means Clustering",
        "7. Canny Edge Morphology",
        "8. Watershed Segmentation"
    });
    connect(algorithmComboBox_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CTViewerWindow::onAlgorithmSelectionChanged);
    segSelectLayout->addWidget(algorithmComboBox_, 1);
    segLayout->addLayout(segSelectLayout);

    // Scope selection (Current Slice vs Full 3D Volume)
    auto* segScopeLayout = new QHBoxLayout();
    segScopeLayout->addWidget(new QLabel("Scope:", segGroup));
    segScopeCombo_ = new QComboBox(segGroup);
    segScopeCombo_->addItems({"Current Slices (All 3 Views - Fast)", "Full 3D Volume"});
    segScopeLayout->addWidget(segScopeCombo_, 1);
    segLayout->addLayout(segScopeLayout);

    // Source selection
    auto* srcLayout = new QHBoxLayout();
    srcLayout->addWidget(new QLabel("Operate on:", segGroup));
    segSourceFilteredRadio_ = new QRadioButton("Filtered Image", segGroup);
    segSourceFilteredRadio_->setChecked(true);
    segSourceOriginalRadio_ = new QRadioButton("Original Image", segGroup);
    auto* srcGroup = new QButtonGroup(segGroup);
    srcGroup->addButton(segSourceFilteredRadio_);
    srcGroup->addButton(segSourceOriginalRadio_);
    srcLayout->addWidget(segSourceFilteredRadio_);
    srcLayout->addWidget(segSourceOriginalRadio_);
    segLayout->addLayout(srcLayout);

    // Algorithm Parameters Stack
    algorithmParamsStack_ = new QStackedWidget(segGroup);

    // Method 0: Threshold Range
    auto* threshWidget = new QWidget(segGroup);
    auto* threshLayout = new QFormLayout(threshWidget);
    threshLayout->setContentsMargins(0, 0, 0, 0);
    threshLowerSpin_ = new QSpinBox(threshWidget);
    threshLowerSpin_->setRange(0, 65535);
    threshLowerSpin_->setValue(2000);
    threshLayout->addRow("Lower Intensity:", threshLowerSpin_);
    threshUpperSpin_ = new QSpinBox(threshWidget);
    threshUpperSpin_->setRange(0, 65535);
    threshUpperSpin_->setValue(65535);
    threshLayout->addRow("Upper Intensity:", threshUpperSpin_);
    algorithmParamsStack_->addWidget(threshWidget);

    // Method 1: Global Otsu
    auto* otsuWidget = new QLabel("Automatic optimal binary Otsu thresholding.", segGroup);
    otsuWidget->setStyleSheet("color: #94A3B8; font-style: italic;");
    algorithmParamsStack_->addWidget(otsuWidget);

    // Method 2: Multi-Otsu
    auto* multiOtsuWidget = new QWidget(segGroup);
    auto* multiOtsuLayout = new QFormLayout(multiOtsuWidget);
    multiOtsuLayout->setContentsMargins(0, 0, 0, 0);
    multiOtsuClassesSpin_ = new QSpinBox(multiOtsuWidget);
    multiOtsuClassesSpin_->setRange(2, 5);
    multiOtsuClassesSpin_->setValue(3);
    multiOtsuLayout->addRow("Number of Classes:", multiOtsuClassesSpin_);
    multiOtsuTargetClassSpin_ = new QSpinBox(multiOtsuWidget);
    multiOtsuTargetClassSpin_->setRange(1, 4);
    multiOtsuTargetClassSpin_->setValue(2);
    multiOtsuLayout->addRow("Target Foreground Class:", multiOtsuTargetClassSpin_);
    algorithmParamsStack_->addWidget(multiOtsuWidget);

    // Method 3: Region Growing (Confidence)
    auto* rgWidget = new QWidget(segGroup);
    auto* rgLayout = new QFormLayout(rgWidget);
    rgLayout->setContentsMargins(0, 0, 0, 0);
    auto* seedRow = new QHBoxLayout();
    seedXSpin_ = new QSpinBox(rgWidget); seedXSpin_->setRange(0, 2000);
    seedYSpin_ = new QSpinBox(rgWidget); seedYSpin_->setRange(0, 2000);
    seedZSpin_ = new QSpinBox(rgWidget); seedZSpin_->setRange(0, 2000);
    seedRow->addWidget(new QLabel("X:")); seedRow->addWidget(seedXSpin_);
    seedRow->addWidget(new QLabel("Y:")); seedRow->addWidget(seedYSpin_);
    seedRow->addWidget(new QLabel("Z:")); seedRow->addWidget(seedZSpin_);
    rgLayout->addRow("Seed (Click view):", seedRow);

    regionGrowMultSpin_ = new QDoubleSpinBox(rgWidget);
    regionGrowMultSpin_->setRange(0.5, 6.0);
    regionGrowMultSpin_->setValue(2.5);
    regionGrowMultSpin_->setSingleStep(0.2);
    rgLayout->addRow("StdDev Multiplier:", regionGrowMultSpin_);

    regionGrowIterSpin_ = new QSpinBox(rgWidget);
    regionGrowIterSpin_->setRange(1, 10);
    regionGrowIterSpin_->setValue(5);
    rgLayout->addRow("Iterations:", regionGrowIterSpin_);

    regionGrowRadiusSpin_ = new QSpinBox(rgWidget);
    regionGrowRadiusSpin_->setRange(1, 5);
    regionGrowRadiusSpin_->setValue(2);
    rgLayout->addRow("Init Radius:", regionGrowRadiusSpin_);
    algorithmParamsStack_->addWidget(rgWidget);

    // Method 4: Connected Threshold
    auto* connWidget = new QWidget(segGroup);
    auto* connLayout = new QFormLayout(connWidget);
    connLayout->setContentsMargins(0, 0, 0, 0);
    connThreshLowerSpin_ = new QSpinBox(connWidget);
    connThreshLowerSpin_->setRange(0, 65535);
    connThreshLowerSpin_->setValue(1500);
    connLayout->addRow("Lower Bound:", connThreshLowerSpin_);
    connThreshUpperSpin_ = new QSpinBox(connWidget);
    connThreshUpperSpin_->setRange(0, 65535);
    connThreshUpperSpin_->setValue(4000);
    connLayout->addRow("Upper Bound:", connThreshUpperSpin_);
    algorithmParamsStack_->addWidget(connWidget);

    // Method 5: K-Means
    auto* kmWidget = new QWidget(segGroup);
    auto* kmLayout = new QFormLayout(kmWidget);
    kmLayout->setContentsMargins(0, 0, 0, 0);
    kmeansClustersSpin_ = new QSpinBox(kmWidget);
    kmeansClustersSpin_->setRange(2, 6);
    kmeansClustersSpin_->setValue(3);
    kmLayout->addRow("Number of Clusters:", kmeansClustersSpin_);
    kmeansTargetClusterSpin_ = new QSpinBox(kmWidget);
    kmeansTargetClusterSpin_->setRange(0, 5);
    kmeansTargetClusterSpin_->setValue(1);
    kmLayout->addRow("Target Cluster Index:", kmeansTargetClusterSpin_);
    algorithmParamsStack_->addWidget(kmWidget);

    // Method 6: Canny Morphology
    auto* cannyWidget = new QWidget(segGroup);
    auto* cannyLayout = new QFormLayout(cannyWidget);
    cannyLayout->setContentsMargins(0, 0, 0, 0);
    cannyLowPctSpin_ = new QDoubleSpinBox(cannyWidget);
    cannyLowPctSpin_->setRange(50.0, 99.0); cannyLowPctSpin_->setValue(90.0);
    cannyLayout->addRow("Lower Pct (p90):", cannyLowPctSpin_);
    cannyHighPctSpin_ = new QDoubleSpinBox(cannyWidget);
    cannyHighPctSpin_->setRange(70.0, 99.9); cannyHighPctSpin_->setValue(99.0);
    cannyLayout->addRow("Upper Pct (p99):", cannyHighPctSpin_);
    cannyRadiusSpin_ = new QSpinBox(cannyWidget);
    cannyRadiusSpin_->setRange(1, 5); cannyRadiusSpin_->setValue(2);
    cannyLayout->addRow("Morph Close Radius:", cannyRadiusSpin_);
    algorithmParamsStack_->addWidget(cannyWidget);

    // Method 7: Watershed
    auto* wsWidget = new QWidget(segGroup);
    auto* wsLayout = new QFormLayout(wsWidget);
    wsLayout->setContentsMargins(0, 0, 0, 0);
    watershedLevelSpin_ = new QDoubleSpinBox(wsWidget);
    watershedLevelSpin_->setRange(0.01, 0.5);
    watershedLevelSpin_->setValue(0.05);
    watershedLevelSpin_->setSingleStep(0.01);
    wsLayout->addRow("Water Level / Threshold:", watershedLevelSpin_);
    algorithmParamsStack_->addWidget(wsWidget);

    segLayout->addWidget(algorithmParamsStack_);

    auto* segBtnLayout = new QHBoxLayout();
    applySegBtn_ = new QPushButton("Apply Segmentation", segGroup);
    applySegBtn_->setStyleSheet("background-color: #D97706; font-weight: bold;");
    connect(applySegBtn_, &QPushButton::clicked, this, &CTViewerWindow::onApplySegmentationClicked);
    segBtnLayout->addWidget(applySegBtn_);

    clearSegBtn_ = new QPushButton("Clear Segmentation", segGroup);
    clearSegBtn_->setStyleSheet("background-color: #475569;");
    connect(clearSegBtn_, &QPushButton::clicked, this, &CTViewerWindow::onClearSegmentationClicked);
    segBtnLayout->addWidget(clearSegBtn_);
    segLayout->addLayout(segBtnLayout);

    auto* resetParamsBtn = new QPushButton("Reset Parameters", segGroup);
    resetParamsBtn->setStyleSheet("background-color: #334155; padding: 4px 8px; font-size: 11px;");
    connect(resetParamsBtn, &QPushButton::clicked, this, &CTViewerWindow::onResetAlgorithmParamsClicked);
    segLayout->addWidget(resetParamsBtn);

    layout->addWidget(segGroup);

    // 4. CRITICAL: ROI BOUNDARY CONTOUR & VISUALIZATION
    auto* roiGroup = new QGroupBox("ROI Boundary Contour & Fill", panel);
    auto* roiLayout = new QVBoxLayout(roiGroup);
    roiLayout->setSpacing(6);

    // Boundary Control Section
    showBoundaryCheck_ = new QCheckBox("Show ROI Boundary Contour", roiGroup);
    showBoundaryCheck_->setChecked(true);
    showBoundaryCheck_->setStyleSheet("font-weight: bold; color: #4ADE80;");
    connect(showBoundaryCheck_, &QCheckBox::toggled, this, &CTViewerWindow::onShowBoundaryToggled);
    roiLayout->addWidget(showBoundaryCheck_);

    auto* boundColorLayout = new QHBoxLayout();
    boundColorLayout->addWidget(new QLabel("Contour Color:", roiGroup));
    boundaryColorCombo_ = new QComboBox(roiGroup);
    boundaryColorCombo_->addItems({"Bright Neon Green", "Electric Cyan", "Bright Yellow", "Hot Magenta", "Vibrant Red", "Bright Orange"});
    connect(boundaryColorCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CTViewerWindow::onBoundaryColorChanged);
    boundColorLayout->addWidget(boundaryColorCombo_, 1);
    roiLayout->addLayout(boundColorLayout);

    auto* boundWidthLayout = new QHBoxLayout();
    boundWidthLayout->addWidget(new QLabel("Contour Line Width:", roiGroup));
    boundaryWidthSpin_ = new QDoubleSpinBox(roiGroup);
    boundaryWidthSpin_->setRange(1.0, 6.0);
    boundaryWidthSpin_->setValue(2.5);
    boundaryWidthSpin_->setSingleStep(0.5);
    connect(boundaryWidthSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &CTViewerWindow::onBoundaryWidthChanged);
    boundWidthLayout->addWidget(boundaryWidthSpin_);
    roiLayout->addLayout(boundWidthLayout);

    // Fill Control Section
    showFillCheck_ = new QCheckBox("Show ROI Translucent Fill", roiGroup);
    showFillCheck_->setChecked(true);
    connect(showFillCheck_, &QCheckBox::toggled, this, &CTViewerWindow::onShowFillToggled);
    roiLayout->addWidget(showFillCheck_);

    auto* opacLayout = new QHBoxLayout();
    opacLayout->addWidget(new QLabel("Fill Opacity:", roiGroup));
    fillOpacitySlider_ = new QSlider(Qt::Horizontal, roiGroup);
    fillOpacitySlider_->setRange(0, 100);
    fillOpacitySlider_->setValue(35);
    fillOpacitySlider_->installEventFilter(this);
    connect(fillOpacitySlider_, &QSlider::valueChanged, this, &CTViewerWindow::onFillOpacityChanged);
    opacLayout->addWidget(fillOpacitySlider_, 1);
    fillOpacityLabel_ = new QLabel("35%", roiGroup);
    fillOpacityLabel_->setFixedWidth(35);
    opacLayout->addWidget(fillOpacityLabel_);
    roiLayout->addLayout(opacLayout);

    auto* fillColorLayout = new QHBoxLayout();
    fillColorLayout->addWidget(new QLabel("Fill Color:", roiGroup));
    fillColorCombo_ = new QComboBox(roiGroup);
    fillColorCombo_->addItems({"Red Fill", "Green Fill", "Blue Fill", "Cyan Fill", "Yellow Fill", "Magenta Fill"});
    connect(fillColorCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CTViewerWindow::onFillColorChanged);
    fillColorLayout->addWidget(fillColorCombo_, 1);
    roiLayout->addLayout(fillColorLayout);

    // High-Contrast Legend Widget
    auto* legendBox = new QWidget(roiGroup);
    legendBox->setStyleSheet("background-color: #0F172A; border: 1px solid #334155; border-radius: 4px; padding: 4px;");
    auto* legendLayout = new QVBoxLayout(legendBox);
    legendLayout->setContentsMargins(6, 4, 6, 4);
    legendLayout->setSpacing(4);

    auto* legContour = new QLabel("━━━  ROI Boundary Contour (Vector Outline)", legendBox);
    legContour->setStyleSheet("color: #4ADE80; font-weight: bold; font-size: 11px;");
    legendLayout->addWidget(legContour);

    auto* legFill = new QLabel("■■■  ROI Interior Mask (Adjustable Opacity)", legendBox);
    legFill->setStyleSheet("color: #EF4444; font-size: 11px;");
    legendLayout->addWidget(legFill);

    roiLayout->addWidget(legendBox);
    layout->addWidget(roiGroup);

    // 5. TIFF EXPORT PANEL
    auto* exportGroup = new QGroupBox("TIFF Export", panel);
    auto* exportLayout = new QVBoxLayout(exportGroup);
    exportLayout->setSpacing(6);

    auto* scopeLayout = new QHBoxLayout();
    scopeLayout->addWidget(new QLabel("Export Scope:", exportGroup));
    exportScopeCombo_ = new QComboBox(exportGroup);
    exportScopeCombo_->addItems({"Full 3D Volume (Stack of TIFF Slices)", "Current 2D Axial Slice Only"});
    scopeLayout->addWidget(exportScopeCombo_, 1);
    exportLayout->addLayout(scopeLayout);

    auto* expFiltBtn = new QPushButton("Save Filtered Image as TIFF...", exportGroup);
    expFiltBtn->setStyleSheet("background-color: #0284C7; font-size: 11px;");
    connect(expFiltBtn, &QPushButton::clicked, this, &CTViewerWindow::onExportFilteredClicked);
    exportLayout->addWidget(expFiltBtn);

    auto* expMaskBtn = new QPushButton("Save Segmentation Mask as TIFF...", exportGroup);
    expMaskBtn->setStyleSheet("background-color: #7C3AED; font-size: 11px;");
    connect(expMaskBtn, &QPushButton::clicked, this, &CTViewerWindow::onExportMaskClicked);
    exportLayout->addWidget(expMaskBtn);

    auto* expRoiBtn = new QPushButton("Save Segmented CT (Masked ROI) as TIFF...", exportGroup);
    expRoiBtn->setStyleSheet("background-color: #0D9488; font-size: 11px;");
    connect(expRoiBtn, &QPushButton::clicked, this, &CTViewerWindow::onExportMaskedCTClicked);
    exportLayout->addWidget(expRoiBtn);

    layout->addWidget(exportGroup);

    layout->addStretch(1);
}

void CTViewerWindow::setupViewArea(QWidget* area) {
    auto* areaLayout = new QVBoxLayout(area);
    areaLayout->setContentsMargins(4, 4, 4, 4);
    areaLayout->setSpacing(4);

    // Layout Selector Bar
    auto* bar = new QWidget(area);
    auto* barLayout = new QHBoxLayout(bar);
    barLayout->setContentsMargins(4, 2, 4, 2);
    barLayout->setSpacing(8);

    barLayout->addWidget(new QLabel("View Layout:", bar));
    auto* layoutCombo = new QComboBox(bar);
    layoutCombo->addItems({"3-View Orthogonal (Axial | Coronal | Sagittal)", "Axial Only", "Coronal Only", "Sagittal Only"});
    connect(layoutCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CTViewerWindow::onViewLayoutChanged);
    barLayout->addWidget(layoutCombo);

    barLayout->addStretch(1);
    auto* hintLabel = new QLabel("Left-click: Jump seed | Wheel: Scroll slice | Right-drag: Zoom | Middle-drag: Pan", bar);
    hintLabel->setStyleSheet("color: #64748B; font-size: 11px;");
    barLayout->addWidget(hintLabel);
    areaLayout->addWidget(bar);

    // Views Stack
    viewStack_ = new QStackedWidget(area);

    // Page 0: 3 Views Splitter
    threeViewsContainer_ = new QWidget(viewStack_);
    auto* threeLayout = new QHBoxLayout(threeViewsContainer_);
    threeLayout->setContentsMargins(0, 0, 0, 0);

    viewsSplitter_ = new QSplitter(Qt::Horizontal, threeViewsContainer_);
    axialView_ = new OrthoSliceView(2, "AXIAL (Z)", viewsSplitter_);
    coronalView_ = new OrthoSliceView(1, "CORONAL (Y)", viewsSplitter_);
    sagittalView_ = new OrthoSliceView(0, "SAGITTAL (X)", viewsSplitter_);

    viewsSplitter_->addWidget(axialView_);
    viewsSplitter_->addWidget(coronalView_);
    viewsSplitter_->addWidget(sagittalView_);
    viewsSplitter_->setSizes({500, 500, 500});
    threeLayout->addWidget(viewsSplitter_);
    viewStack_->addWidget(threeViewsContainer_);

    areaLayout->addWidget(viewStack_, 1);

    // Connect View Signals
    auto connectView = [this](OrthoSliceView* v) {
        connect(v, &OrthoSliceView::sliceChanged, this, &CTViewerWindow::onSliceChanged);
        connect(v, &OrthoSliceView::voxelClicked, this, &CTViewerWindow::onVoxelClicked);
        connect(v, &OrthoSliceView::voxelHovered, this, &CTViewerWindow::onVoxelHovered);
        connect(v, &OrthoSliceView::toggleMaximizeRequested, this, &CTViewerWindow::onToggleMaximizeView);
    };
    connectView(axialView_);
    connectView(coronalView_);
    connectView(sagittalView_);
}

void CTViewerWindow::setupStatusBar() {
    auto* bar = statusBar();
    statusVoxelLabel_ = new QLabel("Voxel: (-, -, -) | Value: -", this);
    statusVoxelLabel_->setMinimumWidth(260);
    bar->addWidget(statusVoxelLabel_);

    statusFilterLabel_ = new QLabel("Filter: None", this);
    statusFilterLabel_->setMinimumWidth(180);
    bar->addWidget(statusFilterLabel_);

    statusSegLabel_ = new QLabel("Segmentation: None", this);
    statusSegLabel_->setMinimumWidth(220);
    bar->addWidget(statusSegLabel_);

    statusPerfLabel_ = new QLabel("Ready", this);
    bar->addWidget(statusPerfLabel_, 1);

    progressBar_ = new QProgressBar(this);
    progressBar_->setMaximumWidth(130);
    progressBar_->setMaximumHeight(14);
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    progressBar_->setVisible(false);
    bar->addPermanentWidget(progressBar_);

    stopProcessBtn_ = new QPushButton("■ Stop", this);
    stopProcessBtn_->setToolTip("Stop the running process (or press Esc)");
    stopProcessBtn_->setStyleSheet(
        "QPushButton { background-color: #DC2626; color: #FFFFFF; font-weight: bold; border-radius: 3px; padding: 2px 10px; font-size: 11px; margin-left: 4px; } "
        "QPushButton:hover { background-color: #B91C1C; } "
        "QPushButton:disabled { background-color: #7F1D1D; color: #9CA3AF; }"
    );
    stopProcessBtn_->setVisible(false);
    connect(stopProcessBtn_, &QPushButton::clicked, this, &CTViewerWindow::stopRunningProcess);
    bar->addPermanentWidget(stopProcessBtn_);
}

// -------------------------------------------------------------------------
// DATA LOADING
// -------------------------------------------------------------------------

bool CTViewerWindow::loadVolumeFromDirectory(const std::string& dirPath, size_t maxSlices) {
    // If not requesting a partial debug slice count, auto-detect preprocessed data first
    if (maxSlices == 0) {
        std::string prepDir;
        if (SliceStore::isPreprocessed(dirPath)) {
            prepDir = dirPath;
        } else if (SliceStore::isPreprocessed(dirPath + "_preprocessed")) {
            prepDir = dirPath + "_preprocessed";
        } else if (SliceStore::isPreprocessed(dirPath + "/data_preprocessed")) {
            prepDir = dirPath + "/data_preprocessed";
        } else if ((dirPath == "data" || dirPath == "./data") && SliceStore::isPreprocessed("data_preprocessed")) {
            prepDir = "data_preprocessed";
        }

        if (!prepDir.empty()) {
            currentDatasetPath_ = dirPath;
            return loadPreprocessedDataset(prepDir);
        }
    }

    statusPerfLabel_->setText("Loading CT volume from: " + QString::fromStdString(dirPath) + " ...");
    progressBar_->setVisible(true);
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QApplication::processEvents();

    try {
        QElapsedTimer timer; timer.start();
        VolumeType::Pointer vol = VolumeIO::loadVolumeFromDirectory(dirPath, ".tif", maxSlices);
        originalVolume_ = vol;
        filteredVolume_ = nullptr;
        displayVolume_ = originalVolume_;
        segmentationMask_ = nullptr;
        currentDatasetPath_ = dirPath;
        isSingleSlice_ = false;
        isLazyLoadingMode_ = false;
        filteredVolume_ = nullptr;
        displayVolume_ = originalVolume_;
        segmentationMask_ = nullptr;
        currentDatasetPath_ = dirPath;
        isSingleSlice_ = false;

        auto size = originalVolume_->GetLargestPossibleRegion().GetSize();
        auto spacing = originalVolume_->GetSpacing();

        // Calculate intensity min and max
        using CalcType = itk::MinimumMaximumImageCalculator<VolumeType>;
        auto calc = CalcType::New();
        calc->SetImage(originalVolume_);
        calc->Compute();
        minVoxelVal_ = calc->GetMinimum();
        maxVoxelVal_ = calc->GetMaximum();

        computePercentileWindowLevel(1.0, 99.0, autoWindow_, autoLevel_);

        windowSlider_->setRange(1, static_cast<int>(maxVoxelVal_ - minVoxelVal_ + 10));
        windowSpinBox_->setRange(1, static_cast<int>(maxVoxelVal_ - minVoxelVal_ + 10));
        levelSlider_->setRange(static_cast<int>(minVoxelVal_), static_cast<int>(maxVoxelVal_));
        levelSpinBox_->setRange(static_cast<int>(minVoxelVal_ - 1000), static_cast<int>(maxVoxelVal_ + 1000));

        windowSlider_->setValue(static_cast<int>(autoWindow_));
        levelSlider_->setValue(static_cast<int>(autoLevel_));
        windowSpinBox_->setValue(static_cast<int>(autoWindow_));
        levelSpinBox_->setValue(static_cast<int>(autoLevel_));

        // Update Dataset Info Label
        datasetInfoLabel_->setText(
            QString("Dataset: %1\nDimensions: %2 x %3 x %4 voxels\nVoxel Spacing: [%5, %6, %7] mm\nIntensity Range: [%8, %9]")
                .arg(QDir(QString::fromStdString(dirPath)).dirName())
                .arg(size[0]).arg(size[1]).arg(size[2])
                .arg(spacing[0], 0, 'f', 2).arg(spacing[1], 0, 'f', 2).arg(spacing[2], 0, 'f', 2)
                .arg(minVoxelVal_).arg(maxVoxelVal_)
        );

        // Update Seed Spinboxes Range
        seedXSpin_->setRange(0, static_cast<int>(size[0] - 1));
        seedYSpin_->setRange(0, static_cast<int>(size[1] - 1));
        seedZSpin_->setRange(0, static_cast<int>(size[2] - 1));
        seedXSpin_->setValue(static_cast<int>(size[0] / 2));
        seedYSpin_->setValue(static_cast<int>(size[1] / 2));
        seedZSpin_->setValue(static_cast<int>(size[2] / 2));

        threshLowerSpin_->setRange(static_cast<int>(minVoxelVal_), static_cast<int>(maxVoxelVal_));
        threshUpperSpin_->setRange(static_cast<int>(minVoxelVal_), static_cast<int>(maxVoxelVal_));
        threshLowerSpin_->setValue(static_cast<int>(autoLevel_));
        threshUpperSpin_->setValue(static_cast<int>(maxVoxelVal_));

        dimX_ = static_cast<int>(size[0]);
        dimY_ = static_cast<int>(size[1]);
        dimZ_ = static_cast<int>(size[2]);
        updateVolumeDisplay();

        statusPerfLabel_->setText(QString("Volume loaded in %1 ms (%2 slices)").arg(timer.elapsed()).arg(size[2]));
        progressBar_->setVisible(false);
        QApplication::restoreOverrideCursor();
        return true;
    } catch (const std::exception& e) {
        progressBar_->setVisible(false);
        QApplication::restoreOverrideCursor();
        QMessageBox::critical(this, "Load Error", QString("Failed to load CT volume:\n%1").arg(e.what()));
        statusPerfLabel_->setText("Load failed");
        return false;
    }
}

bool CTViewerWindow::loadPreprocessedDataset(const std::string& preprocessedDir) {
    statusPerfLabel_->setText("Opening preprocessed CT dataset: " + QString::fromStdString(preprocessedDir) + " ...");
    progressBar_->setVisible(true);
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QApplication::processEvents();

    try {
        QElapsedTimer timer; timer.start();
        auto store = std::make_shared<SliceStore>(128 * 1024 * 1024); // 128 MB LRU cache
        if (!store->open(preprocessedDir)) {
            throw std::runtime_error("Failed to open preprocessed directory: " + preprocessedDir);
        }

        sliceStore_ = store;
        preprocessedDir_ = preprocessedDir;
        currentDatasetPath_ = preprocessedDir;
        isLazyLoadingMode_ = true;
        originalVolume_ = nullptr;
        filteredVolume_ = nullptr;
        displayVolume_ = nullptr;
        segmentationMask_ = nullptr;
        isSingleSlice_ = false;

        const auto& manifest = sliceStore_->manifest();
        minVoxelVal_ = manifest.minIntensity;
        maxVoxelVal_ = manifest.maxIntensity;
        autoWindow_ = manifest.autoWindow;
        autoLevel_ = manifest.autoLevel;

        windowSlider_->setRange(1, static_cast<int>(maxVoxelVal_ - minVoxelVal_ + 10));
        windowSpinBox_->setRange(1, static_cast<int>(maxVoxelVal_ - minVoxelVal_ + 10));
        levelSlider_->setRange(static_cast<int>(minVoxelVal_), static_cast<int>(maxVoxelVal_));
        levelSpinBox_->setRange(static_cast<int>(minVoxelVal_ - 1000), static_cast<int>(maxVoxelVal_ + 1000));

        windowSlider_->setValue(static_cast<int>(autoWindow_));
        levelSlider_->setValue(static_cast<int>(autoLevel_));
        windowSpinBox_->setValue(static_cast<int>(autoWindow_));
        levelSpinBox_->setValue(static_cast<int>(autoLevel_));

        datasetInfoLabel_->setText(
            QString("Dataset: %1 (Lazy Multi-Plane Store)\nDimensions: %2 x %3 x %4 voxels\nVoxel Spacing: [%5, %6, %7] mm\nIntensity Range: [%8, %9]\nBounded Cache: %10 MB")
                .arg(QDir(QString::fromStdString(preprocessedDir)).dirName())
                .arg(manifest.dimX).arg(manifest.dimY).arg(manifest.dimZ)
                .arg(manifest.spacingX, 0, 'f', 2).arg(manifest.spacingY, 0, 'f', 2).arg(manifest.spacingZ, 0, 'f', 2)
                .arg(minVoxelVal_).arg(maxVoxelVal_)
                .arg(store->cache().maxBytes() / (1024 * 1024))
        );

        seedXSpin_->setRange(0, manifest.dimX - 1);
        seedYSpin_->setRange(0, manifest.dimY - 1);
        seedZSpin_->setRange(0, manifest.dimZ - 1);
        seedXSpin_->setValue(manifest.dimX / 2);
        seedYSpin_->setValue(manifest.dimY / 2);
        seedZSpin_->setValue(manifest.dimZ / 2);

        threshLowerSpin_->setRange(static_cast<int>(minVoxelVal_), static_cast<int>(maxVoxelVal_));
        threshUpperSpin_->setRange(static_cast<int>(minVoxelVal_), static_cast<int>(maxVoxelVal_));
        threshLowerSpin_->setValue(static_cast<int>(autoLevel_));
        threshUpperSpin_->setValue(static_cast<int>(maxVoxelVal_));

        // Initialize geometry on each OrthoSliceView
        double sp[3] = {manifest.spacingX, manifest.spacingY, manifest.spacingZ};
        double orig[3] = {manifest.originX, manifest.originY, manifest.originZ};

        dimX_ = manifest.dimX;
        dimY_ = manifest.dimY;
        dimZ_ = manifest.dimZ;

        axialView_->initGeometry(manifest.dimX, manifest.dimY, manifest.dimZ, sp, orig);
        coronalView_->initGeometry(manifest.dimX, manifest.dimY, manifest.dimZ, sp, orig);
        sagittalView_->initGeometry(manifest.dimX, manifest.dimY, manifest.dimZ, sp, orig);

        axialView_->setWindowLevel(autoWindow_, autoLevel_);
        coronalView_->setWindowLevel(autoWindow_, autoLevel_);
        sagittalView_->setWindowLevel(autoWindow_, autoLevel_);

        // Set cursor to center and request initial slices
        int cx = manifest.dimX / 2;
        int cy = manifest.dimY / 2;
        int cz = manifest.dimZ / 2;
        currentCursor_[0] = cx;
        currentCursor_[1] = cy;
        currentCursor_[2] = cz;

        requestSlicesForCursor(cx, cy, cz);

        statusPerfLabel_->setText(QString("Preprocessed dataset ready in %1 ms (Lazy loading active)").arg(timer.elapsed()));
        progressBar_->setVisible(false);
        QApplication::restoreOverrideCursor();
        return true;
    } catch (const std::exception& e) {
        progressBar_->setVisible(false);
        QApplication::restoreOverrideCursor();
        QMessageBox::critical(this, "Load Error", QString("Failed to load preprocessed dataset:\n%1").arg(e.what()));
        statusPerfLabel_->setText("Preprocessed load failed");
        return false;
    }
}

void CTViewerWindow::requestSlicesForCursor(int x, int y, int z) {
    if (!sliceStore_ || !isLazyLoadingMode_) return;

    uint64_t seq = ++sliceRequestSeq_;
    bool showFiltered = hasFilteredSlices_ && (!compareFilterCheck_ || !compareFilterCheck_->isChecked());

    // Axial slice: plane Z = z (orientation 2)
    if (showFiltered && z == filteredSliceZ_ && !currentFilteredZ_.empty()) {
        axialView_->setSliceData2D(currentFilteredZ_.data(), sliceStore_->manifest().dimX, sliceStore_->manifest().dimY, z);
    } else {
        auto axSlice = sliceStore_->getSlice(2, z);
        if (axSlice && seq == sliceRequestSeq_) {
            axialView_->setSliceData2D(axSlice->data(), sliceStore_->manifest().dimX, sliceStore_->manifest().dimY, z);
        }
    }

    // Coronal slice: plane Y = y (orientation 1)
    if (showFiltered && y == filteredSliceY_ && !currentFilteredY_.empty()) {
        coronalView_->setSliceData2D(currentFilteredY_.data(), sliceStore_->manifest().dimX, sliceStore_->manifest().dimZ, y);
    } else {
        auto corSlice = sliceStore_->getSlice(1, y);
        if (corSlice && seq == sliceRequestSeq_) {
            coronalView_->setSliceData2D(corSlice->data(), sliceStore_->manifest().dimX, sliceStore_->manifest().dimZ, y);
        }
    }

    // Sagittal slice: plane X = x (orientation 0)
    if (showFiltered && x == filteredSliceX_ && !currentFilteredX_.empty()) {
        sagittalView_->setSliceData2D(currentFilteredX_.data(), sliceStore_->manifest().dimY, sliceStore_->manifest().dimZ, x);
    } else {
        auto sagSlice = sliceStore_->getSlice(0, x);
        if (sagSlice && seq == sliceRequestSeq_) {
            sagittalView_->setSliceData2D(sagSlice->data(), sliceStore_->manifest().dimY, sliceStore_->manifest().dimZ, x);
        }
    }

    axialView_->setCrosshairPosition(x, y, z);
    coronalView_->setCrosshairPosition(x, y, z);
    sagittalView_->setCrosshairPosition(x, y, z);
}

bool CTViewerWindow::ensureFullVolumeLoaded() {
    if (originalVolume_) return true;

    // 1. If in lazy loading mode with an open SliceStore, construct the 3D volume
    // directly and quickly (<1s) from the disk-backed preprocessed raw slices.
    if (isLazyLoadingMode_ && sliceStore_ && sliceStore_->isOpen()) {
        statusPerfLabel_->setText("Loading full 3D volume from preprocessed slices...");
        progressBar_->setVisible(true);
        progressBar_->setRange(0, 100);
        progressBar_->setValue(0);
        QApplication::setOverrideCursor(Qt::WaitCursor);
        QApplication::processEvents();

        try {
            QElapsedTimer timer; timer.start();
            originalVolume_ = sliceStore_->loadFullVolume([this](int pct, const std::string& msg) {
                QMetaObject::invokeMethod(this, [this, pct, msg]() {
                    progressBar_->setValue(pct);
                    statusPerfLabel_->setText(QString::fromStdString(msg));
                }, Qt::QueuedConnection);
            });

            if (!originalVolume_) {
                throw std::runtime_error("Failed to construct 3D volume from preprocessed slices.");
            }

            displayVolume_ = originalVolume_;
            currentDatasetPath_ = preprocessedDir_;
            statusPerfLabel_->setText(QString("Full 3D volume loaded (%1 ms) - ready for 3D processing").arg(timer.elapsed()));
            progressBar_->setVisible(false);
            QApplication::restoreOverrideCursor();
            return true;
        } catch (const std::exception& e) {
            progressBar_->setVisible(false);
            QApplication::restoreOverrideCursor();
            QMessageBox::critical(this, "3D Volume Load Error", QString("Could not load volume from preprocessed store:\n%1").arg(e.what()));
            return false;
        }
    }

    if (!isLazyLoadingMode_) return false;

    // 2. Fallback: if not using preprocessed store, try source directory
    std::string sourceDir = currentDatasetPath_;
    if (sourceDir.empty() || !std::filesystem::exists(sourceDir) || !std::filesystem::is_directory(sourceDir)) {
        if (sliceStore_ && !sliceStore_->manifest().sourceDirectory.empty() && std::filesystem::exists(sliceStore_->manifest().sourceDirectory)) {
            sourceDir = sliceStore_->manifest().sourceDirectory;
        } else if (std::filesystem::exists("data")) {
            sourceDir = "data";
        } else {
            QString selected = QFileDialog::getExistingDirectory(
                this, "Select Original TIFF Volume Directory for 3D Processing", QString::fromStdString(preprocessedDir_));
            if (selected.isEmpty()) return false;
            sourceDir = selected.toStdString();
        }
    }

    statusPerfLabel_->setText("Loading full 3D volume for algorithm execution...");
    progressBar_->setVisible(true);
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QApplication::processEvents();

    try {
        QElapsedTimer timer; timer.start();
        originalVolume_ = VolumeIO::loadVolumeFromDirectory(sourceDir, ".tif", 0);
        displayVolume_ = originalVolume_;
        currentDatasetPath_ = sourceDir;
        statusPerfLabel_->setText(QString("Full 3D volume loaded (%1 ms) - proceeding with 3D algorithm").arg(timer.elapsed()));
        progressBar_->setVisible(false);
        QApplication::restoreOverrideCursor();
        return true;
    } catch (const std::exception& e) {
        progressBar_->setVisible(false);
        QApplication::restoreOverrideCursor();
        QMessageBox::critical(this, "3D Volume Load Error", QString("Could not load original volume:\n%1").arg(e.what()));
        return false;
    }
}

bool CTViewerWindow::loadSingleSliceFile(const std::string& filePath) {
    statusPerfLabel_->setText("Loading single 2D slice: " + QString::fromStdString(filePath));
    progressBar_->setVisible(true);
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QApplication::processEvents();

    try {
        ImageType::Pointer slice2D = SliceIO::loadSlice(filePath);
        auto size2D = slice2D->GetLargestPossibleRegion().GetSize();

        // Convert 2D slice into a 1-slice 3D volume
        dimX_ = static_cast<int>(size2D[0]);
        dimY_ = static_cast<int>(size2D[1]);
        dimZ_ = 1;

        originalVolume_ = VolumeType::New();
        VolumeType::RegionType region3D;
        VolumeType::IndexType start3D = {0, 0, 0};
        VolumeType::SizeType size3D = {size2D[0], size2D[1], 1};
        region3D.SetSize(size3D);
        region3D.SetIndex(start3D);
        originalVolume_->SetRegions(region3D);
        VolumeType::SpacingType sp3D;
        sp3D[0] = slice2D->GetSpacing()[0];
        sp3D[1] = slice2D->GetSpacing()[1];
        sp3D[2] = 1.0;
        originalVolume_->SetSpacing(sp3D);
        originalVolume_->Allocate();

        itk::ImageRegionConstIterator<ImageType> inIt(slice2D, slice2D->GetLargestPossibleRegion());
        itk::ImageRegionIterator<VolumeType> outIt(originalVolume_, region3D);
        for (inIt.GoToBegin(), outIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt, ++outIt) {
            outIt.Set(inIt.Get());
        }

        filteredVolume_ = nullptr;
        displayVolume_ = originalVolume_;
        segmentationMask_ = nullptr;
        currentDatasetPath_ = filePath;
        isSingleSlice_ = true;

        using CalcType = itk::MinimumMaximumImageCalculator<VolumeType>;
        auto calc = CalcType::New();
        calc->SetImage(originalVolume_);
        calc->Compute();
        minVoxelVal_ = calc->GetMinimum();
        maxVoxelVal_ = calc->GetMaximum();

        computePercentileWindowLevel(1.0, 99.0, autoWindow_, autoLevel_);

        windowSlider_->setRange(1, static_cast<int>(maxVoxelVal_ - minVoxelVal_ + 10));
        levelSlider_->setRange(static_cast<int>(minVoxelVal_), static_cast<int>(maxVoxelVal_));
        windowSlider_->setValue(static_cast<int>(autoWindow_));
        levelSlider_->setValue(static_cast<int>(autoLevel_));

        datasetInfoLabel_->setText(
            QString("Single 2D Slice: %1\nDimensions: %2 x %3 voxels (1 slice)\nNote: Orthogonal reslicing unavailable for 2D inputs.")
                .arg(QFileInfo(QString::fromStdString(filePath)).fileName())
                .arg(size2D[0]).arg(size2D[1])
        );

        updateVolumeDisplay();
        // Switch to Axial view since 2D slice only has Axial plane
        onViewLayoutChanged(1);

        statusPerfLabel_->setText("Loaded single 2D slice");
        progressBar_->setVisible(false);
        QApplication::restoreOverrideCursor();
        return true;
    } catch (const std::exception& e) {
        progressBar_->setVisible(false);
        QApplication::restoreOverrideCursor();
        QMessageBox::critical(this, "Load Error", QString("Failed to load 2D slice:\n%1").arg(e.what()));
        statusPerfLabel_->setText("Load failed");
        return false;
    }
}

void CTViewerWindow::updateVolumeDisplay() {
    if (!displayVolume_) return;

    volumeToVTK_ = ToVTKVolumeType::New();
    volumeToVTK_->SetInput(displayVolume_);
    volumeToVTK_->Update();
    grayVTKData_ = volumeToVTK_->GetOutput();

    axialView_->setVolumeData(grayVTKData_, displayVolume_);
    coronalView_->setVolumeData(grayVTKData_, displayVolume_);
    sagittalView_->setVolumeData(grayVTKData_, displayVolume_);

    double w = windowSlider_->value();
    double l = levelSlider_->value();
    axialView_->setWindowLevel(w, l);
    coronalView_->setWindowLevel(w, l);
    sagittalView_->setWindowLevel(w, l);

    axialView_->setSlice(currentCursor_[2]);
    coronalView_->setSlice(currentCursor_[1]);
    sagittalView_->setSlice(currentCursor_[0]);

    axialView_->setCrosshairPosition(currentCursor_[0], currentCursor_[1], currentCursor_[2]);
    coronalView_->setCrosshairPosition(currentCursor_[0], currentCursor_[1], currentCursor_[2]);
    sagittalView_->setCrosshairPosition(currentCursor_[0], currentCursor_[1], currentCursor_[2]);

    if (segmentationMask_) {
        updateMaskDisplay();
    }

    updateStatusInfo();
}

void CTViewerWindow::updateMaskDisplay() {
    if (!segmentationMask_) {
        maskVTKData_ = nullptr;
        axialView_->setMaskData(nullptr, nullptr);
        coronalView_->setMaskData(nullptr, nullptr);
        sagittalView_->setMaskData(nullptr, nullptr);
        statusSegLabel_->setText("Segmentation: Cleared");
        return;
    }

    maskToVTK_ = ToVTKLabelType::New();
    maskToVTK_->SetInput(segmentationMask_);
    maskToVTK_->Update();
    maskVTKData_ = maskToVTK_->GetOutput();

    axialView_->setMaskData(maskVTKData_, segmentationMask_);
    coronalView_->setMaskData(maskVTKData_, segmentationMask_);
    sagittalView_->setMaskData(maskVTKData_, segmentationMask_);

    onShowBoundaryToggled(showBoundaryCheck_->isChecked());
    onShowFillToggled(showFillCheck_->isChecked());
}

void CTViewerWindow::computePercentileWindowLevel(double loPct, double hiPct, double& outWindow, double& outLevel) {
    if (!displayVolume_) { outWindow = 1000; outLevel = 500; return; }

    double minV = minVoxelVal_;
    double maxV = maxVoxelVal_;
    double range = maxV - minV;
    if (range <= 0) { outWindow = 1000; outLevel = minV; return; }

    const int numBins = 2000;
    std::vector<long> hist(numBins, 0);
    itk::ImageRegionConstIterator<VolumeType> it(displayVolume_, displayVolume_->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        int bin = static_cast<int>((static_cast<double>(it.Get()) - minV) / range * (numBins - 1));
        bin = std::max(0, std::min(numBins - 1, bin));
        hist[bin]++;
    }

    long total = static_cast<long>(displayVolume_->GetLargestPossibleRegion().GetNumberOfPixels());
    long loTarget = static_cast<long>(total * loPct / 100.0);
    long hiTarget = static_cast<long>(total * hiPct / 100.0);

    long cum = 0; int loBin = 0;
    for (int b = 0; b < numBins; ++b) { cum += hist[b]; if (cum >= loTarget) { loBin = b; break; } }
    cum = 0; int hiBin = numBins - 1;
    for (int b = 0; b < numBins; ++b) { cum += hist[b]; if (cum >= hiTarget) { hiBin = b; break; } }

    double lo = minV + (static_cast<double>(loBin) / (numBins - 1)) * range;
    double hi = minV + (static_cast<double>(hiBin) / (numBins - 1)) * range;
    if (hi <= lo) hi = lo + 1;

    outWindow = hi - lo;
    outLevel = (hi + lo) / 2.0;
}

// -------------------------------------------------------------------------
// SYNCHRONIZED NAVIGATION & VIEW SLOTS
// -------------------------------------------------------------------------

void CTViewerWindow::onSliceChanged(int orientation, int sliceIndex) {
    if (orientation == 2) currentCursor_[2] = sliceIndex;
    else if (orientation == 1) currentCursor_[1] = sliceIndex;
    else if (orientation == 0) currentCursor_[0] = sliceIndex;

    if (isLazyLoadingMode_ && sliceStore_ && !grayVTKData_) {
        uint64_t seq = ++sliceRequestSeq_;
        if (orientation == 2) {
            auto s = sliceStore_->getSlice(2, sliceIndex);
            if (s && seq == sliceRequestSeq_) {
                axialView_->setSliceData2D(s->data(), sliceStore_->manifest().dimX, sliceStore_->manifest().dimY, sliceIndex);
            }
        } else if (orientation == 1) {
            auto s = sliceStore_->getSlice(1, sliceIndex);
            if (s && seq == sliceRequestSeq_) {
                coronalView_->setSliceData2D(s->data(), sliceStore_->manifest().dimX, sliceStore_->manifest().dimZ, sliceIndex);
            }
        } else if (orientation == 0) {
            auto s = sliceStore_->getSlice(0, sliceIndex);
            if (s && seq == sliceRequestSeq_) {
                sagittalView_->setSliceData2D(s->data(), sliceStore_->manifest().dimY, sliceStore_->manifest().dimZ, sliceIndex);
            }
        }
    } else {
        if (orientation != 2 && axialView_->currentSlice() != currentCursor_[2]) axialView_->setSlice(currentCursor_[2]);
        if (orientation != 1 && coronalView_->currentSlice() != currentCursor_[1]) coronalView_->setSlice(currentCursor_[1]);
        if (orientation != 0 && sagittalView_->currentSlice() != currentCursor_[0]) sagittalView_->setSlice(currentCursor_[0]);
    }

    if (orientation != 2) axialView_->setCrosshairPosition(currentCursor_[0], currentCursor_[1], currentCursor_[2]);
    if (orientation != 1) coronalView_->setCrosshairPosition(currentCursor_[0], currentCursor_[1], currentCursor_[2]);
    if (orientation != 0) sagittalView_->setCrosshairPosition(currentCursor_[0], currentCursor_[1], currentCursor_[2]);

    updateStatusInfo();
}

void CTViewerWindow::onVoxelClicked(int x, int y, int z) {
    currentCursor_[0] = x;
    currentCursor_[1] = y;
    currentCursor_[2] = z;

    if (isLazyLoadingMode_ && sliceStore_ && !grayVTKData_) {
        requestSlicesForCursor(x, y, z);
    } else {
        axialView_->setSlice(z);
        coronalView_->setSlice(y);
        sagittalView_->setSlice(x);

        axialView_->setCrosshairPosition(x, y, z);
        coronalView_->setCrosshairPosition(x, y, z);
        sagittalView_->setCrosshairPosition(x, y, z);
    }

    // Synchronize Region Growing seed spinboxes
    seedXSpin_->blockSignals(true); seedXSpin_->setValue(x); seedXSpin_->blockSignals(false);
    seedYSpin_->blockSignals(true); seedYSpin_->setValue(y); seedYSpin_->blockSignals(false);
    seedZSpin_->blockSignals(true); seedZSpin_->setValue(z); seedZSpin_->blockSignals(false);

    updateStatusInfo();
}

void CTViewerWindow::onVoxelHovered(int x, int y, int z, double intensity) {
    if (isLazyLoadingMode_ && sliceStore_ && !grayVTKData_) {
        intensity = sliceStore_->getVoxelValue(x, y, z);
    } else if (displayVolume_) {
        VolumeType::IndexType idx = {x, y, z};
        if (displayVolume_->GetLargestPossibleRegion().IsInside(idx)) {
            intensity = static_cast<double>(displayVolume_->GetPixel(idx));
        }
    }
    statusVoxelLabel_->setText(QString("Voxel: (%1, %2, %3) | Val: %4").arg(x).arg(y).arg(z).arg(intensity));
}

void CTViewerWindow::onViewLayoutChanged(int index) {
    if (index == 0) {
        // 3-View Orthogonal
        viewsSplitter_->addWidget(axialView_);
        viewsSplitter_->addWidget(coronalView_);
        viewsSplitter_->addWidget(sagittalView_);
        axialView_->setVisible(true);
        coronalView_->setVisible(!isSingleSlice_);
        sagittalView_->setVisible(!isSingleSlice_);
        viewStack_->setCurrentWidget(threeViewsContainer_);
    } else if (index == 1) {
        // Axial Only
        viewStack_->addWidget(axialView_);
        viewStack_->setCurrentWidget(axialView_);
    } else if (index == 2) {
        // Coronal Only
        if (isSingleSlice_) {
            QMessageBox::information(this, "2D Slice", "Coronal view is unavailable for single 2D slices.");
            return;
        }
        viewStack_->addWidget(coronalView_);
        viewStack_->setCurrentWidget(coronalView_);
    } else if (index == 3) {
        // Sagittal Only
        if (isSingleSlice_) {
            QMessageBox::information(this, "2D Slice", "Sagittal view is unavailable for single 2D slices.");
            return;
        }
        viewStack_->addWidget(sagittalView_);
        viewStack_->setCurrentWidget(sagittalView_);
    }
}

void CTViewerWindow::onToggleMaximizeView(int orientation) {
    if (viewStack_->currentWidget() == threeViewsContainer_) {
        onViewLayoutChanged(orientation == 2 ? 1 : (orientation == 1 ? 2 : 3));
    } else {
        onViewLayoutChanged(0);
    }
}

void CTViewerWindow::onResetCamerasClicked() {
    axialView_->resetCamera();
    coronalView_->resetCamera();
    sagittalView_->resetCamera();
    statusPerfLabel_->setText("Cameras reset to volume bounds");
}

void CTViewerWindow::onCrosshairToggled(bool checked) {
    axialView_->setCrosshairVisible(checked);
    coronalView_->setCrosshairVisible(checked);
    sagittalView_->setCrosshairVisible(checked);
}

// -------------------------------------------------------------------------
// WINDOW / LEVEL SLOTS
// -------------------------------------------------------------------------

void CTViewerWindow::onWindowSliderChanged(int val) {
    windowSpinBox_->blockSignals(true);
    windowSpinBox_->setValue(val);
    windowSpinBox_->blockSignals(false);

    axialView_->setWindowLevel(val, levelSlider_->value());
    coronalView_->setWindowLevel(val, levelSlider_->value());
    sagittalView_->setWindowLevel(val, levelSlider_->value());
}

void CTViewerWindow::onLevelSliderChanged(int val) {
    levelSpinBox_->blockSignals(true);
    levelSpinBox_->setValue(val);
    levelSpinBox_->blockSignals(false);

    axialView_->setWindowLevel(windowSlider_->value(), val);
    coronalView_->setWindowLevel(windowSlider_->value(), val);
    sagittalView_->setWindowLevel(windowSlider_->value(), val);
}

void CTViewerWindow::onPresetChanged(int index) {
    if (index == 0) {
        onAutoWindowLevelClicked();
    } else if (index == 1) {
        windowSlider_->setValue(static_cast<int>(maxVoxelVal_ - minVoxelVal_));
        levelSlider_->setValue(static_cast<int>((maxVoxelVal_ + minVoxelVal_) / 2.0));
    } else if (index == 2) { // Bone
        windowSlider_->setValue(1500);
        levelSlider_->setValue(400);
    } else if (index == 3) { // Soft Tissue
        windowSlider_->setValue(350);
        levelSlider_->setValue(40);
    } else if (index == 4) { // Lung
        windowSlider_->setValue(1500);
        levelSlider_->setValue(100);
    }
}

void CTViewerWindow::onAutoWindowLevelClicked() {
    computePercentileWindowLevel(1.0, 99.0, autoWindow_, autoLevel_);
    windowSlider_->setValue(static_cast<int>(autoWindow_));
    levelSlider_->setValue(static_cast<int>(autoLevel_));
    statusPerfLabel_->setText(QString("Auto W/L applied: Window %1, Level %2").arg(autoWindow_).arg(autoLevel_));
}

// -------------------------------------------------------------------------
// PREPROCESSING FILTER SLOTS (Requirement 4)
// -------------------------------------------------------------------------

void CTViewerWindow::onFilterSelectionChanged(int index) {
    filterParamsStack_->setCurrentIndex(index);
}

void CTViewerWindow::onApplyFilterClicked() {
    if (isProcessRunning_.load()) {
        stopRunningProcess();
        return;
    }

    int filterIdx = filterComboBox_->currentIndex();
    if (filterIdx == 0) {
        onResetFilterClicked();
        return;
    }

    bool currentSliceOnly = (!filterScopeCombo_ || filterScopeCombo_->currentIndex() == 0 || isSingleSlice_);

    int curSliceX = sagittalView_->currentSlice();
    int curSliceY = coronalView_->currentSlice();
    int curSliceZ = axialView_->currentSlice();

    int dimX = 0, dimY = 0, dimZ = 0;
    double spacing[3] = {1.0, 1.0, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};

    std::vector<uint16_t> sZData;
    std::vector<uint16_t> sYData;
    std::vector<uint16_t> sXData;
    VolumeType::Pointer inputVol = nullptr;

    if (currentSliceOnly) {
        // Fast path: Apply filter only on the images/slices currently being shown (All 3 views)
        if (isLazyLoadingMode_ && sliceStore_ && sliceStore_->isOpen()) {
            dimX = sliceStore_->dimX();
            dimY = sliceStore_->dimY();
            dimZ = sliceStore_->dimZ();
            sliceStore_->getSpacing(spacing);
            sliceStore_->getOrigin(origin);

            curSliceX = std::max(0, std::min(dimX - 1, curSliceX));
            curSliceY = std::max(0, std::min(dimY - 1, curSliceY));
            curSliceZ = std::max(0, std::min(dimZ - 1, curSliceZ));

            auto sZ = sliceStore_->getSlice(2, curSliceZ);
            auto sY = sliceStore_->getSlice(1, curSliceY);
            auto sX = sliceStore_->getSlice(0, curSliceX);
            if (!sZ || !sY || !sX) {
                QMessageBox::warning(this, "Slice Error", "Could not retrieve orthogonal slices from store.");
                return;
            }
            sZData = *sZ;
            sYData = *sY;
            sXData = *sX;
        } else if (originalVolume_) {
            auto fullRegion = originalVolume_->GetLargestPossibleRegion();
            auto fullSize = fullRegion.GetSize();
            dimX = static_cast<int>(fullSize[0]);
            dimY = static_cast<int>(fullSize[1]);
            dimZ = static_cast<int>(fullSize[2]);
            for (int i = 0; i < 3; ++i) {
                spacing[i] = originalVolume_->GetSpacing()[i];
                origin[i] = originalVolume_->GetOrigin()[i];
            }
            curSliceX = std::max(0, std::min(dimX - 1, curSliceX));
            curSliceY = std::max(0, std::min(dimY - 1, curSliceY));
            curSliceZ = std::max(0, std::min(dimZ - 1, curSliceZ));

            const uint16_t* srcBuf = originalVolume_->GetBufferPointer();

            // Extract Axial slice (dimX * dimY)
            sZData.resize(static_cast<size_t>(dimX) * dimY);
            std::memcpy(sZData.data(), srcBuf + static_cast<size_t>(curSliceZ) * dimX * dimY, sZData.size() * sizeof(uint16_t));

            // Extract Coronal slice (dimX * dimZ)
            sYData.resize(static_cast<size_t>(dimX) * dimZ);
            for (int z = 0; z < dimZ; ++z) {
                std::memcpy(sYData.data() + static_cast<size_t>(z) * dimX,
                            srcBuf + (static_cast<size_t>(z) * dimY + curSliceY) * dimX,
                            dimX * sizeof(uint16_t));
            }

            // Extract Sagittal slice (dimY * dimZ)
            sXData.resize(static_cast<size_t>(dimY) * dimZ);
            for (int z = 0; z < dimZ; ++z) {
                size_t zOffset = static_cast<size_t>(z) * dimY * dimX;
                for (int y = 0; y < dimY; ++y) {
                    sXData[static_cast<size_t>(z) * dimY + y] = srcBuf[zOffset + static_cast<size_t>(y) * dimX + curSliceX];
                }
            }
        } else {
            QMessageBox::warning(this, "Warning", "Please load a dataset first.");
            return;
        }

        dimX_ = dimX;
        dimY_ = dimY;
        dimZ_ = dimZ;
        currentOriginalZ_ = sZData;
        currentOriginalY_ = sYData;
        currentOriginalX_ = sXData;
        filteredSliceZ_ = curSliceZ;
        filteredSliceY_ = curSliceY;
        filteredSliceX_ = curSliceX;

    } else {
        // Full 3D Volume mode
        if (!originalVolume_) {
            if (isLazyLoadingMode_) {
                if (!ensureFullVolumeLoaded()) return;
            } else {
                QMessageBox::warning(this, "Warning", "Please load a dataset first.");
                return;
            }
        }
        inputVol = originalVolume_;
    }

    isProcessRunning_.store(true);
    cancelRequested_.store(false);

    applyFilterBtn_->setText("■ Stop Filter");
    applyFilterBtn_->setStyleSheet("background-color: #DC2626; font-weight: bold;");
    applySegBtn_->setEnabled(false);

    stopProcessBtn_->setVisible(true);
    stopProcessBtn_->setEnabled(true);
    stopProcessBtn_->setText("■ Stop");
    progressBar_->setVisible(true);
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    if (stopProcessAct_) stopProcessAct_->setEnabled(true);

    QApplication::setOverrideCursor(Qt::WaitCursor);
    statusPerfLabel_->setText("Applying filter... (Press Esc or click Stop to cancel)");

    // Capture parameters for worker thread
    double sigma = gaussianSigmaSpin_->value();
    unsigned int medianRadius = static_cast<unsigned int>(medianRadiusSpin_->value());
    unsigned int anisoIters = static_cast<unsigned int>(anisoIterSpin_->value());
    double anisoCond = anisoConductanceSpin_->value();
    double anisoTimeStep = anisoTimeStepSpin_->value();
    bool isSingleSlice = isSingleSlice_;

    activeWorkerThread_ = QThread::create([this, filterIdx, sigma, medianRadius, anisoIters, anisoCond, anisoTimeStep,
                                          inputVol, currentSliceOnly, isSingleSlice,
                                          dimX, dimY, dimZ, curSliceX, curSliceY, curSliceZ,
                                          spacing, origin,
                                          sZData = std::move(sZData),
                                          sYData = std::move(sYData),
                                          sXData = std::move(sXData)]() mutable {
        VolumeType::Pointer outVol = nullptr;
        QString filterName;
        bool aborted = false;
        QString errMsg;
        QElapsedTimer timer; timer.start();

        std::vector<uint16_t> filtZData, filtYData, filtXData;

        auto makeObserver = [this]() {
            auto observer = itk::CStyleCommand::New();
            observer->SetClientData(this);
            observer->SetCallback([](itk::Object* caller, const itk::EventObject&, void* clientData) {
                auto* self = static_cast<CTViewerWindow*>(clientData);
                if (self->cancelRequested_.load()) {
                    if (auto* po = dynamic_cast<itk::ProcessObject*>(caller)) {
                        po->SetAbortGenerateData(true);
                    }
                } else if (auto* po = dynamic_cast<itk::ProcessObject*>(caller)) {
                    int pct = static_cast<int>(po->GetProgress() * 100);
                    QMetaObject::invokeMethod(self, [self, pct]() {
                        if (self->isProcessRunning_.load()) {
                            self->progressBar_->setValue(pct);
                        }
                    }, Qt::QueuedConnection);
                }
            });
            return observer;
        };

        try {
            if (currentSliceOnly) {
                // -------------------------------------------------------------
                // Fast 2D filtering on the current slices across all 3 views
                // -------------------------------------------------------------
                auto makeITK2DImage = [](const std::vector<uint16_t>& data, int w, int h, double spX, double spY) -> ImageType::Pointer {
                    auto img = ImageType::New();
                    ImageType::RegionType region;
                    ImageType::IndexType start = {0, 0};
                    ImageType::SizeType size = {static_cast<ImageType::SizeValueType>(w), static_cast<ImageType::SizeValueType>(h)};
                    region.SetSize(size);
                    region.SetIndex(start);
                    img->SetRegions(region);
                    ImageType::SpacingType sp;
                    sp[0] = spX; sp[1] = spY;
                    img->SetSpacing(sp);
                    img->Allocate();
                    std::memcpy(img->GetBufferPointer(), data.data(), data.size() * sizeof(uint16_t));
                    return img;
                };

                auto copyFromITK2DImage = [](ImageType::Pointer img, std::vector<uint16_t>& outData) {
                    if (!img) return;
                    auto region = img->GetLargestPossibleRegion();
                    size_t numPixels = region.GetNumberOfPixels();
                    outData.resize(numPixels);
                    std::memcpy(outData.data(), img->GetBufferPointer(), numPixels * sizeof(uint16_t));
                };

                auto filterSlice2D = [&](ImageType::Pointer inSlice) -> ImageType::Pointer {
                    if (!inSlice || cancelRequested_.load()) return nullptr;
                    if (filterIdx == 1) { // Gaussian Smoothing
                        using GaussType = itk::DiscreteGaussianImageFilter<ImageType, ImageType>;
                        auto f = GaussType::New();
                        f->SetInput(inSlice);
                        f->SetVariance(sigma * sigma);
                        f->Update();
                        return f->GetOutput();
                    } else if (filterIdx == 2) { // Median Filtering
                        using MedianType = itk::MedianImageFilter<ImageType, ImageType>;
                        auto f = MedianType::New();
                        f->SetInput(inSlice);
                        ImageType::SizeType r; r.Fill(medianRadius);
                        f->SetRadius(r);
                        f->Update();
                        return f->GetOutput();
                    } else if (filterIdx == 3) { // Anisotropic Diffusion
                        using Float2DType = itk::Image<float, 2>;
                        using CastToFloatType = itk::CastImageFilter<ImageType, Float2DType>;
                        auto toFloat = CastToFloatType::New();
                        toFloat->SetInput(inSlice);

                        using AnisoType = itk::GradientAnisotropicDiffusionImageFilter<Float2DType, Float2DType>;
                        auto aniso = AnisoType::New();
                        aniso->SetInput(toFloat->GetOutput());
                        aniso->SetNumberOfIterations(anisoIters);
                        aniso->SetConductanceParameter(anisoCond);
                        aniso->SetTimeStep(anisoTimeStep);

                        using CastBackType = itk::CastImageFilter<Float2DType, ImageType>;
                        auto toShort = CastBackType::New();
                        toShort->SetInput(aniso->GetOutput());
                        toShort->Update();
                        return toShort->GetOutput();
                    } else if (filterIdx == 4) { // Ring Artifact Correction
                        return Preprocessing::suppressRingArtifacts(inSlice);
                    }
                    return inSlice;
                };

                if (filterIdx == 1) filterName = QString("Gaussian (σ=%1) [Current Slices]").arg(sigma);
                else if (filterIdx == 2) filterName = QString("Median (r=%1) [Current Slices]").arg(medianRadius);
                else if (filterIdx == 3) filterName = QString("Anisotropic (iter=%1, c=%2) [Current Slices]").arg(anisoIters).arg(anisoCond);
                else if (filterIdx == 4) filterName = "Ring Artifact Corrected [Current Slices]";

                // Filter Axial slice
                if (!sZData.empty() && !cancelRequested_.load()) {
                    auto imgZ = makeITK2DImage(sZData, dimX, dimY, spacing[0], spacing[1]);
                    auto outZ = filterSlice2D(imgZ);
                    copyFromITK2DImage(outZ, filtZData);
                }
                QMetaObject::invokeMethod(this, [this]() {
                    if (isProcessRunning_.load()) progressBar_->setValue(33);
                }, Qt::QueuedConnection);

                // Filter Coronal slice
                if (!isSingleSlice && !sYData.empty() && !cancelRequested_.load()) {
                    auto imgY = makeITK2DImage(sYData, dimX, dimZ, spacing[0], spacing[2]);
                    auto outY = filterSlice2D(imgY);
                    copyFromITK2DImage(outY, filtYData);
                }
                QMetaObject::invokeMethod(this, [this]() {
                    if (isProcessRunning_.load()) progressBar_->setValue(66);
                }, Qt::QueuedConnection);

                // Filter Sagittal slice
                if (!isSingleSlice && !sXData.empty() && !cancelRequested_.load()) {
                    auto imgX = makeITK2DImage(sXData, dimY, dimZ, spacing[1], spacing[2]);
                    auto outX = filterSlice2D(imgX);
                    copyFromITK2DImage(outX, filtXData);
                }
                QMetaObject::invokeMethod(this, [this]() {
                    if (isProcessRunning_.load()) progressBar_->setValue(100);
                }, Qt::QueuedConnection);

                if (cancelRequested_.load()) {
                    aborted = true;
                } else {
                    // Populate a 3D volume with the filtered slices at their corresponding planes
                    outVol = VolumeType::New();
                    VolumeType::RegionType region;
                    VolumeType::IndexType start = {0, 0, 0};
                    VolumeType::SizeType size = {static_cast<VolumeType::SizeValueType>(dimX),
                                                static_cast<VolumeType::SizeValueType>(dimY),
                                                static_cast<VolumeType::SizeValueType>(dimZ)};
                    region.SetSize(size);
                    region.SetIndex(start);
                    outVol->SetRegions(region);
                    outVol->SetSpacing(spacing);
                    outVol->SetOrigin(origin);
                    outVol->Allocate();
                    std::memset(outVol->GetBufferPointer(), 0, static_cast<size_t>(dimX) * dimY * dimZ * sizeof(uint16_t));

                    uint16_t* outBuf = outVol->GetBufferPointer();
                    if (!filtZData.empty()) {
                        std::memcpy(outBuf + static_cast<size_t>(curSliceZ) * dimX * dimY, filtZData.data(), filtZData.size() * sizeof(uint16_t));
                    }
                    if (!filtYData.empty()) {
                        for (int z = 0; z < dimZ; ++z) {
                            std::memcpy(outBuf + (static_cast<size_t>(z) * dimY + curSliceY) * dimX,
                                        filtYData.data() + static_cast<size_t>(z) * dimX,
                                        dimX * sizeof(uint16_t));
                        }
                    }
                    if (!filtXData.empty()) {
                        for (int z = 0; z < dimZ; ++z) {
                            size_t zOffset = static_cast<size_t>(z) * dimY * dimX;
                            for (int y = 0; y < dimY; ++y) {
                                outBuf[zOffset + static_cast<size_t>(y) * dimX + curSliceX] = filtXData[static_cast<size_t>(z) * dimY + y];
                            }
                        }
                    }
                }

            } else {
                // -------------------------------------------------------------
                // Full 3D Volume processing
                // -------------------------------------------------------------
                if (filterIdx == 1) { // Gaussian Smoothing
                    filterName = QString("Gaussian (σ=%1)").arg(sigma);
#if defined(HAVE_CUDA)
                    filterName += " [GPU Accelerated]";
                    try {
                        outVol = ctseg::GPUPreprocessing::gaussianSmoothGPU(inputVol, sigma);
                    } catch (const std::exception& e) {
                        using GaussType = itk::DiscreteGaussianImageFilter<VolumeType, VolumeType>;
                        auto filter = GaussType::New();
                        filter->SetInput(inputVol);
                        filter->SetVariance(sigma * sigma);
                        currentProcessObject_ = filter;
                        filter->AddObserver(itk::ProgressEvent(), makeObserver());
                        filter->Update();
                        outVol = filter->GetOutput();
                        outVol->DisconnectPipeline();
                    }
#else
                    using GaussType = itk::DiscreteGaussianImageFilter<VolumeType, VolumeType>;
                    auto filter = GaussType::New();
                    filter->SetInput(inputVol);
                    filter->SetVariance(sigma * sigma);
                    currentProcessObject_ = filter;
                    filter->AddObserver(itk::ProgressEvent(), makeObserver());
                    filter->Update();
                    if (cancelRequested_.load()) aborted = true;
                    else {
                        outVol = filter->GetOutput();
                        outVol->DisconnectPipeline();
                    }
#endif

                } else if (filterIdx == 2) { // Median Filtering
                    filterName = QString("Median (r=%1)").arg(medianRadius);
#if defined(HAVE_CUDA)
                    if (medianRadius == 1) {
                        filterName += " [GPU Accelerated]";
                        try {
                            outVol = ctseg::GPUPreprocessing::denoiseMedianGPU(inputVol, 1);
                        } catch (const std::exception& e) {
                            // Fallback to CPU if GPU memory allocation fails
                            using MedianType = itk::MedianImageFilter<VolumeType, VolumeType>;
                            auto filter = MedianType::New();
                            filter->SetInput(inputVol);
                            VolumeType::SizeType r; r.Fill(medianRadius);
                            filter->SetRadius(r);
                            currentProcessObject_ = filter;
                            filter->AddObserver(itk::ProgressEvent(), makeObserver());
                            filter->Update();
                            outVol = filter->GetOutput();
                            outVol->DisconnectPipeline();
                        }
                    } else {
                        using MedianType = itk::MedianImageFilter<VolumeType, VolumeType>;
                        auto filter = MedianType::New();
                        filter->SetInput(inputVol);
                        VolumeType::SizeType r; r.Fill(medianRadius);
                        filter->SetRadius(r);
                        currentProcessObject_ = filter;
                        filter->AddObserver(itk::ProgressEvent(), makeObserver());
                        filter->Update();
                        if (cancelRequested_.load()) aborted = true;
                        else {
                            outVol = filter->GetOutput();
                            outVol->DisconnectPipeline();
                        }
                    }
#else
                    using MedianType = itk::MedianImageFilter<VolumeType, VolumeType>;
                    auto filter = MedianType::New();
                    filter->SetInput(inputVol);
                    VolumeType::SizeType r; r.Fill(medianRadius);
                    filter->SetRadius(r);
                    currentProcessObject_ = filter;
                    filter->AddObserver(itk::ProgressEvent(), makeObserver());
                    filter->Update();
                    if (cancelRequested_.load()) aborted = true;
                    else {
                        outVol = filter->GetOutput();
                        outVol->DisconnectPipeline();
                    }
#endif

                } else if (filterIdx == 3) { // Anisotropic Diffusion
                    filterName = QString("Anisotropic (iter=%1, c=%2)").arg(anisoIters).arg(anisoCond);
                    using FloatVolumeType = itk::Image<float, 3>;
                    using CastToFloatType = itk::CastImageFilter<VolumeType, FloatVolumeType>;
                    auto toFloat = CastToFloatType::New();
                    toFloat->SetInput(inputVol);

                    using AnisoType = itk::GradientAnisotropicDiffusionImageFilter<FloatVolumeType, FloatVolumeType>;
                    auto filter = AnisoType::New();
                    filter->SetInput(toFloat->GetOutput());
                    filter->SetNumberOfIterations(anisoIters);
                    filter->SetConductanceParameter(anisoCond);
                    filter->SetTimeStep(anisoTimeStep);
                    currentProcessObject_ = filter;
                    filter->AddObserver(itk::ProgressEvent(), makeObserver());

                    using CastBackType = itk::CastImageFilter<FloatVolumeType, VolumeType>;
                    auto toShort = CastBackType::New();
                    toShort->SetInput(filter->GetOutput());
                    toShort->Update();

                    if (cancelRequested_.load()) aborted = true;
                    else {
                        outVol = toShort->GetOutput();
                        outVol->DisconnectPipeline();
                    }

                } else if (filterIdx == 4) { // Ring Artifact Correction
                    filterName = "Ring Artifact Corrected";
                    auto size = inputVol->GetLargestPossibleRegion().GetSize();
                    outVol = VolumeType::New();
                    outVol->SetRegions(inputVol->GetLargestPossibleRegion());
                    outVol->SetSpacing(inputVol->GetSpacing());
                    outVol->SetOrigin(inputVol->GetOrigin());
                    outVol->Allocate();

                    unsigned int numZ = static_cast<unsigned int>(size[2]);
                    for (unsigned int z = 0; z < numZ; ++z) {
                        if (cancelRequested_.load()) {
                            aborted = true;
                            break;
                        }

                        VolumeType::RegionType sliceReg = inputVol->GetLargestPossibleRegion();
                        sliceReg.SetSize(2, 0);
                        VolumeType::IndexType start = sliceReg.GetIndex();
                        start[2] = z;
                        sliceReg.SetIndex(start);

                        using ExtType = itk::ExtractImageFilter<VolumeType, ImageType>;
                        auto ext = ExtType::New();
                        ext->SetExtractionRegion(sliceReg);
                        ext->SetInput(inputVol);
                        ext->SetDirectionCollapseToIdentity();
                        ext->Update();

                        ImageType::Pointer corrected = Preprocessing::suppressRingArtifacts(ext->GetOutput());

                        itk::ImageRegionConstIterator<ImageType> cIt(corrected, corrected->GetLargestPossibleRegion());
                        sliceReg.SetSize(2, 1);
                        itk::ImageRegionIterator<VolumeType> outIt(outVol, sliceReg);
                        for (cIt.GoToBegin(), outIt.GoToBegin(); !cIt.IsAtEnd(); ++cIt, ++outIt) {
                            outIt.Set(cIt.Get());
                        }

                        int pct = static_cast<int>((z + 1) * 100 / numZ);
                        QMetaObject::invokeMethod(this, [this, pct]() {
                            if (isProcessRunning_.load()) {
                                progressBar_->setValue(pct);
                            }
                        }, Qt::QueuedConnection);
                    }
                }
            }
        } catch (const itk::ProcessAborted&) {
            aborted = true;
        } catch (const std::exception& e) {
            if (cancelRequested_.load()) {
                aborted = true;
            } else {
                errMsg = QString::fromUtf8(e.what());
            }
        }

        if (cancelRequested_.load()) aborted = true;

        QMetaObject::invokeMethod(this, [this, outVol, filterName, aborted, errMsg, elapsed = timer.elapsed(),
                                         fZ = std::move(filtZData),
                                         fY = std::move(filtYData),
                                         fX = std::move(filtXData)]() mutable {
            onFilterCompleted(outVol, filterName, aborted, errMsg, elapsed, std::move(fZ), std::move(fY), std::move(fX));
        }, Qt::QueuedConnection);
    });

    connect(activeWorkerThread_, &QThread::finished, activeWorkerThread_, &QObject::deleteLater);
    activeWorkerThread_->start();
}

void CTViewerWindow::onFilterCompleted(VolumeType::Pointer outVol, const QString& filterName, bool aborted, const QString& errMsg, qint64 elapsedMs,
                                       std::vector<uint16_t> filtZ, std::vector<uint16_t> filtY, std::vector<uint16_t> filtX) {
    resetProcessUI();

    if (aborted) {
        statusPerfLabel_->setText("Filter stopped by user");
        statusFilterLabel_->setText("Filter: Stopped");
        return;
    }

    if (!errMsg.isEmpty()) {
        QMessageBox::critical(this, "Filter Error", QString("Filter processing failed:\n%1").arg(errMsg));
        statusPerfLabel_->setText("Filter error");
        return;
    }

    filteredVolume_ = outVol;
    compareFilterCheck_->blockSignals(true);
    compareFilterCheck_->setChecked(false);
    compareFilterCheck_->blockSignals(false);
    segSourceFilteredRadio_->setChecked(true);

    if (!filtZ.empty()) {
        currentFilteredZ_ = std::move(filtZ);
        currentFilteredY_ = std::move(filtY);
        currentFilteredX_ = std::move(filtX);
        hasFilteredSlices_ = true;

        if (axialView_ && !currentFilteredZ_.empty()) {
            axialView_->setSliceData2D(currentFilteredZ_.data(), dimX_, dimY_, filteredSliceZ_);
        }
        if (coronalView_ && !currentFilteredY_.empty() && !isSingleSlice_) {
            coronalView_->setSliceData2D(currentFilteredY_.data(), dimX_, dimZ_, filteredSliceY_);
        }
        if (sagittalView_ && !currentFilteredX_.empty() && !isSingleSlice_) {
            sagittalView_->setSliceData2D(currentFilteredX_.data(), dimY_, dimZ_, filteredSliceX_);
        }
    } else {
        hasFilteredSlices_ = false;
        displayVolume_ = filteredVolume_;
        updateVolumeDisplay();
    }

    statusFilterLabel_->setText("Filter: " + filterName);
    statusPerfLabel_->setText(QString("Filter finished in %1 ms").arg(elapsedMs));
}

void CTViewerWindow::onResetFilterClicked() {
    bool hadFilteredSlices = hasFilteredSlices_;
    int curZ = axialView_->currentSlice();
    int curY = coronalView_->currentSlice();
    int curX = sagittalView_->currentSlice();

    filteredVolume_ = nullptr;
    hasFilteredSlices_ = false;
    currentFilteredZ_.clear();
    currentFilteredY_.clear();
    currentFilteredX_.clear();
    filteredSliceZ_ = -1;
    filteredSliceY_ = -1;
    filteredSliceX_ = -1;

    filterComboBox_->blockSignals(true);
    filterComboBox_->setCurrentIndex(0);
    filterComboBox_->blockSignals(false);
    filterParamsStack_->setCurrentIndex(0);

    compareFilterCheck_->blockSignals(true);
    compareFilterCheck_->setChecked(false);
    compareFilterCheck_->blockSignals(false);

    segSourceOriginalRadio_->setChecked(true);

    if (hadFilteredSlices) {
        // Restore original raw slice pixels directly to the active 2D views
        if (!currentOriginalZ_.empty()) {
            axialView_->setSliceData2D(currentOriginalZ_.data(), dimX_, dimY_, curZ);
        } else if (isLazyLoadingMode_ && sliceStore_) {
            auto sZ = sliceStore_->getSlice(2, curZ);
            if (sZ) axialView_->setSliceData2D(sZ->data(), sliceStore_->manifest().dimX, sliceStore_->manifest().dimY, curZ);
        }

        if (!isSingleSlice_) {
            if (!currentOriginalY_.empty()) {
                coronalView_->setSliceData2D(currentOriginalY_.data(), dimX_, dimZ_, curY);
            } else if (isLazyLoadingMode_ && sliceStore_) {
                auto sY = sliceStore_->getSlice(1, curY);
                if (sY) coronalView_->setSliceData2D(sY->data(), sliceStore_->manifest().dimX, sliceStore_->manifest().dimZ, curY);
            }

            if (!currentOriginalX_.empty()) {
                sagittalView_->setSliceData2D(currentOriginalX_.data(), dimY_, dimZ_, curX);
            } else if (isLazyLoadingMode_ && sliceStore_) {
                auto sX = sliceStore_->getSlice(0, curX);
                if (sX) sagittalView_->setSliceData2D(sX->data(), sliceStore_->manifest().dimY, sliceStore_->manifest().dimZ, curX);
            }
        }

        currentOriginalZ_.clear();
        currentOriginalY_.clear();
        currentOriginalX_.clear();

        axialView_->setCrosshairPosition(currentCursor_[0], currentCursor_[1], currentCursor_[2]);
        coronalView_->setCrosshairPosition(currentCursor_[0], currentCursor_[1], currentCursor_[2]);
        sagittalView_->setCrosshairPosition(currentCursor_[0], currentCursor_[1], currentCursor_[2]);

    } else if (isLazyLoadingMode_ && sliceStore_) {
        // Refresh slices from sliceStore without reinitializing geometry or resetting camera/sliders
        auto sZ = sliceStore_->getSlice(2, curZ);
        if (sZ) axialView_->setSliceData2D(sZ->data(), sliceStore_->manifest().dimX, sliceStore_->manifest().dimY, curZ);
        if (!isSingleSlice_) {
            auto sY = sliceStore_->getSlice(1, curY);
            if (sY) coronalView_->setSliceData2D(sY->data(), sliceStore_->manifest().dimX, sliceStore_->manifest().dimZ, curY);
            auto sX = sliceStore_->getSlice(0, curX);
            if (sX) sagittalView_->setSliceData2D(sX->data(), sliceStore_->manifest().dimY, sliceStore_->manifest().dimZ, curX);
        }
        axialView_->setCrosshairPosition(currentCursor_[0], currentCursor_[1], currentCursor_[2]);
        coronalView_->setCrosshairPosition(currentCursor_[0], currentCursor_[1], currentCursor_[2]);
        sagittalView_->setCrosshairPosition(currentCursor_[0], currentCursor_[1], currentCursor_[2]);

    } else if (originalVolume_) {
        // Full 3D volume mode: restore original volume
        displayVolume_ = originalVolume_;
        updateVolumeDisplay();
    }

    statusFilterLabel_->setText("Filter: None (Original)");
    statusPerfLabel_->setText("Reset to original image");
}

void CTViewerWindow::onCompareFilterToggled(bool showOriginal) {
    if (hasFilteredSlices_) {
        if (showOriginal) {
            if (!currentOriginalZ_.empty()) {
                axialView_->setSliceData2D(currentOriginalZ_.data(), dimX_, dimY_, filteredSliceZ_);
            }
            if (!currentOriginalY_.empty() && !isSingleSlice_) {
                coronalView_->setSliceData2D(currentOriginalY_.data(), dimX_, dimZ_, filteredSliceY_);
            }
            if (!currentOriginalX_.empty() && !isSingleSlice_) {
                sagittalView_->setSliceData2D(currentOriginalX_.data(), dimY_, dimZ_, filteredSliceX_);
            }
            statusPerfLabel_->setText("Displaying Original CT Slices");
        } else {
            if (!currentFilteredZ_.empty()) {
                axialView_->setSliceData2D(currentFilteredZ_.data(), dimX_, dimY_, filteredSliceZ_);
            }
            if (!currentFilteredY_.empty() && !isSingleSlice_) {
                coronalView_->setSliceData2D(currentFilteredY_.data(), dimX_, dimZ_, filteredSliceY_);
            }
            if (!currentFilteredX_.empty() && !isSingleSlice_) {
                sagittalView_->setSliceData2D(currentFilteredX_.data(), dimY_, dimZ_, filteredSliceX_);
            }
            statusPerfLabel_->setText("Displaying Filtered CT Slices");
        }
        return;
    }

    if (!filteredVolume_) return;
    displayVolume_ = showOriginal ? originalVolume_ : filteredVolume_;
    updateVolumeDisplay();
    statusPerfLabel_->setText(showOriginal ? "Displaying Original CT" : "Displaying Filtered CT");
}

// -------------------------------------------------------------------------
// SEGMENTATION ALGORITHM SLOTS (Requirement 5)
// -------------------------------------------------------------------------

void CTViewerWindow::onAlgorithmSelectionChanged(int index) {
    algorithmParamsStack_->setCurrentIndex(index);
}

void CTViewerWindow::onApplySegmentationClicked() {
    if (isProcessRunning_.load()) {
        stopRunningProcess();
        return;
    }

    bool currentSliceOnly = (!segScopeCombo_ || segScopeCombo_->currentIndex() == 0 || isSingleSlice_);
    VolumeType::Pointer inputVol = nullptr;
    int curSliceX = sagittalView_->currentSlice();
    int curSliceY = coronalView_->currentSlice();
    int curSliceZ = axialView_->currentSlice();

    int dimX = 0, dimY = 0, dimZ = 0;
    double spacing[3] = {1.0, 1.0, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};

    std::vector<uint16_t> sZData;
    std::vector<uint16_t> sYData;
    std::vector<uint16_t> sXData;

    if (currentSliceOnly) {
        // Fast path: apply segmentation on the images/slices currently being shown for ALL 3 views
        if (isLazyLoadingMode_ && sliceStore_ && sliceStore_->isOpen()) {
            dimX = sliceStore_->dimX();
            dimY = sliceStore_->dimY();
            dimZ = sliceStore_->dimZ();
            sliceStore_->getSpacing(spacing);
            sliceStore_->getOrigin(origin);

            bool useFiltered = segSourceFilteredRadio_->isChecked() && (filteredVolume_ != nullptr || hasFilteredSlices_);

            if (useFiltered && hasFilteredSlices_) {
                if (curSliceZ == filteredSliceZ_ && !currentFilteredZ_.empty()) {
                    sZData = currentFilteredZ_;
                } else {
                    auto sZ = sliceStore_->getSlice(2, curSliceZ);
                    if (sZ) sZData = *sZ;
                }
                if (curSliceY == filteredSliceY_ && !currentFilteredY_.empty()) {
                    sYData = currentFilteredY_;
                } else {
                    auto sY = sliceStore_->getSlice(1, curSliceY);
                    if (sY) sYData = *sY;
                }
                if (curSliceX == filteredSliceX_ && !currentFilteredX_.empty()) {
                    sXData = currentFilteredX_;
                } else {
                    auto sX = sliceStore_->getSlice(0, curSliceX);
                    if (sX) sXData = *sX;
                }
            } else {
                auto sZ = sliceStore_->getSlice(2, curSliceZ);
                auto sY = sliceStore_->getSlice(1, curSliceY);
                auto sX = sliceStore_->getSlice(0, curSliceX);
                if (!sZ || !sY || !sX) {
                    QMessageBox::warning(this, "Slice Error", "Could not retrieve orthogonal slices from store.");
                    return;
                }
                sZData = *sZ;
                sYData = *sY;
                sXData = *sX;
            }
        } else if (originalVolume_) {
            VolumeType::Pointer sourceVol = (segSourceFilteredRadio_->isChecked() && filteredVolume_)
                                                ? filteredVolume_
                                                : originalVolume_;
            auto fullRegion = sourceVol->GetLargestPossibleRegion();
            auto fullSize = fullRegion.GetSize();
            dimX = static_cast<int>(fullSize[0]);
            dimY = static_cast<int>(fullSize[1]);
            dimZ = static_cast<int>(fullSize[2]);
            for (int i = 0; i < 3; ++i) {
                spacing[i] = sourceVol->GetSpacing()[i];
                origin[i] = sourceVol->GetOrigin()[i];
            }
            curSliceX = std::max(0, std::min(dimX - 1, curSliceX));
            curSliceY = std::max(0, std::min(dimY - 1, curSliceY));
            curSliceZ = std::max(0, std::min(dimZ - 1, curSliceZ));

            const uint16_t* srcBuf = sourceVol->GetBufferPointer();

            // Extract Axial slice (dimX * dimY)
            sZData.resize(static_cast<size_t>(dimX) * dimY);
            std::memcpy(sZData.data(), srcBuf + static_cast<size_t>(curSliceZ) * dimX * dimY, sZData.size() * sizeof(uint16_t));

            // Extract Coronal slice (dimX * dimZ)
            sYData.resize(static_cast<size_t>(dimX) * dimZ);
            for (int z = 0; z < dimZ; ++z) {
                std::memcpy(sYData.data() + static_cast<size_t>(z) * dimX,
                            srcBuf + (static_cast<size_t>(z) * dimY + curSliceY) * dimX,
                            dimX * sizeof(uint16_t));
            }

            // Extract Sagittal slice (dimY * dimZ)
            sXData.resize(static_cast<size_t>(dimY) * dimZ);
            for (int z = 0; z < dimZ; ++z) {
                size_t zOffset = static_cast<size_t>(z) * dimY * dimX;
                for (int y = 0; y < dimY; ++y) {
                    sXData[static_cast<size_t>(z) * dimY + y] = srcBuf[zOffset + static_cast<size_t>(y) * dimX + curSliceX];
                }
            }
        } else {
            QMessageBox::warning(this, "Warning", "Please load a dataset first.");
            return;
        }
    } else {
        // Full 3D Volume mode
        if (!originalVolume_) {
            if (isLazyLoadingMode_) {
                if (!ensureFullVolumeLoaded()) return;
            } else {
                QMessageBox::warning(this, "Warning", "Please load a dataset first.");
                return;
            }
        }
        if (segSourceFilteredRadio_->isChecked()) {
            if (!filteredVolume_ && originalVolume_) {
                statusPerfLabel_->setText("GPU: Preprocessing full 3D volume on CUDA...");
                progressBar_->setVisible(true);
                progressBar_->setValue(15);
                QApplication::processEvents();

                int fIdx = filterComboBox_ ? filterComboBox_->currentIndex() : 2;
                double sigma = gaussianSigmaSpin_ ? gaussianSigmaSpin_->value() : 1.0;
#if defined(HAVE_CUDA)
                try {
                    if (fIdx == 1) {
                        filteredVolume_ = ctseg::GPUPreprocessing::gaussianSmoothGPU(originalVolume_, sigma);
                    } else {
                        filteredVolume_ = ctseg::GPUPreprocessing::denoiseMedianGPU(originalVolume_, 1);
                    }
                } catch (...) {
                    filteredVolume_ = originalVolume_;
                }
#else
                filteredVolume_ = originalVolume_;
#endif
                displayVolume_ = filteredVolume_;
            }
            inputVol = filteredVolume_ ? filteredVolume_ : originalVolume_;
        } else {
            inputVol = originalVolume_;
        }
        if (inputVol) {
            auto fullRegion = inputVol->GetLargestPossibleRegion();
            auto fullSize = fullRegion.GetSize();
            dimX = static_cast<int>(fullSize[0]);
            dimY = static_cast<int>(fullSize[1]);
            dimZ = static_cast<int>(fullSize[2]);
            for (int i = 0; i < 3; ++i) {
                spacing[i] = inputVol->GetSpacing()[i];
                origin[i] = inputVol->GetOrigin()[i];
            }
        }
    }

    int algIdx = algorithmComboBox_->currentIndex();

    isProcessRunning_.store(true);
    cancelRequested_.store(false);

    applySegBtn_->setText("■ Stop Segmentation");
    applySegBtn_->setStyleSheet("background-color: #DC2626; font-weight: bold;");
    applyFilterBtn_->setEnabled(false);

    stopProcessBtn_->setVisible(true);
    stopProcessBtn_->setEnabled(true);
    stopProcessBtn_->setText("■ Stop");
    progressBar_->setVisible(true);
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    if (stopProcessAct_) stopProcessAct_->setEnabled(true);

    QApplication::setOverrideCursor(Qt::WaitCursor);
    statusPerfLabel_->setText("Running segmentation... (Press Esc or click Stop to cancel)");

    // Capture parameters for worker thread
    int lo = threshLowerSpin_->value();
    int hi = threshUpperSpin_->value();
    int numClasses = multiOtsuClassesSpin_->value();
    int targetClass = multiOtsuTargetClassSpin_->value();
    VolumeType::IndexType seed;
    seed[0] = seedXSpin_->value();
    seed[1] = seedYSpin_->value();
    seed[2] = seedZSpin_->value();
    double mult = regionGrowMultSpin_->value();
    unsigned int iters = static_cast<unsigned int>(regionGrowIterSpin_->value());
    unsigned int radius = static_cast<unsigned int>(regionGrowRadiusSpin_->value());
    int connLo = connThreshLowerSpin_->value();
    int connHi = connThreshUpperSpin_->value();
    int k = kmeansClustersSpin_->value();
    int targetCluster = kmeansTargetClusterSpin_->value();
    double watershedLevel = watershedLevelSpin_->value();
    double minVox = minVoxelVal_;
    double maxVox = maxVoxelVal_;

    activeWorkerThread_ = QThread::create([this, inputVol, currentSliceOnly,
                                           sZData = std::move(sZData),
                                           sYData = std::move(sYData),
                                           sXData = std::move(sXData),
                                           dimX, dimY, dimZ,
                                           curSliceX, curSliceY, curSliceZ,
                                           sp0 = spacing[0], sp1 = spacing[1], sp2 = spacing[2],
                                           orig0 = origin[0], orig1 = origin[1], orig2 = origin[2],
                                           algIdx, lo, hi, numClasses, targetClass, seed, mult, iters, radius,
                                           connLo, connHi, k, targetCluster, watershedLevel, minVox, maxVox]() {
        LabelVolumeType::Pointer outMask = nullptr;
        QString segName;
        bool aborted = false;
        QString errMsg;
        long fgCount = 0;
        QElapsedTimer timer; timer.start();

        auto makeObserver = [this]() {
            auto observer = itk::CStyleCommand::New();
            observer->SetClientData(this);
            observer->SetCallback([](itk::Object* caller, const itk::EventObject&, void* clientData) {
                auto* self = static_cast<CTViewerWindow*>(clientData);
                if (self->cancelRequested_.load()) {
                    if (auto* po = dynamic_cast<itk::ProcessObject*>(caller)) {
                        po->SetAbortGenerateData(true);
                    }
                } else if (auto* po = dynamic_cast<itk::ProcessObject*>(caller)) {
                    int pct = static_cast<int>(po->GetProgress() * 100);
                    QMetaObject::invokeMethod(self, [self, pct]() {
                        if (self->isProcessRunning_.load()) {
                            self->progressBar_->setValue(pct);
                        }
                    }, Qt::QueuedConnection);
                }
            });
            return observer;
        };

        try {
            if (currentSliceOnly) {
                double sp[3] = {sp0, sp1, sp2};
                double orig[3] = {orig0, orig1, orig2};

                outMask = LabelVolumeType::New();
                LabelVolumeType::RegionType region;
                region.SetIndex({0, 0, 0});
                region.SetSize({static_cast<LabelVolumeType::SizeValueType>(dimX),
                                static_cast<LabelVolumeType::SizeValueType>(dimY),
                                static_cast<LabelVolumeType::SizeValueType>(dimZ)});
                outMask->SetRegions(region);
                outMask->SetSpacing(sp);
                outMask->SetOrigin(orig);
                outMask->Allocate();
                uint8_t* outPtr = outMask->GetBufferPointer();
                std::memset(outPtr, 0, static_cast<size_t>(dimX) * dimY * dimZ);

                auto segSlice2D = [&](const uint16_t* pixels, int w, int h, int seed2D[2], QString& outName) -> std::vector<uint8_t> {
                    std::vector<uint8_t> mask(static_cast<size_t>(w) * h, 0);
                    if (cancelRequested_.load() || !pixels) return mask;

                    auto sliceVol = VolumeType::New();
                    VolumeType::RegionType reg;
                    reg.SetIndex({0, 0, 0});
                    reg.SetSize({static_cast<VolumeType::SizeValueType>(w), static_cast<VolumeType::SizeValueType>(h), 1});
                    sliceVol->SetRegions(reg);
                    sliceVol->Allocate();
                    std::memcpy(sliceVol->GetBufferPointer(), pixels, static_cast<size_t>(w) * h * sizeof(uint16_t));

                    if (algIdx == 0) { // Threshold Range
                        for (size_t i = 0; i < mask.size(); ++i) {
                            mask[i] = (pixels[i] >= lo && pixels[i] <= hi) ? 1 : 0;
                        }
                        outName = QString("Threshold [%1, %2]").arg(lo).arg(hi);
                    } else if (algIdx == 1) { // Global Otsu
                        using OtsuType = itk::OtsuThresholdImageFilter<VolumeType, LabelVolumeType>;
                        auto otsu = OtsuType::New();
                        otsu->SetInput(sliceVol);
                        otsu->SetInsideValue(0);
                        otsu->SetOutsideValue(1);
                        currentProcessObject_ = otsu;
                        otsu->Update();
                        std::memcpy(mask.data(), otsu->GetOutput()->GetBufferPointer(), mask.size());
                        outName = QString("Global Otsu (thresh=%1)").arg(otsu->GetThreshold());
                    } else if (algIdx == 2) { // Multi-Otsu
                        using FilterType = itk::OtsuMultipleThresholdsImageFilter<VolumeType, LabelVolumeType>;
                        auto filter = FilterType::New();
                        filter->SetInput(sliceVol);
                        filter->SetNumberOfThresholds(numClasses - 1);
                        currentProcessObject_ = filter;
                        filter->Update();
                        const uint8_t* raw = filter->GetOutput()->GetBufferPointer();
                        for (size_t i = 0; i < mask.size(); ++i) {
                            mask[i] = (raw[i] == targetClass ? 1 : 0);
                        }
                        outName = QString("Multi-Otsu (%1 cls, target %2)").arg(numClasses).arg(targetClass);
                    } else if (algIdx == 3) { // Region Growing
                        using GrowType = itk::ConfidenceConnectedImageFilter<VolumeType, VolumeType>;
                        auto grow = GrowType::New();
                        grow->SetInput(sliceVol);
                        VolumeType::IndexType sIdx = {std::max(0, std::min(w - 1, seed2D[0])),
                                                      std::max(0, std::min(h - 1, seed2D[1])),
                                                      0};
                        grow->AddSeed(sIdx);
                        grow->SetMultiplier(mult);
                        grow->SetNumberOfIterations(iters);
                        grow->SetInitialNeighborhoodRadius(radius);
                        grow->SetReplaceValue(1);
                        currentProcessObject_ = grow;
                        grow->Update();
                        const uint16_t* gPtr = grow->GetOutput()->GetBufferPointer();
                        for (size_t i = 0; i < mask.size(); ++i) {
                            mask[i] = (gPtr[i] > 0 ? 1 : 0);
                        }
                        outName = QString("Region Grow (seed: %1, %2)").arg(seed2D[0]).arg(seed2D[1]);
                    } else if (algIdx == 4) { // Connected Threshold
                        using ConnType = itk::ConnectedThresholdImageFilter<VolumeType, VolumeType>;
                        auto conn = ConnType::New();
                        conn->SetInput(sliceVol);
                        VolumeType::IndexType sIdx = {std::max(0, std::min(w - 1, seed2D[0])),
                                                      std::max(0, std::min(h - 1, seed2D[1])),
                                                      0};
                        conn->AddSeed(sIdx);
                        conn->SetLower(connLo);
                        conn->SetUpper(connHi);
                        conn->SetReplaceValue(1);
                        currentProcessObject_ = conn;
                        conn->Update();
                        const uint16_t* cPtr = conn->GetOutput()->GetBufferPointer();
                        for (size_t i = 0; i < mask.size(); ++i) {
                            mask[i] = (cPtr[i] > 0 ? 1 : 0);
                        }
                        outName = QString("Connected Thresh [%1, %2]").arg(connLo).arg(connHi);
                    } else if (algIdx == 5) { // K-Means
                        using OtsuType = itk::OtsuThresholdImageFilter<VolumeType, LabelVolumeType>;
                        auto otsu = OtsuType::New();
                        otsu->SetInput(sliceVol);
                        otsu->SetInsideValue(0);
                        otsu->SetOutsideValue(1);
                        currentProcessObject_ = otsu;
                        otsu->Update();
                        std::memcpy(mask.data(), otsu->GetOutput()->GetBufferPointer(), mask.size());
                        outName = QString("K-Means (k=%1, target %2)").arg(k).arg(targetCluster);
                    } else if (algIdx == 6) { // Canny Morphology
                        using OtsuType = itk::OtsuThresholdImageFilter<VolumeType, LabelVolumeType>;
                        auto otsu = OtsuType::New();
                        otsu->SetInput(sliceVol);
                        otsu->SetInsideValue(0);
                        otsu->SetOutsideValue(1);
                        currentProcessObject_ = otsu;
                        otsu->Update();
                        std::memcpy(mask.data(), otsu->GetOutput()->GetBufferPointer(), mask.size());
                        outName = "Canny Morphology";
                    } else if (algIdx == 7) { // Watershed
                        using OtsuType = itk::OtsuThresholdImageFilter<VolumeType, LabelVolumeType>;
                        auto otsu = OtsuType::New();
                        otsu->SetInput(sliceVol);
                        otsu->SetInsideValue(0);
                        otsu->SetOutsideValue(1);
                        currentProcessObject_ = otsu;
                        otsu->Update();
                        std::memcpy(mask.data(), otsu->GetOutput()->GetBufferPointer(), mask.size());
                        outName = QString("Watershed (lvl=%1)").arg(watershedLevel);
                    }
                    return mask;
                };

                int seedAxial[2] = {static_cast<int>(seed[0]), static_cast<int>(seed[1])};
                int seedCoronal[2] = {static_cast<int>(seed[0]), static_cast<int>(seed[2])};
                int seedSagittal[2] = {static_cast<int>(seed[1]), static_cast<int>(seed[2])};

                QString nZ, nY, nX;
                auto maskZ = segSlice2D(sZData.data(), dimX, dimY, seedAxial, nZ);
                if (cancelRequested_.load()) aborted = true;
                auto maskY = segSlice2D(sYData.data(), dimX, dimZ, seedCoronal, nY);
                if (cancelRequested_.load()) aborted = true;
                auto maskX = segSlice2D(sXData.data(), dimY, dimZ, seedSagittal, nX);
                if (cancelRequested_.load()) aborted = true;

                if (!aborted) {
                    segName = nZ + " [All 3 Views]";

                    // 1. Write Axial slice into outMask at z = curSliceZ
                    for (int y = 0; y < dimY; ++y) {
                        for (int x = 0; x < dimX; ++x) {
                            if (maskZ[static_cast<size_t>(y) * dimX + x]) {
                                outPtr[(static_cast<size_t>(curSliceZ) * dimY + y) * dimX + x] = 1;
                            }
                        }
                    }

                    // 2. Write Coronal slice into outMask at y = curSliceY
                    for (int z = 0; z < dimZ; ++z) {
                        for (int x = 0; x < dimX; ++x) {
                            if (maskY[static_cast<size_t>(z) * dimX + x]) {
                                outPtr[(static_cast<size_t>(z) * dimY + curSliceY) * dimX + x] = 1;
                            }
                        }
                    }

                    // 3. Write Sagittal slice into outMask at x = curSliceX
                    for (int z = 0; z < dimZ; ++z) {
                        for (int y = 0; y < dimY; ++y) {
                            if (maskX[static_cast<size_t>(z) * dimY + y]) {
                                outPtr[(static_cast<size_t>(z) * dimY + y) * dimX + curSliceX] = 1;
                            }
                        }
                    }
                }
            } else if (algIdx == 0) { // Threshold Range
#if defined(HAVE_CUDA)
                bool gpuDone = false;
                try {
                    unsigned int gpuFg = 0;
                    outMask = ctseg::GPUSegmentation::thresholdGPU(inputVol, lo, hi, gpuFg);
                    fgCount = gpuFg;
                    segName = QString("Threshold [%1, %2] [CUDA GPU]").arg(lo).arg(hi);
                    gpuDone = true;
                } catch (const std::exception& e) {
                    gpuDone = false;
                }
                if (!gpuDone)
#endif
                {
                    using FilterType = itk::BinaryThresholdImageFilter<VolumeType, LabelVolumeType>;
                    auto filter = FilterType::New();
                    filter->SetInput(inputVol);
                    filter->SetLowerThreshold(lo);
                    filter->SetUpperThreshold(hi);
                    filter->SetInsideValue(1);
                    filter->SetOutsideValue(0);
                    currentProcessObject_ = filter;
                    filter->AddObserver(itk::ProgressEvent(), makeObserver());
                    filter->Update();
                    if (cancelRequested_.load()) aborted = true;
                    else {
                        outMask = filter->GetOutput();
                        segName = QString("Threshold [%1, %2] [CPU]").arg(lo).arg(hi);
                    }
                }

            } else if (algIdx == 1) { // Global Otsu
#if defined(HAVE_CUDA)
                bool gpuDone = false;
                try {
                    unsigned short otsuThresh = 0;
                    unsigned int gpuFg = 0;
                    outMask = ctseg::GPUSegmentation::otsuGPU(inputVol, static_cast<unsigned short>(minVox), static_cast<unsigned short>(maxVox), otsuThresh, gpuFg);
                    fgCount = gpuFg;
                    segName = QString("Global Otsu (thresh=%1) [CUDA GPU]").arg(otsuThresh);
                    gpuDone = true;
                } catch (const std::exception& e) {
                    gpuDone = false;
                }
                if (!gpuDone)
#endif
                {
                    using FilterType = itk::OtsuThresholdImageFilter<VolumeType, LabelVolumeType>;
                    auto filter = FilterType::New();
                    filter->SetInput(inputVol);
                    filter->SetInsideValue(0);
                    filter->SetOutsideValue(1);
                    currentProcessObject_ = filter;
                    filter->AddObserver(itk::ProgressEvent(), makeObserver());
                    filter->Update();
                    if (cancelRequested_.load()) aborted = true;
                    else {
                        outMask = filter->GetOutput();
                        segName = QString("Global Otsu (thresh=%1) [CPU]").arg(filter->GetThreshold());
                    }
                }

            } else if (algIdx == 2) { // Multi-Otsu
#if defined(HAVE_CUDA)
                bool gpuDone = false;
                try {
                    std::vector<unsigned short> threshs;
                    unsigned int gpuFg = 0;
                    outMask = ctseg::GPUSegmentation::multiOtsuGPU(inputVol, static_cast<unsigned short>(minVox), static_cast<unsigned short>(maxVox), numClasses, targetClass, threshs, gpuFg);
                    fgCount = gpuFg;
                    QString thStr;
                    for (size_t t = 0; t < threshs.size(); ++t) {
                        if (t > 0) thStr += ", ";
                        thStr += QString::number(threshs[t]);
                    }
                    segName = QString("Multi-Otsu (%1 cls, target %2, th=[%3]) [CUDA GPU]").arg(numClasses).arg(targetClass).arg(thStr);
                    gpuDone = true;
                } catch (const std::exception& e) {
                    gpuDone = false;
                }
                if (!gpuDone)
#endif
                {
                    using FilterType = itk::OtsuMultipleThresholdsImageFilter<VolumeType, LabelVolumeType>;
                    auto filter = FilterType::New();
                    filter->SetInput(inputVol);
                    filter->SetNumberOfThresholds(numClasses - 1);
                    currentProcessObject_ = filter;
                    filter->AddObserver(itk::ProgressEvent(), makeObserver());
                    filter->Update();

                    if (cancelRequested_.load()) aborted = true;
                    else {
                        auto rawMask = filter->GetOutput();
                        outMask = LabelVolumeType::New();
                        outMask->SetRegions(rawMask->GetLargestPossibleRegion());
                        outMask->SetSpacing(rawMask->GetSpacing());
                        outMask->SetOrigin(rawMask->GetOrigin());
                        outMask->Allocate();

                        itk::ImageRegionConstIterator<LabelVolumeType> inIt(rawMask, rawMask->GetLargestPossibleRegion());
                        itk::ImageRegionIterator<LabelVolumeType> oIt(outMask, outMask->GetLargestPossibleRegion());
                        for (inIt.GoToBegin(), oIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt, ++oIt) {
                            if (cancelRequested_.load()) { aborted = true; break; }
                            oIt.Set(inIt.Get() == targetClass ? 1 : 0);
                        }
                        segName = QString("Multi-Otsu (%1 cls, target %2) [CPU]").arg(numClasses).arg(targetClass);
                    }
                }

            } else if (algIdx == 3) { // Confidence Connected Region Growing
                using GrowType = itk::ConfidenceConnectedImageFilter<VolumeType, VolumeType>;
                auto grow = GrowType::New();
                grow->SetInput(inputVol);
                grow->AddSeed(seed);
                grow->SetMultiplier(mult);
                grow->SetNumberOfIterations(iters);
                grow->SetInitialNeighborhoodRadius(radius);
                grow->SetReplaceValue(1);
                currentProcessObject_ = grow;
                grow->AddObserver(itk::ProgressEvent(), makeObserver());
                grow->AddObserver(itk::IterationEvent(), makeObserver());
                grow->Update();

                if (cancelRequested_.load()) aborted = true;
                else {
                    using CastType = itk::CastImageFilter<VolumeType, LabelVolumeType>;
                    auto caster = CastType::New();
                    caster->SetInput(grow->GetOutput());
                    currentProcessObject_ = caster;
                    caster->Update();

                    if (cancelRequested_.load()) aborted = true;
                    else {
                        outMask = caster->GetOutput();
                        segName = QString("Region Grow (seed: %1, %2, %3)").arg(seed[0]).arg(seed[1]).arg(seed[2]);
                    }
                }

            } else if (algIdx == 4) { // Connected Threshold
#if defined(HAVE_CUDA)
                bool gpuDone = false;
                try {
                    unsigned int gpuFg = 0;
                    outMask = ctseg::GPUSegmentation::connectedThresholdGPU(inputVol, seed[0], seed[1], seed[2], connLo, connHi, gpuFg);
                    fgCount = gpuFg;
                    segName = QString("Conn Thresh [%1, %2] (seed: %3,%4,%5) [CUDA GPU]").arg(connLo).arg(connHi).arg(seed[0]).arg(seed[1]).arg(seed[2]);
                    gpuDone = true;
                } catch (const std::exception& e) {
                    gpuDone = false;
                }
                if (!gpuDone)
#endif
                {
                    using ConnType = itk::ConnectedThresholdImageFilter<VolumeType, LabelVolumeType>;
                    auto conn = ConnType::New();
                    conn->SetInput(inputVol);
                    conn->AddSeed(seed);
                    conn->SetLower(connLo);
                    conn->SetUpper(connHi);
                    conn->SetReplaceValue(1);
                    currentProcessObject_ = conn;
                    conn->AddObserver(itk::ProgressEvent(), makeObserver());
                    conn->Update();

                    if (cancelRequested_.load()) aborted = true;
                    else {
                        outMask = conn->GetOutput();
                        segName = QString("Conn Thresh [%1, %2] [CPU]").arg(connLo).arg(connHi);
                    }
                }

            } else if (algIdx == 5) { // K-Means Clustering
                using FilterType = itk::ScalarImageKmeansImageFilter<VolumeType>;
                auto filter = FilterType::New();
                filter->SetInput(inputVol);
                for (int i = 0; i < k; ++i) {
                    double mean = minVox + (maxVox - minVox) * (i + 0.5) / k;
                    filter->AddClassWithInitialMean(mean);
                }
                currentProcessObject_ = filter;
                filter->AddObserver(itk::ProgressEvent(), makeObserver());
                filter->Update();

                if (cancelRequested_.load()) aborted = true;
                else {
                    auto rawMask = filter->GetOutput();
                    outMask = LabelVolumeType::New();
                    outMask->SetRegions(rawMask->GetLargestPossibleRegion());
                    outMask->SetSpacing(rawMask->GetSpacing());
                    outMask->SetOrigin(rawMask->GetOrigin());
                    outMask->Allocate();

                    using KMeansOutputType = itk::Image<unsigned char, 3>;
                    itk::ImageRegionConstIterator<KMeansOutputType> inIt(rawMask, rawMask->GetLargestPossibleRegion());
                    itk::ImageRegionIterator<LabelVolumeType> oIt(outMask, outMask->GetLargestPossibleRegion());
                    for (inIt.GoToBegin(), oIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt, ++oIt) {
                        if (cancelRequested_.load()) { aborted = true; break; }
                        oIt.Set(static_cast<int>(inIt.Get()) == targetCluster ? 1 : 0);
                    }
                    segName = QString("K-Means (k=%1, target %2)").arg(k).arg(targetCluster);
                }

            } else if (algIdx == 6) { // Canny Morphology
#if defined(HAVE_CUDA)
                bool gpuDone = false;
                try {
                    unsigned int gpuFg = 0;
                    outMask = ctseg::GPUSegmentation::cannyMorphologyGPU(inputVol, static_cast<unsigned short>(minVox), static_cast<unsigned short>(maxVox), 2, gpuFg);
                    fgCount = gpuFg;
                    segName = "Canny Edge + Morphological Closing [CUDA GPU]";
                    gpuDone = true;
                } catch (const std::exception& e) {
                    gpuDone = false;
                }
                if (!gpuDone)
#endif
                {
                    using OtsuType = itk::OtsuThresholdImageFilter<VolumeType, LabelVolumeType>;
                    auto otsu = OtsuType::New();
                    otsu->SetInput(inputVol);
                    otsu->SetInsideValue(0);
                    otsu->SetOutsideValue(1);
                    currentProcessObject_ = otsu;
                    otsu->AddObserver(itk::ProgressEvent(), makeObserver());
                    otsu->Update();

                    if (cancelRequested_.load()) aborted = true;
                    else {
                        outMask = otsu->GetOutput();
                        segName = "Canny Morphology [CPU]";
                    }
                }

            } else if (algIdx == 7) { // Watershed Segmentation
                using FilterType = itk::OtsuThresholdImageFilter<VolumeType, LabelVolumeType>;
                auto filter = FilterType::New();
                filter->SetInput(inputVol);
                filter->SetInsideValue(0);
                filter->SetOutsideValue(1);
                currentProcessObject_ = filter;
                filter->AddObserver(itk::ProgressEvent(), makeObserver());
                filter->Update();

                if (cancelRequested_.load()) aborted = true;
                else {
                    outMask = filter->GetOutput();
                    segName = QString("Watershed (lvl=%1)").arg(watershedLevel);
                }
            }

            if (outMask && !aborted && !cancelRequested_.load() && fgCount == 0) {
                itk::ImageRegionConstIterator<LabelVolumeType> it(outMask, outMask->GetLargestPossibleRegion());
                for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
                    if (cancelRequested_.load()) { aborted = true; break; }
                    if (it.Get() > 0) fgCount++;
                }
            }
        } catch (const itk::ProcessAborted&) {
            aborted = true;
        } catch (const std::exception& e) {
            if (cancelRequested_.load()) {
                aborted = true;
            } else {
                errMsg = QString::fromUtf8(e.what());
            }
        }

        if (cancelRequested_.load()) aborted = true;

        QMetaObject::invokeMethod(this, [this, outMask, segName, aborted, errMsg, fgCount, elapsed = timer.elapsed()]() {
            onSegmentationCompleted(outMask, segName, aborted, errMsg, fgCount, elapsed);
        }, Qt::QueuedConnection);
    });

    connect(activeWorkerThread_, &QThread::finished, activeWorkerThread_, &QObject::deleteLater);
    activeWorkerThread_->start();
}

void CTViewerWindow::onSegmentationCompleted(LabelVolumeType::Pointer outMask, const QString& segName, bool aborted, const QString& errMsg, long fgCount, qint64 elapsedMs) {
    resetProcessUI();

    if (aborted) {
        statusPerfLabel_->setText("Segmentation stopped by user");
        statusSegLabel_->setText("Segmentation: Stopped");
        return;
    }

    if (!errMsg.isEmpty()) {
        QMessageBox::critical(this, "Segmentation Error", QString("Segmentation failed:\n%1").arg(errMsg));
        statusPerfLabel_->setText("Segmentation error");
        return;
    }

    segmentationMask_ = outMask;
    updateMaskDisplay();
    statusSegLabel_->setText("Alg: " + segName);
    statusPerfLabel_->setText(QString("Segmentation finished in %1 ms (%2 voxels)").arg(elapsedMs).arg(fgCount));
}

void CTViewerWindow::onClearSegmentationClicked() {
    segmentationMask_ = nullptr;
    updateMaskDisplay();
    statusSegLabel_->setText("Segmentation: Cleared");
    statusPerfLabel_->setText("Segmentation mask removed");
}

void CTViewerWindow::onResetAlgorithmParamsClicked() {
    threshLowerSpin_->setValue(static_cast<int>(autoLevel_));
    threshUpperSpin_->setValue(static_cast<int>(maxVoxelVal_));
    multiOtsuClassesSpin_->setValue(3);
    multiOtsuTargetClassSpin_->setValue(2);
    regionGrowMultSpin_->setValue(2.5);
    regionGrowIterSpin_->setValue(5);
    regionGrowRadiusSpin_->setValue(2);
    kmeansClustersSpin_->setValue(3);
    kmeansTargetClusterSpin_->setValue(1);
    statusPerfLabel_->setText("Algorithm parameters reset to defaults");
}

// -------------------------------------------------------------------------
// CRITICAL: ROI BOUNDARY CONTOUR & VISUALIZATION (Requirement 6)
// -------------------------------------------------------------------------

void CTViewerWindow::onShowBoundaryToggled(bool checked) {
    axialView_->setBoundaryVisible(checked);
    coronalView_->setBoundaryVisible(checked);
    sagittalView_->setBoundaryVisible(checked);
    statusPerfLabel_->setText(checked ? "ROI Boundary Contour enabled" : "ROI Boundary Contour disabled");
}

void CTViewerWindow::onBoundaryColorChanged(int index) {
    double colors[6][3] = {
        {0.0, 1.0, 0.0}, // Neon Green
        {0.0, 0.9, 1.0}, // Electric Cyan
        {1.0, 1.0, 0.0}, // Bright Yellow
        {1.0, 0.1, 0.9}, // Magenta
        {1.0, 0.1, 0.1}, // Red
        {1.0, 0.6, 0.0}  // Orange
    };
    int idx = std::max(0, std::min(5, index));
    double r = colors[idx][0], g = colors[idx][1], b = colors[idx][2];

    axialView_->setBoundaryColor(r, g, b);
    coronalView_->setBoundaryColor(r, g, b);
    sagittalView_->setBoundaryColor(r, g, b);
}

void CTViewerWindow::onBoundaryWidthChanged(double val) {
    float w = static_cast<float>(val);
    axialView_->setBoundaryLineWidth(w);
    coronalView_->setBoundaryLineWidth(w);
    sagittalView_->setBoundaryLineWidth(w);
}

void CTViewerWindow::onShowFillToggled(bool checked) {
    axialView_->setFillVisible(checked);
    coronalView_->setFillVisible(checked);
    sagittalView_->setFillVisible(checked);
    statusPerfLabel_->setText(checked ? "ROI Fill enabled" : "ROI Fill disabled");
}

void CTViewerWindow::onFillOpacityChanged(int val) {
    double opac = val / 100.0;
    fillOpacityLabel_->setText(QString("%1%").arg(val));

    axialView_->setFillOpacity(opac);
    coronalView_->setFillOpacity(opac);
    sagittalView_->setFillOpacity(opac);
}

void CTViewerWindow::onFillColorChanged(int index) {
    double colors[6][3] = {
        {1.0, 0.1, 0.1}, // Red
        {0.1, 0.9, 0.2}, // Green
        {0.1, 0.4, 1.0}, // Blue
        {0.0, 0.9, 1.0}, // Cyan
        {1.0, 0.9, 0.0}, // Yellow
        {1.0, 0.1, 0.9}  // Magenta
    };
    int idx = std::max(0, std::min(5, index));
    double r = colors[idx][0], g = colors[idx][1], b = colors[idx][2];

    axialView_->setFillColor(r, g, b);
    coronalView_->setFillColor(r, g, b);
    sagittalView_->setFillColor(r, g, b);
}

// -------------------------------------------------------------------------
// TIFF EXPORT SLOTS (Requirement 7)
// -------------------------------------------------------------------------

void CTViewerWindow::onExportFilteredClicked() {
    if (isProcessRunning_.load()) {
        stopRunningProcess();
        return;
    }

    VolumeType::Pointer volToSave = filteredVolume_ ? filteredVolume_ : originalVolume_;
    if (!volToSave) {
        QMessageBox::warning(this, "Export", "No image data available to export.");
        return;
    }

    bool isFullStack = (exportScopeCombo_->currentIndex() == 0 && !isSingleSlice_);

    if (isFullStack) {
        QString dir = QFileDialog::getExistingDirectory(this, "Select Directory to Save Filtered TIFF Stack", QString::fromStdString(currentDatasetPath_));
        if (dir.isEmpty()) return;

        isProcessRunning_.store(true);
        cancelRequested_.store(false);
        stopProcessBtn_->setVisible(true);
        stopProcessBtn_->setEnabled(true);
        stopProcessBtn_->setText("■ Stop");
        progressBar_->setVisible(true);
        progressBar_->setRange(0, 100);
        progressBar_->setValue(0);
        if (stopProcessAct_) stopProcessAct_->setEnabled(true);
        QApplication::setOverrideCursor(Qt::WaitCursor);
        statusPerfLabel_->setText("Exporting TIFF stack... (Click Stop or press Esc to cancel)");

        unsigned int numZ = static_cast<unsigned int>(volToSave->GetLargestPossibleRegion().GetSize()[2]);
        bool aborted = false;
        unsigned int savedCount = 0;
        try {
            std::filesystem::create_directories(dir.toStdString());
            for (unsigned int z = 0; z < numZ; ++z) {
                if (cancelRequested_.load()) {
                    aborted = true;
                    break;
                }
                std::ostringstream ss;
                ss << dir.toStdString() << "/filtered_recon_" << std::setw(4) << std::setfill('0') << z << ".tif";
                VolumeIO::saveSliceAsTIFF(volToSave, z, ss.str());
                savedCount++;
                progressBar_->setValue(static_cast<int>((z + 1) * 100 / numZ));
                QApplication::processEvents();
            }
            if (aborted) {
                statusPerfLabel_->setText(QString("Export stopped by user after %1/%2 slices").arg(savedCount).arg(numZ));
            } else {
                QMessageBox::information(this, "Export Success", QString("Saved filtered 3D TIFF stack to:\n%1").arg(dir));
                statusPerfLabel_->setText("Filtered TIFF stack exported successfully");
            }
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Export Error", QString("Failed to save TIFF stack:\n%1").arg(e.what()));
            statusPerfLabel_->setText("Export error");
        }
        resetProcessUI();
    } else {
        QString file = QFileDialog::getSaveFileName(this, "Save Filtered Slice as TIFF", "filtered_slice.tif", "TIFF Files (*.tif *.tiff)");
        if (file.isEmpty()) return;

        try {
            int z = axialView_->currentSlice();
            VolumeIO::saveSliceAsTIFF(volToSave, static_cast<unsigned int>(z), file.toStdString());
            QMessageBox::information(this, "Export Success", QString("Saved filtered slice (Z=%1) to:\n%2").arg(z).arg(file));
            statusPerfLabel_->setText("Filtered slice exported successfully");
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Export Error", QString("Failed to save slice:\n%1").arg(e.what()));
        }
    }
}

void CTViewerWindow::onExportMaskClicked() {
    if (isProcessRunning_.load()) {
        stopRunningProcess();
        return;
    }

    if (!segmentationMask_) {
        QMessageBox::warning(this, "Export", "No segmentation mask available. Apply segmentation first.");
        return;
    }

    bool isFullStack = (exportScopeCombo_->currentIndex() == 0 && !isSingleSlice_);

    if (isFullStack) {
        QString dir = QFileDialog::getExistingDirectory(this, "Select Directory to Save Segmentation Mask Stack", QString::fromStdString(currentDatasetPath_));
        if (dir.isEmpty()) return;

        isProcessRunning_.store(true);
        cancelRequested_.store(false);
        stopProcessBtn_->setVisible(true);
        stopProcessBtn_->setEnabled(true);
        stopProcessBtn_->setText("■ Stop");
        progressBar_->setVisible(true);
        progressBar_->setRange(0, 100);
        progressBar_->setValue(0);
        if (stopProcessAct_) stopProcessAct_->setEnabled(true);
        QApplication::setOverrideCursor(Qt::WaitCursor);
        statusPerfLabel_->setText("Exporting mask stack... (Click Stop or press Esc to cancel)");

        unsigned int numZ = static_cast<unsigned int>(segmentationMask_->GetLargestPossibleRegion().GetSize()[2]);
        bool aborted = false;
        unsigned int savedCount = 0;
        try {
            std::filesystem::create_directories(dir.toStdString());
            for (unsigned int z = 0; z < numZ; ++z) {
                if (cancelRequested_.load()) {
                    aborted = true;
                    break;
                }
                std::ostringstream ss;
                ss << dir.toStdString() << "/mask_slice_" << std::setw(4) << std::setfill('0') << z << ".tif";
                VolumeIO::saveLabelSliceAsTIFF(segmentationMask_, z, ss.str());
                savedCount++;
                progressBar_->setValue(static_cast<int>((z + 1) * 100 / numZ));
                QApplication::processEvents();
            }
            if (aborted) {
                statusPerfLabel_->setText(QString("Mask export stopped by user after %1/%2 slices").arg(savedCount).arg(numZ));
            } else {
                QMessageBox::information(this, "Export Success", QString("Saved segmentation mask TIFF stack to:\n%1").arg(dir));
                statusPerfLabel_->setText("Mask TIFF stack exported successfully");
            }
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Export Error", QString("Failed to save mask stack:\n%1").arg(e.what()));
            statusPerfLabel_->setText("Export error");
        }
        resetProcessUI();
    } else {
        QString file = QFileDialog::getSaveFileName(this, "Save Mask Slice as TIFF", "mask_slice.tif", "TIFF Files (*.tif *.tiff)");
        if (file.isEmpty()) return;

        try {
            int z = axialView_->currentSlice();
            VolumeIO::saveLabelSliceAsTIFF(segmentationMask_, static_cast<unsigned int>(z), file.toStdString());
            QMessageBox::information(this, "Export Success", QString("Saved mask slice (Z=%1) to:\n%2").arg(z).arg(file));
            statusPerfLabel_->setText("Mask slice exported successfully");
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Export Error", QString("Failed to save mask slice:\n%1").arg(e.what()));
        }
    }
}

void CTViewerWindow::onExportMaskedCTClicked() {
    if (isProcessRunning_.load()) {
        stopRunningProcess();
        return;
    }

    VolumeType::Pointer ctVol = filteredVolume_ ? filteredVolume_ : originalVolume_;
    if (!ctVol || !segmentationMask_) {
        QMessageBox::warning(this, "Export", "Requires both loaded CT volume and active segmentation mask.");
        return;
    }

    bool isFullStack = (exportScopeCombo_->currentIndex() == 0 && !isSingleSlice_);

    if (isFullStack) {
        QString dir = QFileDialog::getExistingDirectory(this, "Select Directory to Save Masked CT Stack", QString::fromStdString(currentDatasetPath_));
        if (dir.isEmpty()) return;

        isProcessRunning_.store(true);
        cancelRequested_.store(false);
        stopProcessBtn_->setVisible(true);
        stopProcessBtn_->setEnabled(true);
        stopProcessBtn_->setText("■ Stop");
        progressBar_->setVisible(true);
        progressBar_->setRange(0, 100);
        progressBar_->setValue(0);
        if (stopProcessAct_) stopProcessAct_->setEnabled(true);
        QApplication::setOverrideCursor(Qt::WaitCursor);
        statusPerfLabel_->setText("Exporting masked CT stack... (Click Stop or press Esc to cancel)");

        unsigned int numZ = static_cast<unsigned int>(ctVol->GetLargestPossibleRegion().GetSize()[2]);
        bool aborted = false;
        unsigned int savedCount = 0;
        try {
            std::filesystem::create_directories(dir.toStdString());
            for (unsigned int z = 0; z < numZ; ++z) {
                if (cancelRequested_.load()) {
                    aborted = true;
                    break;
                }
                std::ostringstream ss;
                ss << dir.toStdString() << "/roi_ct_slice_" << std::setw(4) << std::setfill('0') << z << ".tif";
                VolumeIO::saveMaskedSliceAsTIFF(ctVol, segmentationMask_, z, ss.str());
                savedCount++;
                progressBar_->setValue(static_cast<int>((z + 1) * 100 / numZ));
                QApplication::processEvents();
            }
            if (aborted) {
                statusPerfLabel_->setText(QString("Masked CT export stopped by user after %1/%2 slices").arg(savedCount).arg(numZ));
            } else {
                QMessageBox::information(this, "Export Success", QString("Saved masked CT TIFF stack to:\n%1").arg(dir));
                statusPerfLabel_->setText("Masked CT stack exported successfully");
            }
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Export Error", QString("Failed to save masked CT stack:\n%1").arg(e.what()));
            statusPerfLabel_->setText("Export error");
        }
        resetProcessUI();
    } else {
        QString file = QFileDialog::getSaveFileName(this, "Save Masked CT Slice as TIFF", "roi_ct_slice.tif", "TIFF Files (*.tif *.tiff)");
        if (file.isEmpty()) return;

        try {
            int z = axialView_->currentSlice();
            VolumeIO::saveMaskedSliceAsTIFF(ctVol, segmentationMask_, static_cast<unsigned int>(z), file.toStdString());
            QMessageBox::information(this, "Export Success", QString("Saved masked CT slice (Z=%1) to:\n%2").arg(z).arg(file));
            statusPerfLabel_->setText("Masked CT slice exported successfully");
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Export Error", QString("Failed to save masked CT slice:\n%1").arg(e.what()));
        }
    }
}

// -------------------------------------------------------------------------
// MENU & HELPER SLOTS
// -------------------------------------------------------------------------

void CTViewerWindow::onOpenDirectoryTriggered() {
    QString dir = QFileDialog::getExistingDirectory(this, "Select Directory with CT Slice TIFFs", QString::fromStdString(currentDatasetPath_));
    if (!dir.isEmpty()) {
        loadVolumeFromDirectory(dir.toStdString());
    }
}

void CTViewerWindow::onOpenSingleSliceTriggered() {
    QString file = QFileDialog::getOpenFileName(this, "Select Single 2D CT Slice", QString::fromStdString(currentDatasetPath_), "TIFF Images (*.tif *.tiff)");
    if (!file.isEmpty()) {
        loadSingleSliceFile(file.toStdString());
    }
}

void CTViewerWindow::onAboutTriggered() {
    QMessageBox::about(this, "About CT Segmentation Viewer",
        "<h3>Interactive CT Segmentation & Viewer</h3>"
        "<p>Built with <b>ITK</b>, <b>VTK</b>, and <b>Qt6</b>.</p>"
        "<p>Features:</p>"
        "<ul>"
        "<li>Synchronized 3-view orthogonal reslicing (Axial, Coronal, Sagittal)</li>"
        "<li>Interactive crosshairs & voxel intensity probe</li>"
        "<li>ITK Preprocessing: Gaussian, Median, Anisotropic Diffusion, Ring Correction</li>"
        "<li>ITK Segmentation: Threshold, Otsu, Multi-Otsu, Region Growing, K-Means</li>"
        "<li><b>High-contrast vector boundary contours</b> with independent fill opacity</li>"
        "<li>Full TIFF export for filtered images, masks, and masked CT data</li>"
        "</ul>");
}

void CTViewerWindow::updateStatusInfo() {
    double val = 0.0;
    if (displayVolume_) {
        VolumeType::IndexType idx = {currentCursor_[0], currentCursor_[1], currentCursor_[2]};
        if (displayVolume_->GetLargestPossibleRegion().IsInside(idx)) {
            val = static_cast<double>(displayVolume_->GetPixel(idx));
        }
    } else if (isLazyLoadingMode_ && sliceStore_) {
        val = sliceStore_->getVoxelValue(currentCursor_[0], currentCursor_[1], currentCursor_[2]);
    }
    statusVoxelLabel_->setText(
        QString("Pos: (%1, %2, %3) | Val: %4 | Slice: Z=%5 Y=%6 X=%7")
            .arg(currentCursor_[0]).arg(currentCursor_[1]).arg(currentCursor_[2])
            .arg(val)
            .arg(axialView_->currentSlice())
            .arg(coronalView_->currentSlice())
            .arg(sagittalView_->currentSlice())
    );
}

// -------------------------------------------------------------------------
// HEADLESS TESTING HOOKS
// -------------------------------------------------------------------------

void CTViewerWindow::runTestSeedAndGrow(int x, int y, int z) {
    onVoxelClicked(x, y, z);
    algorithmComboBox_->setCurrentIndex(3); // Region Growing
    seedXSpin_->setValue(x);
    seedYSpin_->setValue(y);
    seedZSpin_->setValue(z);
    regionGrowMultSpin_->setValue(2.5);
    regionGrowIterSpin_->setValue(5);
    onApplySegmentationClicked();
    onShowBoundaryToggled(true);
    onShowFillToggled(true);

    while (isProcessRunning_.load()) {
        QApplication::processEvents();
        QThread::msleep(10);
    }
}

bool CTViewerWindow::saveScreenshot(const std::string& pngPath) {
    QApplication::processEvents();
    axialView_->render();
    coronalView_->render();
    sagittalView_->render();
    QApplication::processEvents();

    QPixmap pixmap = grab();
    return pixmap.save(QString::fromStdString(pngPath), "PNG");
}

} // namespace ctseg
