#pragma once

#include "../volume/VolumeTypes.hpp"
#include "OrthoSliceView.hpp"

#include <QMainWindow>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QSlider>
#include <QCheckBox>
#include <QPushButton>
#include <QRadioButton>
#include <QButtonGroup>
#include <QGroupBox>
#include <QStackedWidget>
#include <QSplitter>
#include <QLabel>
#include <QProgressBar>
#include <QStatusBar>

#include <itkImageToVTKImageFilter.h>

namespace ctseg {

class CTViewerWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit CTViewerWindow(QWidget* parent = nullptr);
    ~CTViewerWindow() override = default;

    bool loadVolumeFromDirectory(const std::string& dirPath, size_t maxSlices = 0);
    bool loadSingleSliceFile(const std::string& filePath);

    // Headless testing hooks
    void runTestSeedAndGrow(int x, int y, int z);
    bool saveScreenshot(const std::string& pngPath);

private slots:
    // Menu Actions
    void onOpenDirectoryTriggered();
    void onOpenSingleSliceTriggered();
    void onAboutTriggered();

    // View Navigation & Layout
    void onSliceChanged(int orientation, int sliceIndex);
    void onVoxelClicked(int x, int y, int z);
    void onVoxelHovered(int x, int y, int z, double intensity);
    void onViewLayoutChanged(int index);
    void onToggleMaximizeView(int orientation);
    void onResetCamerasClicked();
    void onCrosshairToggled(bool checked);

    // Window / Level
    void onWindowSliderChanged(int val);
    void onLevelSliderChanged(int val);
    void onPresetChanged(int index);
    void onAutoWindowLevelClicked();

    // Preprocessing Filter
    void onFilterSelectionChanged(int index);
    void onApplyFilterClicked();
    void onResetFilterClicked();
    void onCompareFilterToggled(bool showOriginal);

    // Segmentation Algorithm
    void onAlgorithmSelectionChanged(int index);
    void onApplySegmentationClicked();
    void onClearSegmentationClicked();
    void onResetAlgorithmParamsClicked();

    // ROI Visualization & Boundary
    void onShowBoundaryToggled(bool checked);
    void onBoundaryColorChanged(int index);
    void onBoundaryWidthChanged(double val);
    void onShowFillToggled(bool checked);
    void onFillOpacityChanged(int val);
    void onFillColorChanged(int index);

    // TIFF Export
    void onExportFilteredClicked();
    void onExportMaskClicked();
    void onExportMaskedCTClicked();

private:
    void setupUI();
    void setupMenuBar();
    void setupControlPanel(QWidget* panel);
    void setupViewArea(QWidget* area);
    void setupStatusBar();
    void applyDarkTheme();

    void updateVolumeDisplay();
    void updateMaskDisplay();
    void computePercentileWindowLevel(double loPct, double hiPct, double& outWindow, double& outLevel);
    void updateStatusInfo();

    // Pipeline Data
    std::string currentDatasetPath_;
    bool isSingleSlice_ = false;

    VolumeType::Pointer originalVolume_;
    VolumeType::Pointer filteredVolume_;
    VolumeType::Pointer displayVolume_; // points to original or filtered
    LabelVolumeType::Pointer segmentationMask_;

    using ToVTKVolumeType = itk::ImageToVTKImageFilter<VolumeType>;
    itk::SmartPointer<ToVTKVolumeType> volumeToVTK_;
    vtkSmartPointer<vtkImageData> grayVTKData_;

    using ToVTKLabelType = itk::ImageToVTKImageFilter<LabelVolumeType>;
    itk::SmartPointer<ToVTKLabelType> maskToVTK_;
    vtkSmartPointer<vtkImageData> maskVTKData_;

    // Views
    OrthoSliceView* axialView_ = nullptr;
    OrthoSliceView* coronalView_ = nullptr;
    OrthoSliceView* sagittalView_ = nullptr;

    QSplitter* mainSplitter_ = nullptr;
    QSplitter* viewsSplitter_ = nullptr;
    QStackedWidget* viewStack_ = nullptr;
    QWidget* threeViewsContainer_ = nullptr;

    // GUI Controls - Dataset & Contrast
    QLabel* datasetInfoLabel_ = nullptr;
    QSlider* windowSlider_ = nullptr;
    QSlider* levelSlider_ = nullptr;
    QSpinBox* windowSpinBox_ = nullptr;
    QSpinBox* levelSpinBox_ = nullptr;
    QComboBox* presetComboBox_ = nullptr;
    QCheckBox* crosshairsCheckBox_ = nullptr;

    // GUI Controls - Preprocessing
    QComboBox* filterComboBox_ = nullptr;
    QStackedWidget* filterParamsStack_ = nullptr;
    QDoubleSpinBox* gaussianSigmaSpin_ = nullptr;
    QSpinBox* medianRadiusSpin_ = nullptr;
    QSpinBox* anisoIterSpin_ = nullptr;
    QDoubleSpinBox* anisoConductanceSpin_ = nullptr;
    QDoubleSpinBox* anisoTimeStepSpin_ = nullptr;
    QSpinBox* ringRadiusSpin_ = nullptr;
    QPushButton* applyFilterBtn_ = nullptr;
    QPushButton* resetFilterBtn_ = nullptr;
    QCheckBox* compareFilterCheck_ = nullptr;

    // GUI Controls - Segmentation
    QComboBox* algorithmComboBox_ = nullptr;
    QStackedWidget* algorithmParamsStack_ = nullptr;
    QRadioButton* segSourceFilteredRadio_ = nullptr;
    QRadioButton* segSourceOriginalRadio_ = nullptr;

    QSpinBox* threshLowerSpin_ = nullptr;
    QSpinBox* threshUpperSpin_ = nullptr;
    QSpinBox* multiOtsuClassesSpin_ = nullptr;
    QSpinBox* multiOtsuTargetClassSpin_ = nullptr;
    QSpinBox* seedXSpin_ = nullptr;
    QSpinBox* seedYSpin_ = nullptr;
    QSpinBox* seedZSpin_ = nullptr;
    QDoubleSpinBox* regionGrowMultSpin_ = nullptr;
    QSpinBox* regionGrowIterSpin_ = nullptr;
    QSpinBox* regionGrowRadiusSpin_ = nullptr;
    QSpinBox* connThreshLowerSpin_ = nullptr;
    QSpinBox* connThreshUpperSpin_ = nullptr;
    QSpinBox* kmeansClustersSpin_ = nullptr;
    QSpinBox* kmeansTargetClusterSpin_ = nullptr;
    QDoubleSpinBox* cannyLowPctSpin_ = nullptr;
    QDoubleSpinBox* cannyHighPctSpin_ = nullptr;
    QSpinBox* cannyRadiusSpin_ = nullptr;
    QDoubleSpinBox* watershedLevelSpin_ = nullptr;

    QPushButton* applySegBtn_ = nullptr;
    QPushButton* clearSegBtn_ = nullptr;

    // GUI Controls - ROI Boundary & Visualization
    QCheckBox* showBoundaryCheck_ = nullptr;
    QComboBox* boundaryColorCombo_ = nullptr;
    QDoubleSpinBox* boundaryWidthSpin_ = nullptr;
    QCheckBox* showFillCheck_ = nullptr;
    QSlider* fillOpacitySlider_ = nullptr;
    QLabel* fillOpacityLabel_ = nullptr;
    QComboBox* fillColorCombo_ = nullptr;

    // GUI Controls - Export
    QComboBox* exportScopeCombo_ = nullptr;

    // Status bar labels
    QLabel* statusVoxelLabel_ = nullptr;
    QLabel* statusFilterLabel_ = nullptr;
    QLabel* statusSegLabel_ = nullptr;
    QLabel* statusPerfLabel_ = nullptr;
    QProgressBar* progressBar_ = nullptr;

    // Active state
    int currentCursor_[3] = {0, 0, 0};
    double currentIntensity_ = 0.0;
    double autoWindow_ = 1000.0;
    double autoLevel_ = 500.0;
    double minVoxelVal_ = 0.0;
    double maxVoxelVal_ = 65535.0;
};

} // namespace ctseg
