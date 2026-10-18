#pragma once
#include "ROIProcessor.hpp"
#include <itkImageToVTKImageFilter.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkImageActor.h>
#include <vtkImageSliceMapper.h>
#include <vtkImageProperty.h>
#include <vtkLookupTable.h>
#include <vtkCellPicker.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <array>
#include <memory>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace ctseg {

// One orthogonal view (axial/coronal/sagittal): a grayscale slice actor
// plus a translucent-red ROI overlay actor, both fed by the SAME 3D
// vtkImageData and sliced differently via vtkImageSliceMapper's
// orientation setting - so updating the overlay data once updates all
// three views' overlays automatically.
struct OrthoView {
    vtkSmartPointer<vtkRenderer> renderer;
    vtkSmartPointer<vtkImageSliceMapper> grayMapper;
    vtkSmartPointer<vtkImageActor> grayActor;
    vtkSmartPointer<vtkImageSliceMapper> overlayMapper;
    vtkSmartPointer<vtkImageActor> overlayActor;
    vtkSmartPointer<vtkTextActor> label;
    int orientation = 2; // axis the slice is perpendicular to: 0=X(sagittal) 1=Y(coronal) 2=Z(axial)
    int axisIndex = 2;   // unused, kept for compatibility
    std::string title;
};

// Drives the 3-pane interactive window.
//
//   Selecting a region (two ways):
//     * left-click            : set a seed point (region grow starts there)
//     * k, then left-drag     : draw a CIRCLE - press at the centre, drag out
//                               to the radius, release. Becomes a sphere ROI
//                               so it shows as a circle in all three views.
//   Filters on the selected region:
//     r = region grow (inside the circle if one was drawn)
//     e = erode   d = dilate   o = opening   c = closing
//     + / - = morphology radius      x = clear the ROI in the working window
//   s = save ROI as a TIFF stack (./roi_output/)      q / Esc = quit
//   mouse wheel = scroll the slice of the view under the cursor
//
// Every click/circle also moves the other two views to that point, so the
// ROI is visible in axial, coronal and sagittal at once.
class InteractiveViewerStyle : public vtkInteractorStyleImage {
public:
    static InteractiveViewerStyle* New();
    vtkTypeMacro(InteractiveViewerStyle, vtkInteractorStyleImage);

    void Initialize(VolumeType::Pointer volume, std::array<OrthoView, 3>* views,
                     vtkRenderWindow* renderWindow, vtkImageData* grayVTK,
                     VolumeType::SizeType cropHalfSize) {
        volume_ = volume;
        views_ = views;
        renderWindow_ = renderWindow;
        grayVTK_ = grayVTK;
        proc_ = std::make_unique<ROIProcessor>(volume_, cropHalfSize);
        picker_ = vtkSmartPointer<vtkCellPicker>::New();
        picker_->SetTolerance(0.005);
        status_ = "click=seed  k=circle  r e d o c  s=save";
        rebuildOverlay();
        refreshLabels();
    }

    // VTK's default style binds single letters to its own actions
    // (notably 'e' and 'q' = EXIT the application, 'r' = reset window/level,
    // 'w'/'s' = wireframe/surface, 'f' = fly-to, 'p' = pick). Those fire in
    // OnChar AFTER our OnKeyPress, so pressing 'e' for erode ran our erode
    // and then quit the whole viewer (found by driving the real window with
    // synthetic key events). We swallow every built-in char binding and
    // implement quit ourselves in OnKeyPress.
    void OnChar() override {}

    void OnLeftButtonDown() override {
        int* pos = this->Interactor->GetEventPosition();
        vtkRenderer* renderer = this->Interactor->FindPokedRenderer(pos[0], pos[1]);
        int viewIdx = findViewIndex(renderer);
        if (viewIdx < 0) { Superclass::OnLeftButtonDown(); return; }

        // Pick() returns 0 when the click missed the image (dark margin);
        // without this check the stale pick position (0,0,0) was used.
        if (!picker_->Pick(pos[0], pos[1], 0, renderer)) {
            statusMessage("click on the image");
            return;
        }
        double world[3];
        picker_->GetPickPosition(world);
        VolumeType::IndexType idx;
        if (!worldToIndex(world, idx)) {
            statusMessage("click was outside the volume");
            return;
        }

        syncViews(idx);

        if (circleMode_) {
            dragging_ = true;
            dragView_ = viewIdx;
            dragCenterIdx_ = idx;
            for (int i = 0; i < 3; ++i) dragCenterWorld_[i] = world[i];
            renderer->SetWorldPoint(world[0], world[1], world[2], 1.0);
            renderer->WorldToDisplay();
            dragDepth_ = renderer->GetDisplayPoint()[2];
            statusMessage("circle: drag to set radius", false);
            return;
        }

        proc_->setSeed(idx);
        statusMessage("seed (" + std::to_string(idx[0]) + "," + std::to_string(idx[1]) + "," + std::to_string(idx[2]) + ")");
    }

    void OnMouseMove() override {
        if (dragging_) {
            double r = currentDragRadius();
            statusMessage("circle radius = " + std::to_string(static_cast<int>(r)) + " voxels", false);
            return;
        }
        Superclass::OnMouseMove();
    }

    void OnLeftButtonUp() override {
        if (dragging_) {
            dragging_ = false;
            double r = currentDragRadius();
            if (r < 2.0) {
                statusMessage("circle too small - drag further out");
                return;
            }
            proc_->setBallROI(dragCenterIdx_, r);
            rebuildOverlay();
            statusMessage("circle r=" + std::to_string(static_cast<int>(r)) + " - press r / e d o c");
            return;
        }
        Superclass::OnLeftButtonUp();
    }

    // Headless test hook (see main_viewer.cpp --test-roi): exercises the
    // exact same ROIProcessor + overlay-rebuild path a real mouse-click
    // plus 'r' key-press would, without needing an actual mouse.
    void TestSetSeedAndGrow(VolumeType::IndexType idx) {
        proc_->setSeed(idx);
        bool ok = proc_->regionGrow();
        std::cout << "[Viewer][test] regionGrow from (" << idx[0] << "," << idx[1] << "," << idx[2]
                  << ") -> " << (ok ? "success" : "failed") << "\n";
        if (ok) { proc_->dilate(2); syncViews(idx); rebuildOverlay(); }
    }

    void OnMouseWheelForward() override { scrollActiveView(+1); }
    void OnMouseWheelBackward() override { scrollActiveView(-1); }

    void OnKeyPress() override {
        std::string key = this->Interactor->GetKeySym();
        if (key == "q" || key == "Escape") { this->Interactor->ExitCallback(); }
        else if (key == "k") {
            circleMode_ = !circleMode_;
            statusMessage(circleMode_ ? "CIRCLE mode: drag in a view" : "SEED mode: click in a view");
        }
        else if (key == "r") { runOp([&] { return proc_->regionGrow(); }, "region grow"); }
        else if (key == "e") { runOp([&] { return proc_->erode(morphRadius_); }, "erode"); }
        else if (key == "d") { runOp([&] { return proc_->dilate(morphRadius_); }, "dilate"); }
        else if (key == "o") { runOp([&] { return proc_->open(morphRadius_); }, "open"); }
        else if (key == "c") { runOp([&] { return proc_->close(morphRadius_); }, "close"); }
        else if (key == "x") { proc_->resetCrop(); rebuildOverlay(); statusMessage("ROI cleared"); }
        else if (key == "s") {
            proc_->saveAsTiffStack("./roi_output");
            statusMessage("saved TIFFs to ./roi_output/");
        }
        else if (key == "plus" || key == "equal") {
            morphRadius_++; statusMessage("morphology radius = " + std::to_string(morphRadius_));
        }
        else if (key == "minus") {
            if (morphRadius_ > 1) morphRadius_--;
            statusMessage("morphology radius = " + std::to_string(morphRadius_));
        }
        else { Superclass::OnKeyPress(); }
    }

private:
    template <typename Fn>
    void runOp(Fn&& fn, const std::string& opName) {
        bool ok = fn();
        if (ok) { rebuildOverlay(); statusMessage(opName + " applied"); }
        else { statusMessage(opName + ": select a region first (click or k+drag)"); }
    }

    int findViewIndex(vtkRenderer* renderer) const {
        for (int i = 0; i < 3; ++i) if ((*views_)[i].renderer.GetPointer() == renderer) return i;
        return -1;
    }

    bool worldToIndex(const double w[3], VolumeType::IndexType& idx) const {
        VolumeType::PointType p;
        p[0] = w[0]; p[1] = w[1]; p[2] = w[2];
        return volume_->TransformPhysicalPointToIndex(p, idx);
    }

    // Radius (in voxels) of the circle being dragged, measured in the plane
    // of the view where the drag started.
    double currentDragRadius() {
        int* pos = this->Interactor->GetEventPosition();
        vtkRenderer* r = (*views_)[dragView_].renderer;
        r->SetDisplayPoint(pos[0], pos[1], dragDepth_);
        r->DisplayToWorld();
        double* wp = r->GetWorldPoint();
        double cur[3] = {wp[0] / wp[3], wp[1] / wp[3], wp[2] / wp[3]};
        auto sp = volume_->GetSpacing();
        int o = (*views_)[dragView_].orientation; // axis perpendicular to the view
        double sumSq = 0.0;
        for (int a = 0; a < 3; ++a) {
            if (a == o) continue;
            double d = (cur[a] - dragCenterWorld_[a]) / sp[a];
            sumSq += d * d;
        }
        return std::sqrt(sumSq);
    }

    // Move all three views to the slices passing through idx, so a region
    // selected in one view is visible in the other two.
    void syncViews(const VolumeType::IndexType& idx) {
        for (auto& v : *views_) {
            int s = static_cast<int>(idx[v.orientation]);
            s = std::max(v.grayMapper->GetSliceNumberMinValue(), std::min(v.grayMapper->GetSliceNumberMaxValue(), s));
            v.grayMapper->SetSliceNumber(s);
            v.overlayMapper->SetSliceNumber(s);
        }
        refreshLabels();
    }

    void scrollActiveView(int delta) {
        int* pos = this->Interactor->GetEventPosition();
        vtkRenderer* renderer = this->Interactor->FindPokedRenderer(pos[0], pos[1]);
        int viewIdx = findViewIndex(renderer);
        if (viewIdx < 0) return;

        auto& v = (*views_)[viewIdx];
        int slice = v.grayMapper->GetSliceNumber() + delta;
        int minS = v.grayMapper->GetSliceNumberMinValue();
        int maxS = v.grayMapper->GetSliceNumberMaxValue();
        slice = std::max(minS, std::min(maxS, slice));
        v.grayMapper->SetSliceNumber(slice);
        v.overlayMapper->SetSliceNumber(slice);
        std::cout << "[Viewer] " << v.title << " slice " << slice << "/" << maxS << "\n";
        refreshLabels();
    }

    // Recomputes the RGBA overlay volume from the current ROI mask and
    // pushes it into the shared overlay vtkImageData used by all 3 views.
    void rebuildOverlay() {
        // The overlay is the 1-byte-per-voxel label mask itself, coloured at
        // render time by a lookup table (0 = transparent, 1 = translucent red)
        // for only the slice being displayed. An earlier version converted the
        // whole mask to a full-volume RGBA image on every edit - 4 bytes per
        // voxel, i.e. ~3.6 GB for a 900-slice volume, rebuilt per keypress.
        using ToVTKType = itk::ImageToVTKImageFilter<LabelVolumeType>;
        maskToVTK_ = ToVTKType::New();
        maskToVTK_->SetInput(proc_->getMask());
        maskToVTK_->Update();
        overlayVTK_ = maskToVTK_->GetOutput();

        if (!lut_) {
            lut_ = vtkSmartPointer<vtkLookupTable>::New();
            lut_->SetNumberOfTableValues(2);
            lut_->SetTableValue(0, 0, 0, 0, 0.0);        // label 0 -> fully transparent
            lut_->SetTableValue(1, 1.0, 0.1, 0.1, 0.45); // label 1 -> translucent red
            lut_->SetRange(0, 1);
            lut_->Build();
        }
        for (auto& v : *views_) {
            v.overlayActor->GetProperty()->SetLookupTable(lut_);
            v.overlayActor->GetProperty()->UseLookupTableScalarRangeOn();
            v.overlayMapper->SetInputData(overlayVTK_);
        }
        renderWindow_->Render();
    }

    // Each pane's text: view title, current slice, mode, then the status line.
    void refreshLabels() {
        for (auto& v : *views_) {
            std::string t = v.title + "   [slice " + std::to_string(v.grayMapper->GetSliceNumber()) +
                            "/" + std::to_string(v.grayMapper->GetSliceNumberMaxValue()) + "]";
            if (circleMode_) t += "   CIRCLE MODE";
            t += "\n" + status_;
            v.label->SetInput(t.c_str());
        }
        renderWindow_->Render();
    }

    void statusMessage(const std::string& msg, bool print = true) {
        status_ = msg;
        if (print) std::cout << "[Viewer] " << msg << "\n";
        refreshLabels();
    }

    VolumeType::Pointer volume_;
    std::array<OrthoView, 3>* views_ = nullptr;
    vtkRenderWindow* renderWindow_ = nullptr;
    vtkImageData* grayVTK_ = nullptr;
    vtkSmartPointer<vtkImageData> overlayVTK_;
    itk::ImageToVTKImageFilter<LabelVolumeType>::Pointer maskToVTK_; // keeps the mask buffer alive for VTK
    vtkSmartPointer<vtkLookupTable> lut_;
    vtkSmartPointer<vtkCellPicker> picker_;
    std::unique_ptr<ROIProcessor> proc_;
    unsigned int morphRadius_ = 1;
    std::string status_;

    bool circleMode_ = false;
    bool dragging_ = false;
    int dragView_ = 0;
    double dragCenterWorld_[3] = {0, 0, 0};
    double dragDepth_ = 0.0;
    VolumeType::IndexType dragCenterIdx_{};
};

vtkStandardNewMacro(InteractiveViewerStyle);

} // namespace ctseg
