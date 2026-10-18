#pragma once
#include "../../core/ISegmenter.hpp"
#include <stdexcept>

namespace ctseg {

// ML-based segmentation scaffold — NOT a trained implementation.
//
// This class exists to reserve the ISegmenter slot for a deep-learning
// method (e.g. a U-Net) and to document exactly what's needed to make it
// real, per the Plan of Action point 5 ("ML-based segmentation last,
// since it requires labeled data and infrastructure").
//
// Intended real implementation path, matching the project's C++-first
// design (point 8):
//   1. Train in Python once, using MONAI/PyTorch, on a labeled subset of
//      this dataset (ground truth availability is still open — Plan of
//      Action point 14).
//   2. Export the trained model to TorchScript (.pt) or ONNX (.onnx).
//   3. Run INFERENCE ONLY in C++ here, via either:
//        - LibTorch's C++ API (torch::jit::load + module.forward), or
//        - ONNX Runtime's C++ API (Ort::Session)
//      Both avoid any Python dependency at runtime, keeping the deployed
//      module pure C++ as intended.
//   4. Preprocessing must exactly match what the model was trained with
//      (same normalization/resampling) — encode that alongside the
//      exported model file, not just in code comments.
//
// Calling segment() before that integration is wired up fails loudly
// rather than silently returning a fake/empty result.
class MLSegmenter : public ISegmenter {
public:
    std::string name() const override { return "MLSegmentation (scaffold)"; }

    SegmentationResult segment(ImageType::Pointer /*input*/, const ParamMap& /*params*/) override {
        throw std::runtime_error(
            "MLSegmenter is a scaffold only: no trained model is wired in yet. "
            "Requires (1) labeled training data [Plan of Action point 14, still open], "
            "(2) a model trained/exported via MONAI/PyTorch -> TorchScript or ONNX, and "
            "(3) LibTorch or ONNX Runtime C++ linked into this project for inference. "
            "See the comments in this file for the intended integration path.");
    }
};

} // namespace ctseg
