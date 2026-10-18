#pragma once
#include "../core/ISegmenter.hpp"
#include <itkLabelOverlapMeasuresImageFilter.h>
#include <string>
#include <iostream>
#include <iomanip>

namespace ctseg {

struct EvalMetrics {
    double dice = 0.0;
    double jaccard = 0.0;
    double volumeSimilarity = 0.0;
    double falseNegativeError = 0.0;
    double falsePositiveError = 0.0;
};

// Wraps ITK's LabelOverlapMeasuresImageFilter to compute standard
// segmentation-quality metrics against ground truth (Plan of Action
// point 13: quantitative comparison; point 14: ground-truth strategy).
// NOTE: this compares whole label images directly, which assumes label
// IDs already correspond between prediction and ground truth (e.g. both
// use 0=background, 1=class A, ...). For methods like watershed/k-means
// where label IDs are arbitrary, remap to a common class scheme before
// evaluating, or evaluate a single binarized class at a time.
class Evaluator {
public:
    static EvalMetrics compare(LabelImageType::Pointer predicted, LabelImageType::Pointer groundTruth) {
        using FilterType = itk::LabelOverlapMeasuresImageFilter<LabelImageType>;
        auto filter = FilterType::New();
        filter->SetSourceImage(groundTruth);
        filter->SetTargetImage(predicted);
        filter->Update();

        EvalMetrics m;
        m.dice = filter->GetDiceCoefficient();
        m.jaccard = filter->GetJaccardCoefficient();
        m.volumeSimilarity = filter->GetVolumeSimilarity();
        m.falseNegativeError = filter->GetFalseNegativeError();
        m.falsePositiveError = filter->GetFalsePositiveError();
        return m;
    }

    static void printMetrics(const std::string& methodName, const EvalMetrics& m) {
        std::cout << "  [" << methodName << "] Dice=" << std::fixed << std::setprecision(4) << m.dice
                  << " Jaccard=" << m.jaccard
                  << " VolumeSimilarity=" << m.volumeSimilarity
                  << " FNE=" << m.falseNegativeError
                  << " FPE=" << m.falsePositiveError << "\n";
    }
};

} // namespace ctseg
