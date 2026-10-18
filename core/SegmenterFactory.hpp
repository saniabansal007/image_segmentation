#pragma once
#include "ISegmenter.hpp"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <stdexcept>
#include <map>

namespace ctseg {

// Central registry so main.cpp (or any future driver, e.g. the larger
// unified CT application) can list and run methods generically by name,
// without knowing about each concrete segmenter class. New methods are
// added in ONE place (registerAll below) per the modular design goal.
class SegmenterFactory {
public:
    using Creator = std::function<std::unique_ptr<ISegmenter>()>;

    static SegmenterFactory& instance() {
        static SegmenterFactory f;
        return f;
    }

    void registerMethod(const std::string& key, Creator creator) {
        creators_[key] = creator;
        order_.push_back(key);
    }

    std::unique_ptr<ISegmenter> create(const std::string& key) const {
        auto it = creators_.find(key);
        if (it == creators_.end()) {
            throw std::runtime_error("Unknown segmentation method: " + key);
        }
        return it->second();
    }

    const std::vector<std::string>& availableMethods() const { return order_; }

private:
    std::map<std::string, Creator> creators_;
    std::vector<std::string> order_;
};

} // namespace ctseg
