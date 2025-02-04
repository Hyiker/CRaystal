#pragma once
#include <cstdint>
#include <memory>

#include "Core/Macros.h"
namespace indicators {
class ProgressBar;
}

namespace CRay {

class CRAYSTAL_API Progress {
   public:
    class ProgressIterator {
       private:
        int mCurrent;
        Progress* mpProgress;

       public:
        using iterator_category = std::input_iterator_tag;
        using value_type = int;
        using difference_type = int;
        using pointer = int*;
        using reference = int&;

        ProgressIterator(int current, Progress* pProgress)
            : mCurrent(current), mpProgress(pProgress) {}

        int operator*() const { return mCurrent; }

        ProgressIterator& operator++();

        bool operator!=(const ProgressIterator& other) const {
            return mCurrent != other.mCurrent;
        }
    };

    Progress(int total, const std::string& desc);

    void setProgress(int current);

    ProgressIterator begin() { return ProgressIterator(0, this); }
    ProgressIterator end() { return ProgressIterator(mTotal, nullptr); }

    // Declare this to make unique_ptr work with fwd declare.
    ~Progress();

   private:
    int mTotal;
    std::unique_ptr<indicators::ProgressBar> mpBar;
};

}  // namespace CRay
