#include "Progress.h"

#include <indicators/progress_bar.hpp>
namespace CRay {
using namespace indicators;
using ProgressIterator = Progress::ProgressIterator;
ProgressIterator& ProgressIterator::operator++() {
    ++mCurrent;
    if (mpProgress && mpProgress->mpBar) {
        mpProgress->setProgress(mCurrent);
    }
    return *this;
}

Progress::Progress(int total, const std::string& desc) : mTotal(total) {
    mpBar = std::make_unique<ProgressBar>(
        option::BarWidth{80}, option::Start{"["}, option::End{"]"},
        option::Fill{"="}, option::PrefixText{desc},
        option::ShowElapsedTime{true}, option::ShowRemainingTime{true},
        option::ShowPercentage{true}, option::MaxProgress{total});
}

void Progress::setProgress(int current) {
    mpBar->set_option(option::PostfixText{std::to_string(current) + "/" +
                                          std::to_string(mTotal)});
    mpBar->set_progress(current);
}

Progress::~Progress() {}

}  // namespace CRay
