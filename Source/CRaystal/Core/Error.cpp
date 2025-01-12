#include "Error.h"
namespace CRay {

void throwException(const std::source_location& loc, std::string_view msg) {
    std::string fullMsg = fmt::format("{}\n\n{}:{} ({})", msg, loc.file_name(),
                                      loc.line(), loc.function_name());

    throw RuntimeError(fullMsg);
}

void reportAssertion(const std::source_location& loc, std::string_view cond,
                     std::string_view msg) {
    std::string fullMsg =
        fmt::format("Assertion failed: {}\n{}{}\n{}:{} ({})", cond, msg,
                    msg.empty() ? "" : "\n", loc.file_name(), loc.line(),
                    loc.function_name());

    throw AssertionError(fullMsg);
}

}  // namespace CRay
